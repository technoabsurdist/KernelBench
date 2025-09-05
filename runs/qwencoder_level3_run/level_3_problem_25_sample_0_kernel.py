import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv + batchnorm + relu
conv_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void conv_bn_relu_kernel(
    const float* input,
    const float* weight,
    const float* bn_weight,
    const float* bn_bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    float eps
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (out_idx >= total_elements) return;
    
    int n = out_idx / (out_channels * out_height * out_width);
    int c = (out_idx / (out_height * out_width)) % out_channels;
    int h = (out_idx / out_width) % out_height;
    int w = out_idx % out_width;
    
    float sum = 0.0f;
    
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = h * stride + kh - padding;
            int iw = w * stride + kw - padding;
            
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                for (int ic = 0; ic < in_channels; ++ic) {
                    int input_idx = n * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                    int weight_idx = c * (in_channels * kernel_size * kernel_size) + ic * (kernel_size * kernel_size) + kh * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    float normalized = (sum - running_mean[c]) / sqrtf(running_var[c] + eps);
    float bn_result = normalized * bn_weight[c] + bn_bias[c];
    
    // ReLU
    output[out_idx] = fmaxf(0.0f, bn_result);
}

torch::Tensor conv_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    int kernel_size,
    int stride,
    int padding,
    float eps
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_elements = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv_bn_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        eps
    );
    
    return output;
}
"""

conv_bn_relu_cpp_source = """
torch::Tensor conv_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    int kernel_size,
    int stride,
    int padding,
    float eps
);
"""

# Define the custom CUDA kernel for channel shuffle
channel_shuffle_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void channel_shuffle_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx >= total_elements) return;
    
    int channels_per_group = channels / groups;
    
    int n = idx / (channels * height * width);
    int c = (idx / (height * width)) % channels;
    int h = (idx / width) % height;
    int w = idx % width;
    
    int group_idx = c / channels_per_group;
    int channel_in_group = c % channels_per_group;
    
    int new_c = channel_in_group * groups + group_idx;
    int output_idx = n * (channels * height * width) + new_c * (height * width) + h * width + w;
    
    output[output_idx] = input[idx];
}

torch::Tensor channel_shuffle_cuda(torch::Tensor input, int groups) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    auto output = torch::zeros_like(input);
    
    int total_elements = batch_size * channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    channel_shuffle_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        groups
    );
    
    return output;
}
"""

channel_shuffle_cpp_source = """
torch::Tensor channel_shuffle_cuda(torch::Tensor input, int groups);
"""

# Define the custom CUDA kernel for element-wise addition (residual connection)
residual_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void residual_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor residual_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    residual_add_kernel<<<num_blocks, block_size>>>(
        a.data_ptr<float>(), 
        b.data_ptr<float>(), 
        out.data_ptr<float>(), 
        size
    );

    return out;
}
"""

residual_add_cpp_source = """
torch::Tensor residual_add_cuda(torch::Tensor a, torch::Tensor b);
"""

# Compile the inline CUDA code
conv_bn_relu = load_inline(
    name="conv_bn_relu",
    cpp_sources=conv_bn_relu_cpp_source,
    cuda_sources=conv_bn_relu_source,
    functions=["conv_bn_relu_cuda"],
    verbose=False,
)

channel_shuffle = load_inline(
    name="channel_shuffle",
    cpp_sources=channel_shuffle_cpp_source,
    cuda_sources=channel_shuffle_source,
    functions=["channel_shuffle_cuda"],
    verbose=False,
)

residual_add = load_inline(
    name="residual_add",
    cpp_sources=residual_add_cpp_source,
    cuda_sources=residual_add_source,
    functions=["residual_add_cuda"],
    verbose=False,
)

class ChannelShuffleCustom(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffleCustom, self).__init__()
        self.groups = groups
        self.channel_shuffle = channel_shuffle

    def forward(self, x):
        return self.channel_shuffle.channel_shuffle_cuda(x, self.groups)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ModelNew, self).__init__()
        
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        # First 1x1 group convolution with BN and ReLU
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # Depthwise 3x3 convolution with BN (no ReLU here)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # Second 1x1 group convolution with BN and ReLU
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Custom channel shuffle
        self.shuffle = ChannelShuffleCustom(groups)
        
        # Shortcut connection
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(out_channels)
        
        # Custom operators
        self.conv_bn_relu_op = conv_bn_relu
        self.residual_add_op = residual_add
        
        # Store parameters for custom kernels
        self.eps = 1e-5

    def forward(self, x):
        # First conv + bn + relu
        out = self.conv_bn_relu_op.conv_bn_relu_cuda(
            x, 
            self.conv1.weight, 
            self.bn1.weight, 
            self.bn1.bias, 
            self.bn1.running_mean, 
            self.bn1.running_var, 
            1, 1, 0, self.eps
        )
        
        # Depthwise conv + bn (no relu)
        conv2_out = F.conv2d(out, self.conv2.weight, None, 1, 1, 1, out.size(1))
        out = F.batch_norm(conv2_out, self.bn2.running_mean, self.bn2.running_var, 
                           self.bn2.weight, self.bn2.bias, training=self.training)
        
        # Channel shuffle
        out = self.shuffle(out)
        
        # Second conv + bn + relu
        out = self.conv_bn_relu_op.conv_bn_relu_cuda(
            out, 
            self.conv3.weight, 
            self.bn3.weight, 
            self.bn3.bias, 
            self.bn3.running_mean, 
            self.bn3.running_var, 
            1, 1, 0, self.eps
        )
        
        # Shortcut
        if hasattr(self, 'shortcut_conv'):
            shortcut = self.conv_bn_relu_op.conv_bn_relu_cuda(
                x, 
                self.shortcut_conv.weight, 
                self.shortcut_bn.weight, 
                self.shortcut_bn.bias, 
                self.shortcut_bn.running_mean, 
                self.shortcut_bn.running_var, 
                1, 1, 0, self.eps
            )
        else:
            shortcut = x
        
        # Residual addition
        out = self.residual_add_op.residual_add_cuda(out, shortcut)
        
        return out