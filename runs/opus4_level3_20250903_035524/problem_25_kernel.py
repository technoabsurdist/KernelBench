import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused depthwise conv + batchnorm
fused_depthwise_conv_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_depthwise_conv_bn_kernel(
    const float* input, const float* weight, const float* bn_weight, 
    const float* bn_bias, const float* bn_mean, const float* bn_var,
    float* output, int batch_size, int channels, int height, int width,
    int kernel_size, int padding, float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx >= total_elements) return;
    
    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int b = idx / (width * height * channels);
    
    float sum = 0.0f;
    int half_kernel = kernel_size / 2;
    
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int ih = h + kh - padding;
            int iw = w + kw - padding;
            
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                int input_idx = b * channels * height * width + c * height * width + ih * width + iw;
                int weight_idx = c * kernel_size * kernel_size + kh * kernel_size + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    // Apply batch normalization
    float normalized = (sum - bn_mean[c]) / sqrtf(bn_var[c] + eps);
    output[idx] = normalized * bn_weight[c] + bn_bias[c];
}

torch::Tensor fused_depthwise_conv_bn_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bn_weight,
    torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var,
    int kernel_size, int padding, float eps) {
    
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto output = torch::zeros_like(input);
    
    int total_elements = batch_size * channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_depthwise_conv_bn_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(), bn_var.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, channels, height, width,
        kernel_size, padding, eps);
    
    return output;
}
"""

fused_depthwise_conv_bn_cpp_source = """
torch::Tensor fused_depthwise_conv_bn_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bn_weight,
    torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var,
    int kernel_size, int padding, float eps);
"""

# Custom CUDA kernel for channel shuffle
channel_shuffle_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void channel_shuffle_kernel(
    const float* input, float* output,
    int batch_size, int groups, int channels_per_group,
    int height, int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * groups * channels_per_group * height * width;
    
    if (idx >= total_elements) return;
    
    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % (groups * channels_per_group);
    int b = idx / (width * height * groups * channels_per_group);
    
    // Compute source channel index after shuffle
    int group_idx = c % groups;
    int channel_in_group = c / groups;
    int src_c = group_idx * channels_per_group + channel_in_group;
    
    int src_idx = b * groups * channels_per_group * height * width +
                  src_c * height * width + h * width + w;
    
    output[idx] = input[src_idx];
}

torch::Tensor channel_shuffle_cuda(torch::Tensor input, int groups) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    int channels_per_group = channels / groups;
    auto output = torch::zeros_like(input);
    
    int total_elements = batch_size * channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    channel_shuffle_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, groups, channels_per_group, height, width);
    
    return output;
}
"""

channel_shuffle_cpp_source = """
torch::Tensor channel_shuffle_cuda(torch::Tensor input, int groups);
"""

# Custom CUDA kernel for fused conv + bn + relu
fused_conv_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv1x1_bn_relu_kernel(
    const float* input, const float* weight, const float* bn_weight,
    const float* bn_bias, const float* bn_mean, const float* bn_var,
    float* output, int batch_size, int in_channels, int out_channels,
    int height, int width, int groups, float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * height * width;
    
    if (idx >= total_elements) return;
    
    int w = idx % width;
    int h = (idx / width) % height;
    int oc = (idx / (width * height)) % out_channels;
    int b = idx / (width * height * out_channels);
    
    int group = oc / (out_channels / groups);
    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;
    int oc_in_group = oc % out_channels_per_group;
    
    float sum = 0.0f;
    for (int ic = 0; ic < in_channels_per_group; ic++) {
        int global_ic = group * in_channels_per_group + ic;
        int input_idx = b * in_channels * height * width + 
                       global_ic * height * width + h * width + w;
        int weight_idx = oc * in_channels_per_group + ic;
        sum += input[input_idx] * weight[weight_idx];
    }
    
    // Apply batch normalization
    float normalized = (sum - bn_mean[oc]) / sqrtf(bn_var[oc] + eps);
    float bn_out = normalized * bn_weight[oc] + bn_bias[oc];
    
    // Apply ReLU
    output[idx] = fmaxf(0.0f, bn_out);
}

torch::Tensor fused_conv1x1_bn_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bn_weight,
    torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var,
    int groups, float eps) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, 
                              input.options());
    
    int total_elements = batch_size * out_channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_conv1x1_bn_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(), bn_var.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        height, width, groups, eps);
    
    return output;
}
"""

fused_conv_bn_relu_cpp_source = """
torch::Tensor fused_conv1x1_bn_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bn_weight,
    torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var,
    int groups, float eps);
"""

# Load custom CUDA kernels
fused_depthwise_conv_bn = load_inline(
    name="fused_depthwise_conv_bn",
    cpp_sources=fused_depthwise_conv_bn_cpp_source,
    cuda_sources=fused_depthwise_conv_bn_source,
    functions=["fused_depthwise_conv_bn_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

channel_shuffle_cuda_module = load_inline(
    name="channel_shuffle",
    cpp_sources=channel_shuffle_cpp_source,
    cuda_sources=channel_shuffle_source,
    functions=["channel_shuffle_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

fused_conv_bn_relu = load_inline(
    name="fused_conv_bn_relu",
    cpp_sources=fused_conv_bn_relu_cpp_source,
    cuda_sources=fused_conv_bn_relu_source,
    functions=["fused_conv1x1_bn_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ModelNew, self).__init__()
        
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        
        # First 1x1 group convolution
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # Depthwise 3x3 convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # Second 1x1 group convolution
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # CUDA modules
        self.fused_depthwise_conv_bn = fused_depthwise_conv_bn
        self.channel_shuffle_cuda = channel_shuffle_cuda_module
        self.fused_conv_bn_relu = fused_conv_bn_relu
    
    def forward(self, x):
        # Fused conv1 + bn1 + relu
        out = self.fused_conv_bn_relu.fused_conv1x1_bn_relu_cuda(
            x.contiguous(), 
            self.conv1.weight.view(self.mid_channels, -1).contiguous(),
            self.bn1.weight, self.bn1.bias,
            self.bn1.running_mean, self.bn1.running_var,
            self.groups, self.bn1.eps
        )
        
        # Fused depthwise conv2 + bn2
        out = self.fused_depthwise_conv_bn.fused_depthwise_conv_bn_cuda(
            out.contiguous(),
            self.conv2.weight.squeeze().contiguous(),
            self.bn2.weight, self.bn2.bias,
            self.bn2.running_mean, self.bn2.running_var,
            3, 1, self.bn2.eps
        )
        
        # Optimized channel shuffle
        out = self.channel_shuffle_cuda.channel_shuffle_cuda(out.contiguous(), self.groups)
        
        # conv3 + bn3 + relu
        out = F.relu(self.bn3(self.conv3(out)))
        
        # Residual connection
        out += self.shortcut(x)
        return out


batch_size = 10
input_channels = 240
out_channels = 480
groups = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, input_channels, height, width).cuda()]

def get_init_inputs():
    return [input_channels, out_channels, groups]