import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv + batch norm + relu
conv_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void conv3x3_kernel(const float* input, const float* weight, float* output,
                               int batch, int in_channels, int in_height, int in_width,
                               int out_channels, int out_height, int out_width,
                               int stride, int padding) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch * out_channels * out_height * out_width) return;

    int tmp = out_idx;
    int w_out = tmp % out_width;
    tmp /= out_width;
    int h_out = tmp % out_height;
    tmp /= out_height;
    int c_out = tmp % out_channels;
    int n = tmp / out_channels;

    float sum = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;
                
                if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                    int input_idx = ((n * in_channels + c_in) * in_height + h_in) * in_width + w_in;
                    int weight_idx = ((c_out * in_channels + c_in) * 3 + kh) * 3 + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    output[out_idx] = sum;
}

__global__ void batch_norm_relu_kernel(const float* input, const float* weight, const float* bias,
                                      const float* running_mean, const float* running_var,
                                      float* output, int size, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float mean = running_mean[idx % (size / gridDim.x)];
        float var = running_var[idx % (size / gridDim.x)];
        float w = weight[idx % (size / gridDim.x)];
        float b = bias[idx % (size / gridDim.x)];
        
        float normalized = (input[idx] - mean) / sqrtf(var + eps);
        float bn_out = w * normalized + b;
        output[idx] = fmaxf(0.0f, bn_out);  // ReLU
    }
}

torch::Tensor conv_bn_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bn_weight,
                               torch::Tensor bn_bias, torch::Tensor running_mean, torch::Tensor running_var,
                               int stride, int padding) {
    auto batch = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    auto out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    auto out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto conv_out = torch::zeros({batch, out_channels, out_height, out_width}, 
                                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int conv_block_size = 256;
    const int conv_num_blocks = (batch * out_channels * out_height * out_width + conv_block_size - 1) / conv_block_size;
    
    conv3x3_kernel<<<conv_num_blocks, conv_block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), conv_out.data_ptr<float>(),
        batch, in_channels, in_height, in_width, out_channels, out_height, out_width, stride, padding
    );
    
    auto final_out = torch::zeros_like(conv_out);
    const int bn_block_size = 256;
    const int bn_num_blocks = (batch * out_channels * out_height * out_width + bn_block_size - 1) / bn_block_size;
    
    batch_norm_relu_kernel<<<bn_num_blocks, bn_block_size>>>(
        conv_out.data_ptr<float>(), bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        final_out.data_ptr<float>(), batch * out_channels * out_height * out_width, 1e-5
    );
    
    return final_out;
}
"""

conv_bn_relu_cpp_source = """
torch::Tensor conv_bn_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bn_weight,
                               torch::Tensor bn_bias, torch::Tensor running_mean, torch::Tensor running_var,
                               int stride, int padding);
"""

# Compile the inline CUDA code for fused conv + batch norm + relu
conv_bn_relu = load_inline(
    name="conv_bn_relu",
    cpp_sources=conv_bn_relu_cpp_source,
    cuda_sources=conv_bn_relu_source,
    functions=["conv_bn_relu_cuda"],
    verbose=False,
)

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size
    );

    return out;
}
"""

elementwise_add_cpp_source = """
torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);
"""

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride
        
        # Store custom CUDA functions
        self.conv_bn_relu = conv_bn_relu
        self.elementwise_add = elementwise_add

    def forward(self, x):
        identity = x

        # Fused conv1 + bn1 + relu
        out = self.conv_bn_relu.conv_bn_relu_cuda(
            x, self.conv1.weight, self.bn1.weight, self.bn1.bias,
            self.bn1.running_mean, self.bn1.running_var, self.stride, 1
        )

        # Fused conv2 + bn2 (without relu)
        conv_out = self.conv_bn_relu.conv_bn_relu_cuda(
            out, self.conv2.weight, self.bn2.weight, self.bn2.bias,
            self.bn2.running_mean, self.bn2.running_var, 1, 1
        )
        # Apply relu separately since we need to add before the final relu
        out = F.relu(conv_out)

        if self.downsample is not None:
            # Apply downsample conv + bn
            downsample_conv = self.downsample[0]
            downsample_bn = self.downsample[1]
            identity = self.conv_bn_relu.conv_bn_relu_cuda(
                x, downsample_conv.weight, downsample_bn.weight, downsample_bn.bias,
                downsample_bn.running_mean, downsample_bn.running_var, self.stride, 0
            )

        # Element-wise addition
        out = self.elementwise_add.elementwise_add_cuda(out, identity)
        
        # Final relu
        out = F.relu(out)

        return out