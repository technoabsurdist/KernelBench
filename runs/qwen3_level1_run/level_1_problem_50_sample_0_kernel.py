import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void conv_forward_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * out_height * out_width;
    
    if (out_idx >= total_outputs) return;
    
    int w = out_idx % out_width;
    out_idx /= out_width;
    int h = out_idx % out_height;
    out_idx /= out_height;
    int c = out_idx % out_channels;
    int n = out_idx / out_channels;
    
    float sum = 0.0f;
    
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = h * stride - padding + kh;
            int iw = w * stride - padding + kw;
            
            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                for (int ic = 0; ic < in_channels; ++ic) {
                    int input_idx = n * (in_channels * in_height * in_width) +
                                   ic * (in_height * in_width) +
                                   ih * in_width + iw;
                                   
                    int weight_idx = c * (in_channels * kernel_size * kernel_size) +
                                    ic * (kernel_size * kernel_size) +
                                    kh * kernel_size + kw;
                                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    sum += bias[c];
    output[out_idx * (out_channels * out_height * out_width) + 
            c * (out_height * out_width) + 
            h * out_width + w] = sum;
}

torch::Tensor custom_conv_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    // Input shape: (N, C_in, H_in, W_in)
    // Weight shape: (C_out, C_in, K, K)
    // Output shape: (N, C_out, H_out, W_out)
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int stride = 4;
    int padding = 2;
    
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    int total_outputs = batch_size * out_channels * out_height * out_width;
    const int num_blocks = (total_outputs + block_size - 1) / block_size;
    
    conv_forward_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding
    );
    
    return output;
}
"""

conv_cpp_source = """
torch::Tensor custom_conv_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code for convolution
custom_conv = load_inline(
    name="custom_conv",
    cpp_sources=conv_cpp_source,
    cuda_sources=conv_source,
    functions=["custom_conv_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1_weight = nn.Parameter(torch.randn(96, 3, 11, 11))
        self.conv1_bias = nn.Parameter(torch.randn(96))
        self.custom_conv = custom_conv
        
    def forward(self, x):
        return self.custom_conv.custom_conv_cuda(x, self.conv1_weight, self.conv1_bias)