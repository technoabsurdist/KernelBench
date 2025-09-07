import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
conv_kernel_source = """
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
    
    int w_out = out_idx % out_width;
    out_idx /= out_width;
    int h_out = out_idx % out_height;
    out_idx /= out_height;
    int c_out = out_idx % out_channels;
    int b_out = out_idx / out_channels;
    
    float sum = 0.0f;
    
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h_in = h_out * stride - padding + kh;
            int w_in = w_out * stride - padding + kw;
            
            if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    int input_idx = b_out * (in_channels * in_height * in_width) +
                                   c_in * (in_height * in_width) +
                                   h_in * in_width + w_in;
                                   
                    int weight_idx = c_out * (in_channels * kernel_size * kernel_size) +
                                    c_in * (kernel_size * kernel_size) +
                                    kh * kernel_size + kw;
                                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    sum += bias[c_out];
    output[out_idx * (out_height * out_width) + h_out * out_width + w_out] = sum;
}

torch::Tensor custom_conv2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int in_height = input_sizes[2];
    int in_width = input_sizes[3];
    
    int out_channels = weight_sizes[0];
    int kernel_size = weight_sizes[2];
    
    int stride = 4;
    int padding = 2;
    
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int threads_per_block = 256;
    const int total_threads = batch_size * out_channels * out_height * out_width;
    const int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    conv_forward_kernel<<<num_blocks, threads_per_block>>>(
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
torch::Tensor custom_conv2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code for convolution
custom_conv = load_inline(
    name="custom_conv",
    cpp_sources=conv_cpp_source,
    cuda_sources=conv_kernel_source,
    functions=["custom_conv2d"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.custom_conv = custom_conv
        
    def forward(self, x):
        return self.custom_conv.custom_conv2d(x, self.conv1.weight, self.conv1.bias)