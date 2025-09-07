import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv + div + leaky_relu
fused_conv_div_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void conv_div_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int pad,
    int stride,
    float divisor,
    float negative_slope
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * height * width;
    
    if (out_idx >= total_outputs) return;
    
    int w = out_idx % width;
    int h = (out_idx / width) % height;
    int c = (out_idx / (width * height)) % out_channels;
    int b = out_idx / (width * height * out_channels);
    
    float sum = 0.0f;
    
    for (int i = 0; i < in_channels; i++) {
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_h = h * stride + ky - pad;
                int in_w = w * stride + kx - pad;
                
                if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                    int in_idx = b * (in_channels * height * width) + 
                                i * (height * width) + 
                                in_h * width + in_w;
                                
                    int weight_idx = c * (in_channels * kernel_size * kernel_size) +
                                   i * (kernel_size * kernel_size) +
                                   ky * kernel_size + kx;
                                   
                    sum += input[in_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    sum += bias[c];
    sum /= divisor;
    
    if (sum < 0) {
        sum *= negative_slope;
    }
    
    output[out_idx] = sum;
}

torch::Tensor fused_conv_div_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor,
    int pad,
    int stride,
    float negative_slope
) {
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int height = input_sizes[2];
    int width = input_sizes[3];
    
    int out_channels = weight_sizes[0];
    int kernel_size = weight_sizes[2];
    
    int out_height = (height + 2 * pad - kernel_size) / stride + 1;
    int out_width = (width + 2 * pad - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int total_threads = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;
    
    conv_div_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        pad,
        stride,
        divisor,
        negative_slope
    );
    
    return output;
}
"""

fused_conv_div_relu_cpp_source = """
torch::Tensor fused_conv_div_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor,
    int pad,
    int stride,
    float negative_slope
);
"""

# Compile the inline CUDA code for fused conv + div + leaky_relu
fused_conv_div_relu = load_inline(
    name="fused_conv_div_relu",
    cpp_sources=fused_conv_div_relu_cpp_source,
    cuda_sources=fused_conv_div_relu_source,
    functions=["fused_conv_div_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused conv + div + leaky_relu operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.fused_op = fused_conv_div_relu

    def forward(self, x):
        return self.fused_op.fused_conv_div_relu_cuda(
            x,
            self.conv.weight,
            self.conv.bias,
            self.divisor,
            self.conv.padding[0],
            self.conv.stride[0],
            0.01
        )

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
divisor = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]