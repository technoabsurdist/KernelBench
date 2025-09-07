import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv + min + bias + scale
fused_conv_min_bias_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA kernel for fused operations: conv + min + bias + scale
__global__ void fused_conv_min_bias_scale_kernel(
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
    float constant_value,
    float scaling_factor
) {
    int out_h = (height + 2 * pad - kernel_size) / stride + 1;
    int out_w = (width + 2 * pad - kernel_size) / stride + 1;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_h * out_w;
    
    if (idx >= total_elements) return;
    
    int n = idx / (out_channels * out_h * out_w);
    int c_out = (idx / (out_h * out_w)) % out_channels;
    int h_out = (idx / out_w) % out_h;
    int w_out = idx % out_w;
    
    float sum = 0.0f;
    
    // Convolution computation
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h_out * stride - pad + kh;
                int w_in = w_out * stride - pad + kw;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = ((n * in_channels + c_in) * height + h_in) * width + w_in;
                    int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Apply min with constant
    sum = fminf(sum, constant_value);
    
    // Add bias
    sum += bias[c_out];
    
    // Scale
    sum *= scaling_factor;
    
    output[idx] = sum;
}

torch::Tensor fused_conv_min_bias_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int pad,
    int stride,
    float constant_value,
    float scaling_factor
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    
    int out_h = (height + 2 * pad - kernel_size) / stride + 1;
    int out_w = (width + 2 * pad - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    int total_elements = batch_size * out_channels * out_h * out_w;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_conv_min_bias_scale_kernel<<<num_blocks, block_size>>>(
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
        constant_value,
        scaling_factor
    );
    
    return output;
}
"""

fused_conv_min_bias_scale_cpp_source = """
torch::Tensor fused_conv_min_bias_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int pad,
    int stride,
    float constant_value,
    float scaling_factor
);
"""

# Compile the inline CUDA code
fused_conv_min_bias_scale = load_inline(
    name="fused_conv_min_bias_scale",
    cpp_sources=fused_conv_min_bias_scale_cpp_source,
    cuda_sources=fused_conv_min_bias_scale_source,
    functions=["fused_conv_min_bias_scale_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for conv + min + bias + scale operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.kernel_size = kernel_size
        self.constant_value = constant_value
        self.scaling_factor = scaling_factor
        self.pad = kernel_size // 2
        self.stride = 1
        self.fused_op = fused_conv_min_bias_scale

    def forward(self, x):
        return self.fused_op.fused_conv_min_bias_scale_cuda(
            x,
            self.conv_weight,
            self.bias,
            self.kernel_size,
            self.pad,
            self.stride,
            self.constant_value,
            self.scaling_factor
        )

batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]