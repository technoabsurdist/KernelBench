import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv-subtract-mish
fused_conv_subtract_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

__global__ void subtract_mish_kernel(float* output, float subtract_val_1, float subtract_val_2, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = output[idx] - subtract_val_1 - subtract_val_2;
        // Mish activation: x * tanh(softplus(x))
        float sp = x >= 20.0f ? x : (x <= -20.0f ? 0.0f : logf(expf(x) + 1.0f));
        float t = tanhf(sp);
        output[idx] = x * t;
    }
}

torch::Tensor fused_conv_subtract_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_val_1,
    float subtract_val_2,
    int kernel_size,
    int padding) {
    
    // Convolution using PyTorch's built-in function for simplicity
    auto conv_output = torch::conv2d(input, weight, bias, 1, padding, 1, 1);
    
    // Get output dimensions
    int batch_size = conv_output.size(0);
    int out_channels = conv_output.size(1);
    int height = conv_output.size(2);
    int width = conv_output.size(3);
    int size = batch_size * out_channels * height * width;
    
    // Launch kernel for subtract and Mish
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    subtract_mish_kernel<<<num_blocks, block_size>>>(
        conv_output.data_ptr<float>(), 
        subtract_val_1, 
        subtract_val_2, 
        size
    );
    
    return conv_output;
}
"""

fused_conv_subtract_mish_cpp_source = """
torch::Tensor fused_conv_subtract_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_val_1,
    float subtract_val_2,
    int kernel_size,
    int padding);
"""

# Compile the inline CUDA code for fused operation
fused_conv_subtract_mish = load_inline(
    name="fused_conv_subtract_mish",
    cpp_sources=fused_conv_subtract_mish_cpp_source,
    cuda_sources=fused_conv_subtract_mish_source,
    functions=["fused_conv_subtract_mish_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused convolution, subtraction, and Mish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.fused_op = fused_conv_subtract_mish
        self.padding = kernel_size // 2

    def forward(self, x):
        return self.fused_op.fused_conv_subtract_mish_cuda(
            x,
            self.conv.weight,
            self.conv.bias,
            self.subtract_value_1,
            self.subtract_value_2,
            self.conv.kernel_size[0],
            self.padding
        )

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]