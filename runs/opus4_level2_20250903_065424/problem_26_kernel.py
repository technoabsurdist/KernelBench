import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused add + hardswish
fused_add_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float hardswish_func(float x) {
    float x_plus_3 = x + 3.0f;
    float relu6 = fminf(fmaxf(x_plus_3, 0.0f), 6.0f);
    return x * relu6 / 6.0f;
}

__global__ void fused_add_hardswish_kernel(
    const float* conv_out,
    const float* add_input, 
    float* output,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = conv_out[idx] + add_input[idx];
        output[idx] = sum * hardswish_func(sum);
    }
}

torch::Tensor fused_add_hardswish_cuda(torch::Tensor conv_out, torch::Tensor add_input) {
    auto size = conv_out.numel();
    auto output = torch::empty_like(conv_out);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_add_hardswish_kernel<<<num_blocks, block_size>>>(
        conv_out.data_ptr<float>(),
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

fused_add_hardswish_cpp_source = """
torch::Tensor fused_add_hardswish_cuda(torch::Tensor conv_out, torch::Tensor add_input);
"""

# Compile the inline CUDA code
fused_add_hardswish = load_inline(
    name="fused_add_hardswish",
    cpp_sources=fused_add_hardswish_cpp_source,
    cuda_sources=fused_add_hardswish_source,
    functions=["fused_add_hardswish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
    Optimized with custom CUDA kernel for fused add + hardswish operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_add_hardswish = fused_add_hardswish

    def forward(self, x, add_input):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
            add_input (torch.Tensor): Input tensor to be added after transposed convolution, of shape (batch_size, out_channels, D, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W) after HardSwish activation.
        """
        x = self.conv_transpose(x)
        x = self.fused_add_hardswish.fused_add_hardswish_cuda(x, add_input)
        return x

batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W), torch.rand(batch_size, out_channels, D*stride, H*stride, W*stride)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]