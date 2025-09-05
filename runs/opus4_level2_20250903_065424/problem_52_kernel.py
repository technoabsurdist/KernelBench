import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Mish activation
mish_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float mish_activation(float x) {
    // Compute softplus: log(1 + exp(x))
    float softplus = logf(1.0f + expf(x));
    // Compute tanh(softplus)
    float tanh_softplus = tanhf(softplus);
    // Return x * tanh(softplus)
    return x * tanh_softplus;
}

__global__ void mish_activation_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = mish_activation(input[idx]);
    }
}

torch::Tensor mish_activation_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    mish_activation_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

mish_activation_cpp_source = (
    "torch::Tensor mish_activation_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for Mish activation
mish_activation = load_inline(
    name="mish_activation",
    cpp_sources=mish_activation_cpp_source,
    cuda_sources=mish_activation_source,
    functions=["mish_activation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for fused Mish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.mish_activation = mish_activation

    def forward(self, x):
        x = self.conv(x)
        x = self.mish_activation.mish_activation_cuda(x.contiguous())
        x = self.bn(x)
        return x

batch_size = 64
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]