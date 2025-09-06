import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel source for a fused double Mish activation
# This kernel applies the Mish function twice in a row without intermediate
# writes to global memory, reducing memory bandwidth and improving performance.
# Mish(x) = x * tanh(softplus(x))
# softplus(x) = log(1 + exp(x))
double_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel to apply Mish activation twice in a fused manner
__global__ void double_mish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // First Mish application
        float x1 = input[idx];
        float softplus_x1 = logf(1.0f + expf(x1));
        float mish_x1 = x1 * tanhf(softplus_x1);

        // Second Mish application on the result of the first
        float x2 = mish_x1;
        float softplus_x2 = logf(1.0f + expf(x2));
        float mish_x2 = x2 * tanhf(softplus_x2);

        output[idx] = mish_x2;
    }
}

// C++ wrapper function to launch the CUDA kernel from PyTorch
torch::Tensor double_mish_cuda(torch::Tensor input) {
    // Ensure input is a contiguous CUDA tensor of type float
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input tensor must be of type float32");

    auto output = torch::empty_like(input);
    auto size = input.numel();

    if (size == 0) {
        return output;
    }

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    double_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    // Check for any CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

# C++ source for the function signature
double_mish_cpp_source = "torch::Tensor double_mish_cuda(torch::Tensor input);"

# JIT compile the inline CUDA code. This is done once when the module is imported.
double_mish_op = load_inline(
    name="double_mish_op",
    cpp_sources=double_mish_cpp_source,
    cuda_sources=double_mish_source,
    functions=["double_mish_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses two Mish activations into a single custom CUDA kernel.
    The convolution operation is left as the standard PyTorch implementation, as it is
    already highly optimized (typically using cuDNN). The main optimization here is
    the fusion of the two subsequent element-wise activation functions.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # The custom op is loaded and assigned for use in the forward pass
        self.double_mish = double_mish_op.double_mish_cuda

    def forward(self, x):
        x = self.conv(x)
        # Apply the fused double Mish operation.
        # We call .contiguous() to ensure the memory layout is compatible with the
        # custom C++/CUDA kernel, which expects a dense C-style array.
        x = self.double_mish(x.contiguous())
        return x