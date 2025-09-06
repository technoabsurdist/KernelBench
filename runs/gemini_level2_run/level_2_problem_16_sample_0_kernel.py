import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fusing Mish, Add, Hardtanh, and Scale operations
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For logf, expf, tanhf, fmaxf, fminf

__global__ void fused_op_kernel(const float* input, float* output, float add_value, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 1. Load data
        float x = input[idx];

        // 2. Mish activation: x * tanh(softplus(x))
        // softplus(x) = log(1 + exp(x))
        float softplus_val = logf(1.0f + expf(x));
        float tanh_val = tanhf(softplus_val);
        x = x * tanh_val;

        // 3. Add value
        x = x + add_value;

        // 4. Hardtanh activation: max(-1, min(1, x))
        x = fmaxf(-1.0f, fminf(1.0f, x));

        // 5. Scale
        x = x * scale;

        // 6. Store result
        output[idx] = x;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor input, float add_value, float scale) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    auto output = torch::empty_like(input);
    auto size = input.numel();

    if (size == 0) {
        return output;
    }

    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_op_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        add_value,
        scale,
        size
    );
    
    // Check for any CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor input, float add_value, float scale);"
)

# Compile the inline CUDA code for the fused operation
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses Mish, Add, Hardtanh, and Scale operations into a single CUDA kernel.
    The ConvTranspose2d operation remains as a standard PyTorch operator.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        # The transposed convolution is a complex operation, best left to the highly optimized cuDNN implementation.
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        
        # Store constants for the fused kernel
        self.add_value = add_value
        self.scale = scale
        
        # Store the compiled custom operator
        self.fused_op = fused_op

    def forward(self, x):
        # 1. Perform the transposed convolution using the standard PyTorch layer
        x = self.conv_transpose(x)
        
        # 2. Apply the fused sequence of operations using our custom CUDA kernel
        x = self.fused_op.fused_op_cuda(x, self.add_value, self.scale)
        
        return x