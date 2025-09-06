import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for fused bias subtraction and tanh activation
fused_bias_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel for element-wise (input - bias) followed by tanh activation.
// This kernel fuses two operations: bias subtraction and the tanh activation function.
// It handles the broadcasting of the bias tensor (shape [C, 1, 1]) to the input tensor (shape [N, C, H, W]).
__global__ void fused_bias_tanh_kernel(
    const float* input,
    const float* bias,
    float* output,
    int total_elements,
    int C,
    int H,
    int W) {

    // Use a grid-stride loop to ensure all elements are processed, regardless of grid size.
    // This is a robust pattern for CUDA kernels.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_elements;
         i += blockDim.x * gridDim.x) {

        // Calculate the channel index 'c' from the linear index 'i'.
        // The tensor layout is NCHW.
        // plane_size = H * W.
        // The index of the 2D plane (an HxW slice) is i / plane_size.
        // The channel index within the batch is (i / plane_size) % C.
        int plane_size = H * W;
        int c = (i / plane_size) % C;

        // Perform the fused operation:
        // 1. Subtract the broadcasted bias element (bias[c]).
        // 2. Apply the tanh activation function (tanhf for float).
        output[i] = tanhf(input[i] - bias[c]);
    }
}

// C++ wrapper function that PyTorch can call.
// It handles tensor validation, memory allocation, and launching the CUDA kernel.
torch::Tensor fused_bias_tanh_cuda(torch::Tensor input, torch::Tensor bias) {
    // Input validation checks
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on a CUDA device");
    TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on a CUDA device");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor (N, C, H, W)");
    TORCH_CHECK(bias.dim() == 3, "Bias must be a 3D tensor (C, 1, 1)");
    TORCH_CHECK(input.size(1) == bias.size(0), "Number of channels in input and bias must match");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias tensor must be contiguous");

    // Create an output tensor with the same shape and device as the input
    auto output = torch::empty_like(input);
    const int total_elements = input.numel();

    // Return early if there's nothing to process
    if (total_elements == 0) {
        return output;
    }

    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    // Configure and launch the kernel
    const int block_size = 256;
    // Calculate the number of blocks needed, capping at a reasonable limit.
    // The grid-stride loop in the kernel handles any number of elements.
    const int num_blocks = std::min((total_elements + block_size - 1) / block_size, 4096);

    fused_bias_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        C, H, W
    );
    
    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

# Define the C++ function signature for the JIT compiler
fused_bias_tanh_cpp_source = (
    "torch::Tensor fused_bias_tanh_cuda(torch::Tensor input, torch::Tensor bias);"
)

# Use torch's JIT compiler to build the custom CUDA operator
fused_op = load_inline(
    name="fused_bias_tanh",
    cpp_sources=fused_bias_tanh_cpp_source,
    cuda_sources=fused_bias_tanh_source,
    functions=["fused_bias_tanh_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, followed by a fused
    (bias subtraction + tanh activation) custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        # The transposed convolution is kept as a standard PyTorch operator
        # as its cuDNN implementation is highly optimized. Replicating this
        # performance is non-trivial.
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        
        # The bias is still a learnable parameter, same as the original model.
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Store the compiled custom operator for use in the forward pass.
        self.fused_bias_tanh = fused_op.fused_bias_tanh_cuda

    def forward(self, x):
        # 1. Apply the standard, highly-optimized transposed convolution
        x = self.conv_transpose(x)
        
        # 2. Apply the custom fused kernel for bias subtraction and tanh activation.
        # This fusion avoids creating an intermediate tensor for (x - self.bias)
        # and reduces kernel launch overhead, leading to a speedup.
        x = self.fused_bias_tanh(x, self.bias)
        
        return x