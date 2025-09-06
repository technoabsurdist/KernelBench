import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation:
# bias_add -> clamp -> scale -> clamp -> divide
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_op_kernel(
    const float* input,
    const float* bias,
    float* output,
    const float scaling_factor,
    const int total_elements,
    const int C,
    const int H,
    const int W) {

    const int HW = H * W;

    // Using a grid-stride loop for robust handling of any input size
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += blockDim.x * gridDim.x) {

        // Calculate channel index for bias lookup.
        // The bias tensor has shape (C, 1, 1) and is broadcasted over H and W.
        // The memory layout is NCHW.
        int c = (idx / HW) % C;

        // 1. Add bias
        float val = input[idx] + bias[c];

        // 2. Clamp [0, 1]
        val = fminf(fmaxf(val, 0.0f), 1.0f);

        // 3. Scale
        val = val * scaling_factor;

        // 4. Clamp [0, 1]
        val = fminf(fmaxf(val, 0.0f), 1.0f);

        // 5. Divide
        val = val / scaling_factor;

        output[idx] = val;
    }
}

// C++ wrapper function to be called from Python
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor) {

    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(bias.size(0) == input.size(1), "Bias channels must match input channels");

    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    const auto total_elements = input.numel();

    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    // Launch the kernel
    fused_op_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        total_elements,
        C,
        H,
        W
    );

    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

# C++ source for the function signature
fused_op_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor);"
)

# Compile the inline CUDA code using JIT
fused_op_module = load_inline(
    name="fused_op_module",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, then a fused operation of
    (bias_add -> clamp -> scale -> clamp -> divide) using a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        # Keep the highly optimized ConvTranspose2d from PyTorch
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        
        # The bias and scaling factor are used by the custom kernel
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor
        
        # Assign the compiled CUDA function
        self.fused_op = fused_op_module.fused_op_cuda

    def forward(self, x):
        # 1. Perform transposed convolution
        x = self.conv_transpose(x)
        
        # 2. Apply the fused sequence of operations using the custom CUDA kernel
        x = self.fused_op(x, self.bias, self.scaling_factor)
        
        return x