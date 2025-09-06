import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: element-wise add + x * hardswish(x)
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For fmaxf, fminf

// Device function for HardSwish activation: x * ReLU6(x + 3) / 6
__device__ __forceinline__ float hardswish(float x) {
    return x * fmaxf(0.0f, fminf(6.0f, x + 3.0f)) / 6.0f;
}

// Fused kernel that performs:
// 1. Element-wise addition: tmp = conv_out + add_input
// 2. Custom activation: out = tmp * hardswish(tmp)
__global__ void fused_add_hardswish_mul_kernel(const float* conv_out, const float* add_input, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Step 1: Element-wise addition
        float tmp = conv_out[idx] + add_input[idx];

        // Step 2: Calculate hardswish(tmp)
        float hs_val = hardswish(tmp);

        // Step 3: Multiply the sum by its hardswish value
        out[idx] = tmp * hs_val;
    }
}

// C++ wrapper function to be called from Python
torch::Tensor fused_op_cuda(torch::Tensor conv_out, torch::Tensor add_input) {
    // Input validation
    TORCH_CHECK(conv_out.is_cuda(), "Input 'conv_out' must be a CUDA tensor");
    TORCH_CHECK(add_input.is_cuda(), "Input 'add_input' must be a CUDA tensor");
    TORCH_CHECK(conv_out.is_contiguous(), "Input 'conv_out' must be contiguous");
    TORCH_CHECK(add_input.is_contiguous(), "Input 'add_input' must be contiguous");
    TORCH_CHECK(conv_out.sizes() == add_input.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(conv_out.scalar_type() == torch::kFloat32, "Input 'conv_out' must be a float32 tensor");
    TORCH_CHECK(add_input.scalar_type() == torch::kFloat32, "Input 'add_input' must be a float32 tensor");

    // Prepare output tensor
    auto out = torch::empty_like(conv_out);
    auto size = conv_out.numel();

    if (size == 0) {
        return out;
    }

    // Kernel launch configuration
    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Launch the kernel
    fused_add_hardswish_mul_kernel<<<num_blocks, block_size>>>(
        conv_out.data_ptr<float>(),
        add_input.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );

    // Check for errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# C++ source for the function signature, required by load_inline
fused_op_cpp_source = "torch::Tensor fused_op_cuda(torch::Tensor conv_out, torch::Tensor add_input);"

# JIT compile the CUDA and C++ code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that replaces the sequence of element-wise operations (add, hardswish, mul)
    with a single, fused custom CUDA kernel. The ConvTranspose3d layer is kept as is,
    as it is already highly optimized by cuDNN.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        # Keep the standard ConvTranspose3d layer, as it's difficult to beat cuDNN's performance
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        
        # This bias parameter was unused in the original model's forward pass.
        # We keep it to maintain architectural consistency.
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Store the compiled custom CUDA function
        self.fused_op = fused_op

    def forward(self, x, add_input):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
            add_input (torch.Tensor): Input tensor to be added after transposed convolution, of shape (batch_size, out_channels, D', H', W').
        Returns:
            torch.Tensor: Output tensor after the fused operation.
        """
        # 1. Apply the standard, highly optimized ConvTranspose3d layer
        conv_out = self.conv_transpose(x)
        
        # 2. Apply the custom fused kernel for `(conv_out + add_input) * hardswish(conv_out + add_input)`
        # This replaces two separate PyTorch element-wise operations with a single kernel launch,
        # reducing kernel launch overhead and memory bandwidth usage.
        output = self.fused_op.fused_op_cuda(conv_out, add_input)
        
        return output