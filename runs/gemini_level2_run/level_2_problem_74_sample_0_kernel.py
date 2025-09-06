import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: LeakyReLU -> Multiply -> LeakyReLU
fused_leaky_mul_leaky_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom LeakyReLU function to be used inside the kernel
__device__ inline float leaky_relu_forward(float x, float negative_slope) {
    return x > 0 ? x : x * negative_slope;
}

__global__ void fused_leaky_mul_leaky_kernel(
    const float* input, 
    const float* multiplier, 
    float* output, 
    float negative_slope, 
    long long total_elements,
    int C,
    int DHW) // Stride for the channel dimension (D*H*W)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        // First LeakyReLU
        float val = leaky_relu_forward(input[idx], negative_slope);

        // Get channel index for broadcasting the multiplier
        // The multiplier is of shape (C), so we find which channel this element belongs to.
        int channel_idx = (idx / DHW) % C;

        // Multiply by the corresponding multiplier value
        val *= multiplier[channel_idx];

        // Second LeakyReLU
        output[idx] = leaky_relu_forward(val, negative_slope);
    }
}

torch::Tensor fused_leaky_mul_leaky_cuda(
    torch::Tensor input, 
    torch::Tensor multiplier, 
    float negative_slope) 
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(multiplier.is_cuda(), "Multiplier must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 5, "Input must be a 5D tensor");
    TORCH_CHECK(multiplier.dim() == 1, "Multiplier must be a 1D tensor");

    auto C = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);

    TORCH_CHECK(multiplier.size(0) == C, "Multiplier size must match input channel dimension");

    auto total_elements = input.numel();
    auto output = torch::empty_like(input);
    
    int DHW = D * H * W; // Stride for the channel dimension

    const int block_size = 256;
    // Use long long for num_blocks to avoid overflow for very large tensors
    const long long num_blocks = (total_elements + block_size - 1) / block_size;

    fused_leaky_mul_leaky_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        total_elements,
        C,
        DHW
    );
    
    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

fused_leaky_mul_leaky_cpp_source = (
    "torch::Tensor fused_leaky_mul_leaky_cuda(torch::Tensor input, torch::Tensor multiplier, float negative_slope);"
)

# Compile the inline CUDA code for the fused operation
fused_leaky_mul_leaky = load_inline(
    name="fused_leaky_mul_leaky",
    cpp_sources=fused_leaky_mul_leaky_cpp_source,
    cuda_sources=fused_leaky_mul_leaky_source,
    functions=["fused_leaky_mul_leaky_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies a fused (LeakyReLU -> Multiply -> LeakyReLU) kernel,
    and performs a max pooling operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.max_pool = nn.MaxPool3d(kernel_size=2)
        self.negative_slope = 0.2

        # The fused kernel is loaded and stored as an attribute
        self.fused_op = fused_leaky_mul_leaky

    def forward(self, x):
        x = self.conv_transpose(x)
        
        # Call the custom fused CUDA kernel.
        # The multiplier parameter has shape (C, 1, 1, 1), but the kernel expects a 1D tensor of shape (C).
        # We use .squeeze() to remove the singleton dimensions.
        x = self.fused_op.fused_leaky_mul_leaky_cuda(x, self.multiplier.squeeze(), self.negative_slope)
        
        x = self.max_pool(x)
        return x