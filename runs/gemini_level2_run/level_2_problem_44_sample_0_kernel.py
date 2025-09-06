import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused scalar multiplication and global average pooling
fused_mul_gap_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused kernel for:
// 1. Element-wise multiplication by a scalar
// 2. Global average pooling over spatial dimensions (H, W)
// This replaces (x * multiplier) followed by torch.mean(x, dim=[2, 3])
__global__ void fused_mul_gap_kernel(const float* input, float multiplier, float* output, int B, int C, int H, int W) {
    // Each block is responsible for one (batch, channel) pair.
    int b = blockIdx.x;
    int c = blockIdx.y;

    // Guard against out-of-bounds blocks
    if (b >= B || c >= C) {
        return;
    }

    // Shared memory for reduction within the block
    extern __shared__ float sdata[];

    float my_sum = 0.0f;
    int spatial_size = H * W;
    // Calculate the starting index for the current (b, c) plane
    int plane_offset = (b * C + c) * spatial_size;

    // Each thread in the block computes a partial sum over the spatial dimensions
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        my_sum += input[plane_offset + i];
    }

    sdata[threadIdx.x] = my_sum;
    __syncthreads();

    // Perform reduction in shared memory.
    // This is a standard parallel reduction algorithm.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 of each block writes the final result.
    if (threadIdx.x == 0) {
        float total_sum = sdata[0];
        // Apply multiplier and calculate the mean
        output[b * C + c] = (total_sum * multiplier) / (float)spatial_size;
    }
}

torch::Tensor fused_mul_gap_cuda(torch::Tensor input, float multiplier) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input tensor must be 4-dimensional (B, C, H, W)");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    const auto B = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);

    // Create the output tensor with shape (B, C, 1, 1)
    auto output = torch::empty({B, C, 1, 1}, input.options());

    // A block size of 256 is a reasonable default and a power of 2.
    const int block_size = 256;
    dim3 grid_dim(B, C);
    dim3 block_dim(block_size);

    // Shared memory size for the reduction
    size_t shared_mem_size = block_size * sizeof(float);

    // Launch the kernel
    fused_mul_gap_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        input.data_ptr<float>(),
        multiplier,
        output.data_ptr<float>(),
        B, C, H, W
    );
    
    // Check for errors after kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

fused_mul_gap_cpp_source = (
    "torch::Tensor fused_mul_gap_cuda(torch::Tensor input, float multiplier);"
)

# Compile the inline CUDA code
# This fuses the scalar multiplication and the two global average pooling operations
# The second pooling is redundant and is effectively removed.
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_mul_gap_cpp_source,
    cuda_sources=fused_mul_gap_source,
    functions=["fused_mul_gap_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, then a fused operation of 
    scalar multiplication and global average pooling using a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        # The ConvTranspose2d layer is kept as is, since its implementation is highly optimized (cuDNN)
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier
        # Store the compiled custom CUDA function
        self.fused_mul_gap = fused_op.fused_mul_gap_cuda

    def forward(self, x):
        x = self.conv_transpose(x)
        # Call the custom fused kernel which replaces:
        # x = x * self.multiplier
        # x = torch.mean(x, dim=[2, 3], keepdim=True)
        # x = torch.mean(x, dim=[2, 3], keepdim=True)
        x = self.fused_mul_gap(x, self.multiplier)
        return x