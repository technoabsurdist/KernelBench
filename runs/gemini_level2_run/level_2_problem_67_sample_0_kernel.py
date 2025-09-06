import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GELU and global average pooling
fused_gelu_avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// GELU approximation using tanh, same as in PyTorch
__device__ __forceinline__ float gelu_activation(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

// Kernel to perform fused GELU and global average pooling
__global__ void gelu_global_avg_pool_kernel(const float* input, float* output, int B, int C, int H, int W) {
    // Each block computes the average for one channel of one batch item.
    // The grid is launched as (C, B), so blockIdx.x is channel and blockIdx.y is batch.
    int channel_idx = blockIdx.x;
    int batch_idx = blockIdx.y;

    // Total elements in one feature map (H * W)
    int map_size = H * W;

    // Pointer to the start of the current feature map
    const float* current_map = input + (batch_idx * C + channel_idx) * map_size;

    // Shared memory for reduction within the block
    extern __shared__ float sdata[];

    // Each thread computes a partial sum
    float partial_sum = 0.0f;
    int tid = threadIdx.x;

    // Grid-stride loop to sum up elements in the feature map
    // Each thread processes multiple elements if map_size > blockDim.x
    for (int i = tid; i < map_size; i += blockDim.x) {
        // Apply GELU and add to the partial sum
        partial_sum += gelu_activation(current_map[i]);
    }

    sdata[tid] = partial_sum;
    __syncthreads();

    // Parallel reduction in shared memory.
    // This implementation assumes blockDim.x is a power of 2.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final result (sum / count) to the output tensor
    if (tid == 0) {
        // The output tensor is of shape (B, C)
        output[batch_idx * C + channel_idx] = sdata[0] / (float)map_size;
    }
}

torch::Tensor gelu_global_avg_pool_cuda(torch::Tensor input) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input tensor must be 4-dimensional (B, C, H, W)");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    const auto B = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);

    // Create the output tensor of shape (B, C)
    auto output = torch::zeros({B, C}, input.options());

    // Kernel launch configuration
    const int block_size = 256; // A common choice, must be a power of 2 for this reduction
    dim3 threads(block_size);
    dim3 blocks(C, B); // One block per (batch, channel) pair

    // Shared memory size: block_size * sizeof(float)
    int shared_mem_size = block_size * sizeof(float);

    // Launch the kernel
    gelu_global_avg_pool_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W
    );
    
    // Check for errors after kernel launch for robust error handling
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

fused_gelu_avg_pool_cpp_source = "torch::Tensor gelu_global_avg_pool_cuda(torch::Tensor input);"

# Compile the inline CUDA code
# This is done once when the module is imported
fused_op = load_inline(
    name="fused_gelu_avg_pool",
    cpp_sources=fused_gelu_avg_pool_cpp_source,
    cuda_sources=fused_gelu_avg_pool_source,
    functions=["gelu_global_avg_pool_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses GELU and global average pooling into a single custom CUDA kernel.
    This approach reduces memory bandwidth by eliminating the intermediate tensor between
    the GELU and pooling operations, and also reduces kernel launch overhead.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        # The convolution layer remains a standard, highly optimized PyTorch/cuDNN operator.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Store the compiled fused operator
        self.fused_gelu_avg_pool = fused_op.gelu_global_avg_pool_cuda

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels)
        """
        # 1. Perform convolution
        x = self.conv(x)
        
        # 2. Apply the fused GELU + global average pooling kernel
        x = self.fused_gelu_avg_pool(x)
        
        # The squeeze operations from the original model are no longer needed
        # because our custom kernel directly outputs a 2D tensor of shape (B, C).
        return x