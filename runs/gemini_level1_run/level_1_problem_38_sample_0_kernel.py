import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused L1 normalization
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // for fabsf

// Fused kernel for L1 normalization: abs -> mean -> div
// This kernel processes one row per block.
__global__ void l1_norm_kernel(const float* __restrict__ x, float* __restrict__ out, int batch_size, int dim, float epsilon) {
    // Use dynamic shared memory for reduction, size is passed from host
    extern __shared__ double sdata[];

    // Get the row index this block is responsible for
    const int row_idx = blockIdx.x;
    if (row_idx >= batch_size) return;

    // Each thread calculates a partial sum of absolute values for the row.
    // Using double for the accumulator to maintain precision over many additions.
    double my_sum = 0.0;
    const int offset = row_idx * dim;
    for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        my_sum += (double)fabsf(x[offset + col]);
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

    // After reduction, thread 0 holds the total sum for the row.
    // It calculates the mean and writes it back to shared memory for all threads to use.
    if (threadIdx.x == 0) {
        // Add epsilon to avoid division by zero
        double mean_val = sdata[0] / dim + epsilon;
        sdata[0] = mean_val; // Reuse shared memory to broadcast the mean
    }
    __syncthreads();

    // All threads read the broadcasted mean from shared memory
    const double mean_val_div = sdata[0];

    // Perform the element-wise division
    for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        out[offset + col] = x[offset + col] / (float)mean_val_div;
    }
}

torch::Tensor l1_norm_cuda(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    const auto batch_size = x.size(0);
    const auto dim = x.size(1);

    auto out = torch::empty_like(x);

    // Kernel launch configuration
    // Using a larger block size is generally good for reduction-heavy kernels
    const int block_size = 512;
    const int num_blocks = batch_size;
    // Shared memory size: one double per thread in the block for the reduction
    const size_t shared_mem_size = block_size * sizeof(double);
    const float epsilon = 1e-8f;

    l1_norm_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        dim,
        epsilon
    );

    // Check for any CUDA errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

l1_norm_cpp_source = (
    "torch::Tensor l1_norm_cuda(torch::Tensor x);"
)

# JIT compile the inline CUDA code
l1_norm_fused = load_inline(
    name="l1_norm_fused",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_norm_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using a single fused CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the model and the custom CUDA operator.
        """
        super(ModelNew, self).__init__()
        self.l1_norm_op = l1_norm_fused.l1_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using the custom kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        return self.l1_norm_op(x)