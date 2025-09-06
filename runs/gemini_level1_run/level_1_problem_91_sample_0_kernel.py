import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for reverse cumulative sum along dim=1
reverse_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// A block-wide inclusive scan using shared memory.
// The contents of s_data are replaced with the scan.
// This is a race-free implementation.
__device__ void block_inclusive_scan(float* s_data, int block_dim) {
    for (int offset = 1; offset < block_dim; offset *= 2) {
        __syncthreads();
        float temp = 0;
        if (threadIdx.x >= offset) {
            temp = s_data[threadIdx.x - offset];
        }
        __syncthreads();
        if (threadIdx.x >= offset) {
            s_data[threadIdx.x] += temp;
        }
    }
    __syncthreads();
}

__global__ void reverse_cumsum_dim1_kernel(const float* __restrict__ x, float* __restrict__ out, int N, int M) {
    // Each block processes one row.
    const int row = blockIdx.x;
    if (row >= N) {
        return;
    }

    // Use dynamic shared memory, size is passed from the host.
    extern __shared__ float s_data[];

    const float* x_row = x + row * M;
    float* out_row = out + row * M;

    float running_sum = 0.0f;

    const int num_chunks = (M + blockDim.x - 1) / blockDim.x;

    // Iterate through chunks from right to left.
    for (int chunk_idx = num_chunks - 1; chunk_idx >= 0; --chunk_idx) {
        const int global_idx = chunk_idx * blockDim.x + threadIdx.x;

        // Load data for the current chunk into shared memory.
        if (global_idx < M) {
            s_data[threadIdx.x] = x_row[global_idx];
        } else {
            s_data[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Perform an inclusive scan on the tile in shared memory.
        block_inclusive_scan(s_data, blockDim.x);

        // After the scan, the last element s_data[blockDim.x - 1] holds the sum of the tile.
        const float tile_sum = s_data[blockDim.x - 1];

        // Calculate the exclusive scan value for the current thread.
        const float exclusive_val = (threadIdx.x > 0) ? s_data[threadIdx.x - 1] : 0.0f;
        
        // Ensure all threads have read from s_data before the next step.
        // A sync is not strictly needed here because block_inclusive_scan ends with one,
        // and exclusive_val is read from a different thread's index, but it's safer.
        __syncthreads();

        // The reverse cumulative sum within the tile is (tile_sum - exclusive_val).
        // Add the running_sum from all tiles to the right.
        const float final_val = (tile_sum - exclusive_val) + running_sum;

        // Write the result to global memory.
        if (global_idx < M) {
            out_row[global_idx] = final_val;
        }

        // The next iteration (for the tile to the left) needs to add the sum of the current tile.
        // All threads in the block compute the same running_sum, which is fine.
        running_sum += tile_sum;
        
        // Ensure all writes are done and running_sum is updated before the next chunk starts.
        __syncthreads();
    }
}

torch::Tensor reverse_cumsum_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D for this custom kernel");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    const auto N = x.size(0);
    const auto M = x.size(1);

    auto out = torch::empty_like(x);

    const int block_size = 256;
    const dim3 threads(block_size);
    const dim3 blocks(N);
    const size_t shared_mem_size = block_size * sizeof(float);

    reverse_cumsum_dim1_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        M
    );
    
    // Check for any kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
"""

reverse_cumsum_cpp_source = (
    "torch::Tensor reverse_cumsum_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code
# This fused kernel replaces three separate PyTorch operations (flip, cumsum, flip)
# with a single, memory-efficient kernel, significantly reducing memory bandwidth
# requirements and kernel launch overhead.
rev_cumsum_op = load_inline(
    name="rev_cumsum_op",
    cpp_sources=reverse_cumsum_cpp_source,
    cuda_sources=reverse_cumsum_source,
    functions=["reverse_cumsum_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    An optimized model that performs a reverse cumulative sum operation using a
    custom fused CUDA kernel.

    This implementation is optimized specifically for a 2D tensor and dim=1.

    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
                   Must be 1 for this optimized version.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        if dim != 1:
            raise NotImplementedError(
                "Custom CUDA kernel only supports dim=1 for reverse cumsum."
            )
        self.dim = dim

    def forward(self, x):
        # The custom operator fuses flip + cumsum + flip into a single kernel call.
        return rev_cumsum_op.reverse_cumsum_cuda(x)