import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for exclusive cumulative sum.
# This kernel is optimized for 2D tensors and performs the operation along dim=1.
# It fuses the "prepend zero", "cumsum", and "slice" operations into a single pass,
# avoiding intermediate memory allocations and data copies.
# The implementation uses a parallel scan algorithm within each CUDA block to efficiently
# process long rows.
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// The block dimension must be a power of 2 for the scan algorithm to work correctly.
#define BLOCK_DIM 256

__global__ void exclusive_cumsum_dim1_kernel(const float* in, float* out, int B, int N) {
    // Each block is responsible for processing one row of the input tensor.
    int row = blockIdx.x;
    if (row >= B) {
        return;
    }

    // Shared memory to store the sum of each thread's chunk.
    // This will be used to perform a block-wide parallel scan.
    __shared__ float s_sums[BLOCK_DIM];

    int tid = threadIdx.x;
    const float* row_in = in + row * N;
    float* row_out = out + row * N;

    // --- Step 1: Parallel Reduction within Chunks ---
    // Each thread computes the sum of its assigned chunk of elements from the row.
    // We use a strided loop to ensure all elements are processed.
    float my_sum = 0.0f;
    for (int i = tid; i < N; i += BLOCK_DIM) {
        my_sum += row_in[i];
    }
    s_sums[tid] = my_sum;
    __syncthreads();

    // --- Step 2: Block-wide Inclusive Scan ---
    // Perform an inclusive scan on the partial sums stored in shared memory.
    // After this, s_sums[tid] will contain the sum of chunks 0 through tid.
    for (int stride = 1; stride < BLOCK_DIM; stride *= 2) {
        float val = (tid >= stride) ? s_sums[tid - stride] : 0.0f;
        __syncthreads();
        if (tid >= stride) {
            s_sums[tid] += val;
        }
        __syncthreads();
    }

    // --- Step 3: Compute Exclusive Prefix and Final Output ---
    // The exclusive prefix for the current thread's chunk is the inclusive sum
    // of the previous thread's chunk.
    float exclusive_prefix = (tid > 0) ? s_sums[tid - 1] : 0.0f;
    __syncthreads();

    // Each thread now iterates through its chunk again, calculating the final
    // exclusive cumsum for each element and writing it to global memory.
    // The final value is the sum of all preceding chunks (exclusive_prefix) plus
    // the sum of preceding elements within the current chunk (current_chunk_scan).
    float current_chunk_scan = 0.0f;
    for (int i = tid; i < N; i += BLOCK_DIM) {
        float val = row_in[i];
        row_out[i] = exclusive_prefix + current_chunk_scan;
        current_chunk_scan += val;
    }
}

// C++ wrapper function that will be called from Python.
torch::Tensor exclusive_cumsum_cuda(torch::Tensor x, int64_t dim) {
    // Input validation.
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Custom kernel only supports 2D tensors, but got ", x.dim(), "D");
    TORCH_CHECK(dim == 1, "Custom kernel only supports dim=1, but got dim=", dim);
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");

    int B = x.size(0); // Batch size
    int N = x.size(1); // Length of the dimension to scan

    // Allocate the output tensor.
    auto out = torch::empty_like(x);

    // Configure and launch the CUDA kernel.
    const int block_size = BLOCK_DIM;
    const int num_blocks = B; // One block per row.

    exclusive_cumsum_dim1_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        N
    );
    
    // Check for any errors during kernel launch.
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# Define the C++ function signature for the JIT compiler.
cpp_source = "torch::Tensor exclusive_cumsum_cuda(torch::Tensor x, int64_t dim);"

# Compile the CUDA and C++ code inline.
custom_exclusive_cumsum = load_inline(
    name="custom_exclusive_cumsum",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["exclusive_cumsum_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    An optimized model that performs an exclusive cumulative sum using a custom CUDA kernel.
    The custom kernel is specialized for 2D tensors and dim=1.

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
                   Must be 1 for this optimized implementation.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        # The custom kernel is hardcoded for dim=1, so we enforce this constraint.
        if dim != 1:
            raise NotImplementedError("Custom CUDA kernel only supports dim=1")
        self.dim = dim

    def forward(self, x):
        # Call the custom CUDA function.
        return custom_exclusive_cumsum.exclusive_cumsum_cuda(x, self.dim)