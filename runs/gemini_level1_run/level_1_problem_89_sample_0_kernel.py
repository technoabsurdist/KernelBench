import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel and C++ wrapper for a parallel cumsum operation along dim=1
cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// A work-efficient parallel scan (prefix sum) implemented in a CUDA kernel.
// This kernel is designed for 2D tensors, performing the scan along dim=1.
// Each CUDA block is responsible for scanning one row of the input tensor.
__global__ void cumsum_dim1_kernel(const float* in, float* out, int rows, int cols) {
    // Shared memory for the block-wide scan on partial sums from each thread.
    extern __shared__ float s_data[];

    // Identify the row this block is processing.
    int row_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int block_dim = blockDim.x;

    // Pointers to the start of the current row for input and output.
    const float* row_in_ptr = in + row_idx * cols;
    float* row_out_ptr = out + row_idx * cols;

    // 1. Each thread computes a sequential prefix sum on its contiguous chunk of the row.
    // This approach ensures coalesced memory access.
    int items_per_thread = (cols + block_dim - 1) / block_dim;
    int start_idx = thread_idx * items_per_thread;
    int end_idx = min(start_idx + items_per_thread, cols);

    float thread_sum = 0.0f;
    if (start_idx < cols) {
        for (int i = start_idx; i < end_idx; ++i) {
            thread_sum += row_in_ptr[i];
            row_out_ptr[i] = thread_sum;
        }
    }

    // Store the total sum for this thread's chunk in shared memory.
    s_data[thread_idx] = thread_sum;
    __syncthreads();

    // 2. Perform an exclusive scan on the partial sums in shared memory using the Blelloch algorithm.
    // This is a two-phase (up-sweep and down-sweep) algorithm.

    // Up-sweep (reduction) phase:
    for (unsigned int d = 1; d < block_dim; d *= 2) {
        __syncthreads();
        if (thread_idx >= d) {
            s_data[thread_idx] += s_data[thread_idx - d];
        }
    }

    // Down-sweep phase:
    if (thread_idx == block_dim - 1) {
        s_data[thread_idx] = 0; // The last element gets 0 for an exclusive scan.
    }
    __syncthreads();

    for (unsigned int d = block_dim / 2; d > 0; d /= 2) {
        __syncthreads();
        if (thread_idx < block_dim - d) {
            float temp = s_data[thread_idx + d];
            s_data[thread_idx + d] += s_data[thread_idx];
            s_data[thread_idx] = temp;
        }
    }
    __syncthreads();

    // s_data[thread_idx] now contains the exclusive scan result, which is the offset
    // to be added to all elements in this thread's chunk.
    float offset = s_data[thread_idx];

    // 3. Add the calculated offset to all elements in this thread's chunk.
    if (offset != 0.0f) {
        for (int i = start_idx; i < end_idx; ++i) {
            row_out_ptr[i] += offset;
        }
    }
}

// C++ function to be bound with PyTorch, acting as the interface to the CUDA kernel.
torch::Tensor cumsum_cuda(torch::Tensor x, int64_t dim) {
    // Input validation.
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Custom cumsum kernel only supports 2D tensors, but got ", x.dim(), " dimensions.");
    TORCH_CHECK(dim == 1, "Custom cumsum kernel only supports dim=1, but got dim=", dim);
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be of type float32");

    // Ensure the input tensor is contiguous for optimal memory access patterns.
    auto x_contiguous = x.contiguous();

    int rows = x_contiguous.size(0);
    int cols = x_contiguous.size(1);

    auto out = torch::empty_like(x_contiguous);

    // Configure and launch the CUDA kernel.
    // A block size of 512 is a good balance for many GPUs.
    const int block_size = 512;
    const int num_blocks = rows;
    
    // Shared memory size is determined by the number of threads in a block.
    size_t smem_size = block_size * sizeof(float);

    cumsum_dim1_kernel<<<num_blocks, block_size, smem_size>>>(
        x_contiguous.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols
    );
    
    // Check for any errors during kernel launch.
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

cumsum_cpp_source = "torch::Tensor cumsum_cuda(torch::Tensor x, int64_t dim);"

# JIT compile the CUDA and C++ code using PyTorch's C++ extension utilities.
custom_cumsum = load_inline(
    name="custom_cumsum",
    cpp_sources=cumsum_cpp_source,
    cuda_sources=cumsum_source,
    functions=["cumsum_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    An optimized model that uses a custom CUDA kernel for the cumulative sum (prefix sum) operation.
    The custom kernel is specifically optimized for a 2D tensor and a scan along dimension 1.

    Parameters:
        dim (int): The dimension along which to perform the scan operation.
    """

    def __init__(self, dim):
        """
        Initialize the optimized Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.custom_cumsum_op = custom_cumsum

    def forward(self, x):
        """
        Forward pass for the optimized Scan model.

        If the specified dimension is 1 (the optimized case), it calls the custom CUDA kernel.
        Otherwise, it falls back to the default PyTorch `torch.cumsum` implementation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative sum along `dim`.
        """
        # Our custom kernel is highly specialized for dim=1 on a 2D float tensor.
        if self.dim == 1 and x.dim() == 2 and x.is_cuda and x.dtype == torch.float32:
            return self.custom_cumsum_op.cumsum_cuda(x, self.dim)
        else:
            # Fallback to the standard PyTorch implementation for non-supported cases.
            return torch.cumsum(x, dim=self.dim)