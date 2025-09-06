import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for fused L2 normalization
l2_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for fused L2 normalization
__global__ void l2_norm_kernel(const float* x, float* out, int batch_size, int dim) {
    // Each block processes one row (one sample in the batch)
    int row_idx = blockIdx.x;

    // Use shared memory for reduction within a block
    extern __shared__ float s_data[];

    // Each thread calculates the sum of squares for a portion of the row
    float thread_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = x[row_idx * dim + i];
        thread_sum_sq += val * val;
    }
    s_data[threadIdx.x] = thread_sum_sq;
    __syncthreads();

    // Perform reduction in shared memory. This is a standard parallel reduction algorithm.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_data[threadIdx.x] += s_data[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 now has the sum of squares for the entire row.
    // It calculates the inverse norm and stores it back to shared memory for other threads.
    if (threadIdx.x == 0) {
        // Use rsqrtf for performance, add a small epsilon for numerical stability
        float row_sum_sq = s_data[0];
        s_data[0] = rsqrtf(row_sum_sq + 1e-8f);
    }
    __syncthreads();

    // All threads read the inverse norm calculated by thread 0
    float inv_norm = s_data[0];

    // Each thread normalizes its portion of the row by multiplying with the inverse norm
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        int index = row_idx * dim + i;
        out[index] = x[index] * inv_norm;
    }
}

// C++ wrapper function that will be called from Python
torch::Tensor l2_norm_cuda(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D");

    const auto batch_size = x.size(0);
    const auto dim = x.size(1);

    // Create an output tensor of the same shape and type as the input
    auto out = torch::empty_like(x);

    // Kernel launch configuration
    const int block_size = 256; // A common choice for block size
    const int grid_size = batch_size; // Launch one block per row
    
    // Shared memory size: one float per thread in the block
    const int shared_mem_size = block_size * sizeof(float);

    // Launch the kernel
    l2_norm_kernel<<<grid_size, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        dim
    );
    
    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# C++ source for the function signature
l2_norm_cpp_source = "torch::Tensor l2_norm_cuda(torch::Tensor x);"

# Use load_inline to JIT compile the CUDA/C++ code
l2_norm_op = load_inline(
    name="l2_norm_op",
    cpp_sources=l2_norm_cpp_source,
    cuda_sources=l2_norm_source,
    functions=["l2_norm_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs L2 normalization using a custom fused CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the model and the custom L2Norm operator.
        """
        super(ModelNew, self).__init__()
        # The compiled operator is stored for use in the forward pass
        self.l2_norm_op = l2_norm_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied, same shape as input.
        """
        return self.l2_norm_op.l2_norm_cuda(x)