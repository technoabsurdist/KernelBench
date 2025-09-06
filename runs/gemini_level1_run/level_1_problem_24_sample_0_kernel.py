import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused CUDA kernel for LogSoftmax
# This kernel performs the entire LogSoftmax operation in a single pass for each row.
# It uses a standard parallel reduction algorithm with shared memory.
# 1. Find the maximum value in the row.
# 2. Subtract the max from each element and compute the sum of their exponentials.
# 3. Compute the log of the sum.
# 4. Final result is: x_i - max - log(sum(exp(x_j - max))).
log_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>
#include <c10/cuda/CUDAException.h>

__global__ void log_softmax_fwd_kernel(const float* __restrict__ x, float* __restrict__ out, int batch_size, int dim) {
    // Each block processes one row of the input tensor.
    int row = blockIdx.x;
    if (row >= batch_size) {
        return;
    }

    // Shared memory for performing reductions (max and sum) within the block.
    extern __shared__ float sdata[];

    const float* row_x = x + row * dim;
    float* row_out = out + row * dim;

    // Step 1: Find the maximum value in the row.
    // Each thread finds the maximum in its own subset of the row's elements.
    float thread_max = -FLT_MAX;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        thread_max = max(thread_max, row_x[i]);
    }
    
    // Perform a block-level reduction to find the overall maximum for the row.
    sdata[threadIdx.x] = thread_max;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = max(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }
    const float row_max = sdata[0];
    __syncthreads(); // Ensure all threads have the correct row_max.

    // Step 2: Calculate the sum of exponentials.
    // Each thread calculates a partial sum using the row_max for numerical stability.
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        thread_sum += expf(row_x[i] - row_max);
    }

    // Perform a block-level reduction to get the total sum for the row.
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float row_sum_exp = sdata[0];
    __syncthreads(); // Ensure all threads have the correct row_sum_exp.

    // Step 3: Compute the final LogSoftmax values and write to the output tensor.
    const float log_row_sum_exp = logf(row_sum_exp);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        row_out[i] = row_x[i] - row_max - log_row_sum_exp;
    }
}

// C++ wrapper function that launches the CUDA kernel.
torch::Tensor log_softmax_cuda(torch::Tensor x, int64_t dim_arg) {
    // Input validation.
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D");
    TORCH_CHECK(dim_arg == 1 || dim_arg == -1, "Only dim=1 or dim=-1 is supported for this custom kernel");
    
    // Ensure the input tensor is contiguous for correct memory access.
    auto x_contig = x.contiguous();

    const auto batch_size = x_contig.size(0);
    const auto dim = x_contig.size(1);
    auto out = torch::empty_like(x_contig);

    // Kernel launch configuration.
    // A block size of 256 is a reasonable default.
    const int block_size = 256;
    dim3 grid(batch_size);
    dim3 block(block_size);
    
    // Shared memory size is determined by the number of threads in the block.
    size_t smem_size = block_size * sizeof(float);

    // Launch the kernel.
    log_softmax_fwd_kernel<<<grid, block, smem_size>>>(
        x_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        dim
    );
    
    // Check for any errors during kernel launch.
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

log_softmax_cpp_source = "torch::Tensor log_softmax_cuda(torch::Tensor x, int64_t dim_arg);"

# Compile the inline CUDA code
custom_log_softmax = load_inline(
    name="custom_log_softmax",
    cpp_sources=log_softmax_cpp_source,
    cuda_sources=log_softmax_source,
    functions=["log_softmax_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that uses a custom fused CUDA kernel for LogSoftmax.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        return custom_log_softmax.log_softmax_cuda(x, self.dim)

batch_size = 4096
dim = 393216

def get_inputs():
    # Ensure input is on CUDA
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed