import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused KL-Divergence
# This kernel fuses the log, subtraction, multiplication, and sum reduction operations
# from the KL-Divergence formula: sum(targets * (log(targets) - log(predictions)))
kl_div_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused kernel for KL-Divergence calculation with reduction.
// It computes the sum of element-wise KL divergence terms and uses atomicAdd
// for the final reduction across blocks.
__global__ void kl_div_fused_kernel(
    const float* predictions,
    const float* targets,
    float* out_sum, // A single-element tensor, initialized to 0
    const long N)
{
    // Use shared memory for per-block reduction
    extern __shared__ float sdata[];

    // Each thread computes a partial sum over a strided region of the input
    // using a grid-stride loop. This ensures all elements are processed regardless
    // of the grid size.
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    float thread_sum = 0.0f;
    for (long i = idx; i < N; i += stride) {
        float pred = predictions[i];
        float targ = targets[i];
        // The KL-divergence term is y * (log(y) - log(x)).
        // If target (y) is 0, the term is 0. This also avoids log(0).
        // We also check pred > 0 to avoid log(0) from floating point inaccuracies.
        if (targ > 0.0f && pred > 0.0f) {
            thread_sum += targ * (logf(targ) - logf(pred));
        }
    }

    sdata[tid] = thread_sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // The first thread of the block adds the block's partial sum to the global output
    // using an atomic operation to prevent race conditions.
    if (tid == 0) {
        atomicAdd(out_sum, sdata[0]);
    }
}

// C++ wrapper function that launches the CUDA kernel
torch::Tensor kl_div_fused_cuda(torch::Tensor predictions, torch::Tensor targets) {
    // Input validation
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.is_contiguous(), "predictions must be contiguous");
    TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
    TORCH_CHECK(predictions.numel() == targets.numel(), "predictions and targets must have the same number of elements");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be a float32 tensor");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be a float32 tensor");

    const auto N = predictions.numel();
    if (N == 0) {
        return torch::zeros({}, predictions.options());
    }
    // The original reduction is 'batchmean', so we need the batch size.
    const auto batch_size = predictions.size(0);
    TORCH_CHECK(batch_size > 0, "batch size must be greater than 0 for batchmean reduction");

    // Allocate output tensor for the sum, initialized to zero
    auto out_sum = torch::zeros({1}, predictions.options());

    // Kernel launch configuration
    const int block_size = 256;
    // Heuristic for grid size to balance parallelism and atomic contention
    const int num_blocks_for_full_occupancy = (N + block_size - 1) / block_size;
    // Cap the number of blocks to prevent launching too many, which can be inefficient.
    const int grid_size = std::min(num_blocks_for_full_occupancy, 4096);
    
    const size_t shared_mem_size = block_size * sizeof(float);

    // Launch the kernel
    kl_div_fused_kernel<<<grid_size, block_size, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        out_sum.data_ptr<float>(),
        N
    );
    
    // Check for any CUDA errors that might have occurred during the kernel launch.
    C10_CUDA_CHECK(cudaGetLastError());

    // Final division for 'batchmean' reduction.
    // The result of the kernel is a 1-element tensor containing the sum.
    // We divide by batch_size and return a scalar tensor to match the original op.
    return (out_sum / batch_size).squeeze();
}
"""

kl_div_fused_cpp_source = (
    "torch::Tensor kl_div_fused_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

# JIT compile the inline CUDA code.
# This might take a moment the first time it's run.
kl_div_fused = load_inline(
    name="kl_div_fused",
    cpp_sources=kl_div_fused_cpp_source,
    cuda_sources=kl_div_fused_source,
    functions=["kl_div_fused_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    An optimized model that computes Kullback-Leibler Divergence using a custom fused CUDA kernel.
    The custom kernel replaces the sequence of torch.log, subtraction, multiplication, and sum reduction
    with a single, more efficient operation, reducing memory bandwidth and kernel launch overhead.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # The original operation is kl_div(log(predictions), targets, reduction='batchmean').
        # Our fused kernel implements this entire operation, including the initial log on both inputs.
        return kl_div_fused.kl_div_fused_cuda(predictions, targets)