import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Mean Squared Error
mse_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused kernel to compute sum of squared differences with a two-stage reduction.
// Stage 1: This kernel computes partial sums for subsets of the data.
// Each block computes one partial sum using a grid-stride loop and intra-block reduction.
// Using double for accumulators to maintain precision with large inputs.
__global__ void mse_map_reduce_kernel(const float* pred, const float* target, double* block_sums, long long N) {
    // Shared memory for intra-block reduction.
    extern __shared__ double sdata[];

    // Each thread computes a local partial sum over a strided subset of the input.
    double my_sum = 0.0;
    int tid = threadIdx.x;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (long long i = idx; i < N; i += stride) {
        float diff = pred[i] - target[i];
        my_sum += (double)diff * diff;
    }

    sdata[tid] = my_sum;
    __syncthreads();

    // Perform intra-block reduction in shared memory.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // The first thread of the block writes the block's partial sum to global memory.
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

// C++ wrapper function callable from Python.
torch::Tensor mse_cuda(torch::Tensor predictions, torch::Tensor targets) {
    // Input validation.
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(), "Input tensors must have the same number of elements");
    TORCH_CHECK(predictions.is_contiguous(), "predictions must be contiguous");
    TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");

    const long long N = predictions.numel();
    if (N == 0) {
        return torch::tensor(0.0, predictions.options());
    }

    // Kernel launch configuration.
    const int block_size = 1024;
    int num_blocks;
    
    // Heuristic to determine the number of blocks based on GPU properties.
    // This aims to saturate the GPU with work.
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, predictions.device().index());
    num_blocks = prop.multiProcessorCount * 8;

    // Allocate intermediate tensor for partial sums from each block.
    // Use double precision for sums to avoid overflow/precision issues.
    auto block_sums = torch::zeros({num_blocks}, predictions.options().dtype(torch::kFloat64));

    // Shared memory size for the kernel.
    const int shared_mem_size = block_size * sizeof(double);

    // Launch the reduction kernel.
    mse_map_reduce_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_sums.data_ptr<double>(),
        N
    );
    
    // Check for kernel launch errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    // Stage 2: Sum the partial sums from each block and compute the mean.
    // This is done on the GPU using PyTorch's highly optimized sum.
    auto total_sum = block_sums.sum();
    auto mean = total_sum / N;

    // Cast the final result back to the original float type.
    return mean.to(predictions.dtype());
}
"""

mse_cpp_source = """
torch::Tensor mse_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

# JIT compile the inline CUDA/C++ code.
mse_op = load_inline(
    name="mse_op",
    cpp_sources=mse_cpp_source,
    cuda_sources=mse_source,
    functions=["mse_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    An optimized model that computes the Mean Squared Error loss using a custom
    fused CUDA kernel.

    The kernel fuses the element-wise subtraction, squaring, and the sum-reduction
    part of the mean operation into a single pass over the data. This reduces
    memory bandwidth usage and kernel launch overhead, leading to significant
    speedups for large tensors. A two-stage parallel reduction algorithm with
    double-precision accumulators is used for performance and numerical stability.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.mse_cuda = mse_op.mse_cuda

    def forward(self, predictions, targets):
        # The custom kernel requires contiguous tensors.
        return self.mse_cuda(predictions.contiguous(), targets.contiguous())