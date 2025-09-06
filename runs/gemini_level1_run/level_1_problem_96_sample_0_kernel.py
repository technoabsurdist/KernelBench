import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Smooth L1 Loss
# This kernel fuses the element-wise loss calculation and the first stage of a parallel reduction.
smooth_l1_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel to compute Smooth L1 loss and perform a partial reduction per block.
// This kernel uses a grid-stride loop to handle any number of elements, making it robust.
__global__ void smooth_l1_loss_kernel(const float* predictions, const float* targets, float* block_sums, long long size) {
    // Shared memory for per-block reduction
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes its local sum using a grid-stride loop.
    // This ensures all elements are processed regardless of the grid size.
    float my_sum = 0.0f;
    for (long long idx = i; idx < size; idx += gridDim.x * blockDim.x) {
        float diff = predictions[idx] - targets[idx];
        float abs_diff = fabsf(diff);
        if (abs_diff < 1.0f) {
            my_sum += 0.5f * diff * diff;
        } else {
            my_sum += abs_diff - 0.5f;
        }
    }

    sdata[tid] = my_sum;
    __syncthreads();

    // In-block reduction using shared memory.
    // This is a standard parallel reduction algorithm.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // The first thread in the block writes the block's partial sum to global memory.
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

// C++ wrapper function that launches the CUDA kernel and handles the final reduction.
torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    // Input validation
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "predictions and targets must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "Input tensors must be of type float32");

    long long size = predictions.numel();
    if (size == 0) {
        return torch::tensor(0.0, predictions.options());
    }

    // Kernel launch configuration
    // Use a fixed block size, a power of 2 is good for reductions.
    const int block_size = 256;
    // Heuristic for the number of blocks. We want enough to saturate the GPU.
    // A fixed large number of blocks combined with a grid-stride loop is a robust approach.
    const int num_blocks = 2048;

    // Allocate a tensor on the GPU to store the partial sum from each block
    auto block_sums = torch::zeros({num_blocks}, predictions.options());

    // Launch the kernel for the first stage of reduction
    // The third argument specifies the amount of dynamic shared memory per block.
    smooth_l1_loss_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_sums.data_ptr<float>(),
        size
    );

    // Sum the partial results from each block on the GPU using PyTorch's optimized sum.
    // This is simpler and often as fast as writing a second reduction kernel.
    auto total_sum = block_sums.sum();

    // Return the mean loss
    return total_sum / size;
}
"""

# C++ source for the function signature, required by load_inline
smooth_l1_loss_cpp_source = (
    "torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

# JIT compile the inline CUDA code.
# This will be compiled the first time the script is run.
# Subsequent runs will use the cached compiled library.
custom_smooth_l1_loss = load_inline(
    name="custom_smooth_l1_loss",
    cpp_sources=smooth_l1_loss_cpp_source,
    cuda_sources=smooth_l1_loss_source,
    functions=["smooth_l1_loss_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks
    using a custom CUDA kernel for improved performance.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Call the custom CUDA function instead of torch.nn.functional.smooth_l1_loss
        return custom_smooth_l1_loss.smooth_l1_loss_cuda(predictions, targets)