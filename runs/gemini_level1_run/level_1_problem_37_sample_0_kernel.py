import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA source code
frobenius_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// A two-pass reduction is used to calculate the sum of squares of all elements in the tensor.
// This is necessary because the input tensor is too large to be reduced by a single kernel launch.

// Kernel for the first pass of sum of squares reduction.
// Each block computes a partial sum of squares for a chunk of the input data using a grid-stride loop.
// The partial sums are stored in a temporary global memory buffer (block_sums).
__global__ void reduce_sum_squares_kernel_pass1(const float* __restrict__ x, float* __restrict__ block_sums, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;

    float my_sum = 0.0f;
    while (i < N) {
        float val = x[i];
        my_sum += val * val;
        i += gridSize;
    }
    sdata[tid] = my_sum;
    __syncthreads();

    // Perform in-block reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 of each block writes the block's partial sum to global memory
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

// Kernel for the second and final pass of sum of squares reduction.
// A single block is launched to reduce the partial sums calculated in the first pass.
__global__ void reduce_sum_squares_kernel_pass2(const float* __restrict__ block_sums, float* __restrict__ final_sum, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;

    float my_sum = 0.0f;
    // Each thread sums up elements from the block_sums array.
    while (i < N) {
        my_sum += block_sums[i];
        i += blockDim.x;
    }
    sdata[tid] = my_sum;
    __syncthreads();

    // Perform final in-block reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final total sum
    if (tid == 0) {
        *final_sum = sdata[0];
    }
}


// Kernel to normalize the input tensor by its Frobenius norm.
// The norm is calculated from the final sum of squares.
// This kernel fuses the sqrt and division operations into a single multiplication with rsqrtf.
__global__ void normalize_kernel(const float* __restrict__ x, const float* __restrict__ sum_sq, float* __restrict__ out, int N) {
    // Use rsqrtf for performance: computes 1.0 / sqrt(f).
    // Add a small epsilon for numerical stability to avoid division by zero.
    float norm_inv = rsqrtf(*sum_sq + 1e-12f);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;

    // Use a grid-stride loop to ensure all elements are processed.
    while (i < N) {
        out[i] = x[i] * norm_inv;
        i += gridSize;
    }
}

// C++ wrapper function that orchestrates the CUDA kernels.
// This function is exposed to PyTorch.
torch::Tensor frobenius_normalize_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");

    const int64_t N = x.numel();
    if (N == 0) {
        return x;
    }

    // --- Reduction Step ---
    // Tuning parameters for the reduction kernels
    const int PASS1_BLOCK_SIZE = 256;
    const int GRID_SIZE = 1024; // Number of blocks for the first pass, resulting in this many partial sums

    // Allocate intermediate and final buffers on the GPU
    auto block_sums = torch::empty({GRID_SIZE}, x.options());
    auto final_sum_sq = torch::empty({1}, x.options());

    // Launch first reduction kernel
    // Shared memory size: PASS1_BLOCK_SIZE * sizeof(float)
    reduce_sum_squares_kernel_pass1<<<GRID_SIZE, PASS1_BLOCK_SIZE, PASS1_BLOCK_SIZE * sizeof(float)>>>(
        x.data_ptr<float>(), block_sums.data_ptr<float>(), N);

    // Launch second reduction kernel to sum the partial sums
    const int PASS2_BLOCK_SIZE = 1024; // Must be >= GRID_SIZE for a single pass
    reduce_sum_squares_kernel_pass2<<<1, PASS2_BLOCK_SIZE, PASS2_BLOCK_SIZE * sizeof(float)>>>(
        block_sums.data_ptr<float>(), final_sum_sq.data_ptr<float>(), GRID_SIZE);

    // --- Normalization Step ---
    auto out = torch::empty_like(x);

    // Launch normalization kernel
    const int NORM_BLOCK_SIZE = 256;
    // Use a fixed, reasonably large grid size and a grid-stride loop inside the kernel
    const int NORM_GRID_SIZE = 2048;

    normalize_kernel<<<NORM_GRID_SIZE, NORM_BLOCK_SIZE>>>(
        x.data_ptr<float>(), final_sum_sq.data_ptr<float>(), out.data_ptr<float>(), N);

    // Check for any CUDA errors during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA error in frobenius_normalize_cuda: ", cudaGetErrorString(err));
    }

    return out;
}
"""

# C++ source for function signature
frobenius_norm_cpp_source = "torch::Tensor frobenius_normalize_cuda(torch::Tensor x);"

# Compile the inline CUDA code
# This fuses the norm calculation and element-wise division into a single C++ function call
fused_frobenius_norm = load_inline(
    name="fused_frobenius_norm",
    cpp_sources=frobenius_norm_cpp_source,
    cuda_sources=frobenius_norm_source,
    functions=["frobenius_normalize_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using a custom CUDA kernel.
    The kernel fuses the reduction (sum of squares), square root, and division operations
    to minimize kernel launch overhead and memory access.
    """
    def __init__(self):
        """
        Initializes the layer with the custom fused operator.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Frobenius norm normalization using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        return fused_frobenius_norm.frobenius_normalize_cuda(x)