import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and its C++ wrapper for max reduction
max_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

// A common block size for reduction operations. Must be a power of 2.
#define BLOCK_SIZE 256

// CUDA kernel for max reduction along a specified dimension.
// This implementation assumes the input tensor is contiguous.
// The grid is 2D, with dimensions (inner_size, outer_size).
// Each block is responsible for computing one element of the output tensor.
// Threads within a block cooperate to perform the reduction along the reduction dimension.
__global__ void max_reduction_kernel(const float* input, float* output,
                                     const int outer_size, const int reduction_size, const int inner_size) {
    // Get the 2D indices for the output element this block is responsible for
    const int inner_idx = blockIdx.x;
    const int outer_idx = blockIdx.y;

    // Calculate the starting pointer for the slice of data to be reduced
    const float* slice_start = input + outer_idx * reduction_size * inner_size + inner_idx;

    // Each thread computes a partial maximum over a subset of the reduction dimension
    float thread_max = -std::numeric_limits<float>::infinity();
    for (int i = threadIdx.x; i < reduction_size; i += blockDim.x) {
        // The stride between elements in the reduction dimension is inner_size
        thread_max = fmaxf(thread_max, slice_start[i * inner_size]);
    }

    // Use shared memory for an efficient, block-level reduction
    __shared__ float s_data[BLOCK_SIZE];
    s_data[threadIdx.x] = thread_max;
    __syncthreads();

    // Perform the reduction in shared memory.
    // This is a standard parallel reduction algorithm.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_data[threadIdx.x] = fmaxf(s_data[threadIdx.x], s_data[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Thread 0 of each block writes the final reduced value to the output tensor
    if (threadIdx.x == 0) {
        output[outer_idx * inner_size + inner_idx] = s_data[0];
    }
}

// C++ wrapper function that will be called from Python.
// It handles tensor checks, calculates dimensions, and launches the CUDA kernel.
torch::Tensor max_reduction_cuda(torch::Tensor x, int64_t dim) {
    // Input validation checks
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");

    const auto ndim = x.dim();
    // Handle negative dimension index (e.g., -1)
    const int64_t positive_dim = at::maybe_wrap_dim(dim, ndim);

    // Calculate the sizes for the kernel launch configuration.
    // outer_size: product of dimensions before the reduction dim.
    // reduction_size: size of the reduction dim.
    // inner_size: product of dimensions after the reduction dim.
    int outer_size = 1;
    for (int i = 0; i < positive_dim; ++i) {
        outer_size *= x.size(i);
    }

    const int reduction_size = x.size(positive_dim);

    int inner_size = 1;
    for (int i = positive_dim + 1; i < ndim; ++i) {
        inner_size *= x.size(i);
    }

    // Determine the shape of the output tensor
    std::vector<int64_t> output_shape;
    for (int i = 0; i < ndim; ++i) {
        if (i != positive_dim) {
            output_shape.push_back(x.size(i));
        }
    }
    // Handle the case where the output is a scalar (all dimensions reduced)
    if (output_shape.empty()) {
        output_shape.push_back(1);
    }

    auto output = torch::empty(output_shape, x.options());

    // If the dimension to be reduced has size 0, the output is filled with -inf
    if (reduction_size == 0) {
        output.fill_(-std::numeric_limits<float>::infinity());
        return output;
    }

    // Configure and launch the CUDA kernel
    const dim3 grid(inner_size, outer_size);
    const dim3 block(BLOCK_SIZE);

    max_reduction_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        outer_size,
        reduction_size,
        inner_size
    );

    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

max_reduction_cpp_source = (
    "torch::Tensor max_reduction_cuda(torch::Tensor x, int64_t dim);"
)

# JIT compile the inline CUDA code
max_reduction_op = load_inline(
    name="max_reduction_op",
    cpp_sources=max_reduction_cpp_source,
    cuda_sources=max_reduction_source,
    functions=["max_reduction_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that uses a custom CUDA kernel for Max reduction.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super().__init__()
        self.dim = dim
        # Store the compiled operator
        self.max_reduction_op = max_reduction_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a custom CUDA kernel for Max reduction to the input tensor.

        Args:
            x (torch.Tensor): Input tensor. Must be a contiguous float32 CUDA tensor.

        Returns:
            torch::Tensor: Output tensor after Max reduction.
        """
        return self.max_reduction_op.max_reduction_cuda(x, self.dim)