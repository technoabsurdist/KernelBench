import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for sum reduction along dimension 1
sum_reduce_dim1_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for sum reduction along dimension 1 of a 3D tensor (N, C, H)
// The grid is launched with dimensions (N, H)
// Each block reduces along the C dimension
__global__ void sum_reduce_dim1_kernel(const float* x, float* out, int N, int C, int H) {
    // Shared memory for intra-block reduction
    extern __shared__ float sdata[];

    // Thread index within the block
    int tid = threadIdx.x;
    // Block index identifies the (n, h) coordinate of the output
    int n = blockIdx.x;
    int h = blockIdx.y;

    // Stride for accessing elements along the C dimension
    long long c_stride = H;
    // Stride for accessing elements along the N dimension
    long long n_stride = (long long)C * H;

    // Pointer to the start of the slice to be reduced: x[n, 0, h]
    const float* x_slice_ptr = x + n * n_stride + h;

    // Each thread computes a partial sum by looping over the C dimension
    float thread_sum = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        thread_sum += x_slice_ptr[c * c_stride];
    }

    // Store the partial sum in shared memory
    sdata[tid] = thread_sum;
    __syncthreads();

    // Perform reduction in shared memory. This is a standard parallel reduction.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // The first thread in the block writes the final sum to global memory
    if (tid == 0) {
        // Output index for out[n, 0, h]
        out[n * H + h] = sdata[0];
    }
}

// C++ wrapper function that launches the CUDA kernel
torch::Tensor sum_reduce_dim1_cuda(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.dim() == 3, "Input tensor must be 3-dimensional");

    // Get tensor dimensions
    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);

    // Create the output tensor with shape (N, 1, H)
    auto out = torch::empty({N, 1, H}, x.options());

    // Configure kernel launch parameters
    // Use a power-of-two block size for efficient reduction
    const int block_size = 256;
    dim3 block_dim(block_size);
    dim3 grid_dim(N, H);

    // Shared memory size per block
    const size_t shared_mem_size = block_size * sizeof(float);

    // Launch the kernel
    sum_reduce_dim1_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H
    );

    return out;
}
"""

# C++ source for the function signature, required by load_inline
sum_reduce_dim1_cpp_source = """
torch::Tensor sum_reduce_dim1_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code using torch.utils.cpp_extension
# This creates a Python module that can be called directly.
sum_reduce_dim1 = load_inline(
    name="sum_reduce_dim1",
    cpp_sources=sum_reduce_dim1_cpp_source,
    cuda_sources=sum_reduce_dim1_source,
    functions=["sum_reduce_dim1_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that uses a custom CUDA kernel for sum reduction.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension. If the input is a 3D tensor
        and the reduction dimension is 1, it uses a highly optimized custom CUDA kernel.
        Otherwise, it falls back to the default PyTorch implementation.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch::Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        # Check if the conditions are met to use the custom kernel
        if self.dim == 1 and x.dim() == 3 and x.is_cuda:
            # Ensure tensor is contiguous for correct memory access in CUDA
            x_contiguous = x.contiguous()
            return sum_reduce_dim1.sum_reduce_dim1_cuda(x_contiguous)
        else:
            # Fallback to the original PyTorch implementation for other cases
            return torch.sum(x, dim=self.dim, keepdim=True)