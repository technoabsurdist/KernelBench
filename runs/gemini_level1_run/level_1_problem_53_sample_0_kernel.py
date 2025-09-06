import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for min reduction along dimension 1
min_reduce_dim1_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat> // For FLT_MAX

// CUDA kernel for min reduction along dimension 1 of a 3D tensor.
// This kernel is optimized for the case where the reduction happens along the second dimension (dim=1).
// It assumes the input tensor is contiguous in memory (row-major).
//
// Grid dimensions are matched to the output tensor's dimensions: (H, N).
// Each block is responsible for computing a single element of the output tensor.
// Threads within a block cooperate to reduce one slice of the input tensor.
__global__ void min_reduce_dim1_kernel(const float* in_data, float* out_data, int N, int C, int H) {
    // Use shared memory for efficient reduction within a thread block.
    extern __shared__ float sdata[];

    // Get thread and block indices.
    // The grid is configured as (H, N), so blockIdx.x corresponds to the H dimension
    // and blockIdx.y corresponds to the N dimension.
    int h = blockIdx.x;
    int n = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Calculate the base pointer for the slice we are reducing.
    // This points to the first element of x[n, :, h].
    const float* slice_ptr = in_data + n * C * H + h;

    // Each thread computes a partial minimum from the input slice.
    // Since we are reducing along dim=1, the elements are not contiguous in memory.
    // The stride between elements is H.
    float thread_min = FLT_MAX;
    for (int i = tid; i < C; i += block_size) {
        thread_min = min(thread_min, slice_ptr[i * H]);
    }

    // Store the partial minimum in shared memory.
    sdata[tid] = thread_min;
    __syncthreads(); // Synchronize to ensure all partial minimums are in shared memory.

    // Perform reduction in shared memory.
    // This is a standard parallel reduction algorithm using a tree-based approach.
    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads(); // Synchronize at each step of the reduction.
    }

    // Thread 0 of each block writes the final reduced value to the output tensor.
    if (tid == 0) {
        // The output tensor has shape (N, H).
        out_data[n * H + h] = sdata[0];
    }
}

// C++ wrapper function that will be called from Python.
torch::Tensor min_reduce_dim1_cuda(torch::Tensor x) {
    // Input validation.
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be a float32 tensor");
    TORCH_CHECK(x.dim() == 3, "Input tensor must be 3-dimensional");
    
    // Ensure tensor is contiguous for simplified indexing in the kernel.
    x = x.contiguous();

    // Get input dimensions.
    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);

    // Create the output tensor with the reduced shape (N, H).
    auto out = torch::empty({N, H}, x.options());

    // Configure kernel launch parameters.
    // A block size of 1024 is chosen as it's the maximum on many modern GPUs and
    // provides a good balance of parallelism.
    const int block_size = 1024;
    const dim3 grid_size(H, N);
    const dim3 block_dim(block_size);
    const size_t shared_mem_size = block_size * sizeof(float);

    // Launch the CUDA kernel.
    min_reduce_dim1_kernel<<<grid_size, block_dim, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H
    );

    // Check for any CUDA errors after kernel launch to ensure correctness.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return out;
}
"""

min_reduce_dim1_cpp_source = (
    "torch::Tensor min_reduce_dim1_cuda(torch::Tensor x);"
)

# JIT compile the CUDA and C++ code. This is done once when the module is imported.
min_reduce_dim1_op = load_inline(
    name="min_reduce_dim1_op",
    cpp_sources=min_reduce_dim1_cpp_source,
    cuda_sources=min_reduce_dim1_source,
    functions=["min_reduce_dim1_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that performs min reduction over dimension 1 using a custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over. Must be 1 for the custom kernel.
        """
        super(ModelNew, self).__init__()
        if dim != 1:
            raise ValueError("Custom CUDA kernel is specialized for dim=1 only.")
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies min reduction over dimension 1 to the input tensor using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction.
        """
        return min_reduce_dim1_op.min_reduce_dim1_cuda(x)