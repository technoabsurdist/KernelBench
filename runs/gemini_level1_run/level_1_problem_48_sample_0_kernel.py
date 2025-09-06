import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and its C++ wrapper for mean reduction
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for mean reduction along dimension 1 of a 3D tensor
// This kernel is optimized for the specific case of reducing a (B, N, M) tensor to (B, M)
__global__ void mean_dim1_kernel(const float* __restrict__ x, float* __restrict__ out, int B, int N, int M) {
    // Each block computes one element of the output tensor, corresponding to out[b][m]
    const int b = blockIdx.y;
    const int m = blockIdx.x;

    // Check bounds for the block to avoid writing out of bounds
    if (b >= B || m >= M) {
        return;
    }

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Use shared memory for efficient reduction within the block
    extern __shared__ float sdata[];

    // Each thread computes a partial sum over the N dimension
    float partial_sum = 0.0f;
    for (int i = tid; i < N; i += block_size) {
        // Calculate the linear index for x[b][i][m]
        int x_idx = b * N * M + i * M + m;
        partial_sum += x[x_idx];
    }
    sdata[tid] = partial_sum;
    __syncthreads();

    // Perform reduction in shared memory. This is a standard parallel reduction algorithm.
    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 of each block writes the final result (sum / N) to the output tensor
    if (tid == 0) {
        // Calculate the linear index for out[b][m]
        int out_idx = b * M + m;
        out[out_idx] = sdata[0] / static_cast<float>(N);
    }
}

// C++ wrapper function that launches the CUDA kernel
torch::Tensor mean_cuda(torch::Tensor x) {
    // Input validation checks
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.dim() == 3, "Input tensor must be 3-dimensional, but got ", x.dim(), " dimensions");

    const int B = x.size(0);
    const int N = x.size(1);
    const int M = x.size(2);

    // Create the output tensor with the reduced shape
    auto out = torch::empty({B, M}, x.options());

    // Kernel launch configuration
    const int block_size = 256; // A common choice, can be tuned for specific hardware
    const dim3 grid(M, B); // A 2D grid where each block corresponds to an output element
    const dim3 block(block_size);
    const size_t shared_mem_size = block_size * sizeof(float);

    // Launch the kernel
    mean_dim1_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        B, N, M
    );
    
    // Check for any CUDA errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

cpp_source = "torch::Tensor mean_cuda(torch::Tensor x);"

# Compile the inline CUDA code
custom_mean = load_inline(
    name="custom_mean",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["mean_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that uses a custom CUDA kernel for mean reduction over dimension 1.
    """
    def __init__(self, dim: int):
        """
        Initializes the model. The custom kernel is specialized for dim=1.

        Args:
            dim (int): The dimension to reduce over. Must be 1 for the custom kernel.
        """
        super(ModelNew, self).__init__()
        if dim != 1:
            # This implementation is specialized for the given inputs where dim=1.
            # A more general implementation would require a more complex kernel or multiple kernels.
            raise NotImplementedError("Custom CUDA kernel is only implemented for dim=1")
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduces the input tensor along dimension 1 by taking the mean using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, M).

        Returns:
            torch.Tensor: Output tensor of shape (B, M).
        """
        # The custom_mean.mean_cuda function expects a 3D contiguous CUDA tensor.
        # The C++ wrapper includes checks for this.
        return custom_mean.mean_cuda(x)

batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [1]