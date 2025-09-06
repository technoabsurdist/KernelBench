import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for matrix-vector multiplication
matvec_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matvec_kernel(const float* A, const float* B, float* C, int M, int K) {
    // Each block computes one element of the output vector C.
    int row = blockIdx.x;

    // Boundary check for rows, in case grid_size > M
    if (row >= M) {
        return;
    }

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Each thread computes a partial sum. The block iterates over the row in chunks.
    // This ensures coalesced memory access to A and B.
    float thread_sum = 0.0f;
    for (int j_base = 0; j_base < K; j_base += block_size) {
        int col = j_base + tid;
        if (col < K) {
            thread_sum += A[row * K + col] * B[col];
        }
    }

    // Reduction within the block using shared memory.
    extern __shared__ float s_data[];
    s_data[tid] = thread_sum;
    __syncthreads();

    // Perform reduction in shared memory.
    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 of each block writes the final result to global memory.
    if (tid == 0) {
        C[row] = s_data[0];
    }
}

torch::Tensor matvec_cuda(torch::Tensor A, torch::Tensor B) {
    // Input validation
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions of A and B must match for matmul");
    TORCH_CHECK(B.size(1) == 1, "B must be a column vector (K, 1)");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be on a CUDA device");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be a float32 tensor");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be a float32 tensor");

    const int M = A.size(0);
    const int K = A.size(1);

    // Create the output tensor C of shape (M, 1)
    auto C = torch::zeros({M, 1}, A.options());

    // Kernel launch configuration
    const int block_size = 256;
    // Launch one block for each row of the output matrix C.
    const int grid_size = M;

    // Allocate shared memory for the reduction within each block.
    const size_t shared_mem_size = block_size * sizeof(float);

    // Launch the kernel
    matvec_kernel<<<grid_size, block_size, shared_mem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        K
    );
    
    // Check for errors after kernel launch for debugging
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}
"""

matvec_cpp_source = "torch::Tensor matvec_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matvec_mult = load_inline(
    name="matvec_mult",
    cpp_sources=matvec_cpp_source,
    cuda_sources=matvec_source,
    functions=["matvec_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication (C = A * B)
    using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matvec_mult = matvec_mult

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication using the custom CUDA kernel.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return self.matvec_mult.matvec_cuda(A, B)