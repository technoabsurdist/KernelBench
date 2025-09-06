import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel using cuBLAS for symmetric matrix multiplication
symm_matmul_source = """
#include <torch/extension.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Helper for error checking cuBLAS calls
#define CUBLAS_CHECK(err) do { \
    cublasStatus_t err_ = (err); \
    if (err_ != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error("cuBLAS error " + std::to_string(err_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)

torch::Tensor symm_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Input validation
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same dimensions");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be a float32 tensor");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be a float32 tensor");

    // Ensure tensors are contiguous in memory for cuBLAS
    A = A.contiguous();
    B = B.contiguous();

    const int N = A.size(0);
    
    // Create the output tensor
    auto C = torch::empty({N, N}, A.options());

    // Create a cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Call cublasSsymm: C = alpha * A * B + beta * C
    // A is the symmetric matrix.
    // We specify that A is on the left and we will use its upper triangle.
    // cuBLAS will only read the upper triangle of A, reducing memory bandwidth.
    CUBLAS_CHECK(cublasSsymm(handle,
                             CUBLAS_SIDE_LEFT,      // A is on the left: A * B
                             CUBLAS_FILL_UPPER,     // Use the upper triangle of A
                             N,                     // m: rows of B and C
                             N,                     // n: cols of B and C
                             &alpha,                // alpha = 1.0
                             A.data_ptr<float>(),   // Pointer to A
                             N,                     // lda: leading dimension of A
                             B.data_ptr<float>(),   // Pointer to B
                             N,                     // ldb: leading dimension of B
                             &beta,                 // beta = 0.0
                             C.data_ptr<float>(),   // Pointer to C
                             N                      // ldc: leading dimension of C
                             ));

    // Destroy the cuBLAS handle
    CUBLAS_CHECK(cublasDestroy(handle));

    return C;
}
"""

symm_matmul_cpp_source = (
    "torch::Tensor symm_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for symmetric matrix multiplication
# We need to link against the cuBLAS library
symm_matmul = load_inline(
    name="symm_matmul",
    cpp_sources=symm_matmul_cpp_source,
    cuda_sources=symm_matmul_source,
    functions=["symm_matmul_cuda"],
    verbose=True,
    extra_ldflags=["-lcublas"],
)


class ModelNew(nn.Module):
    """
    Optimized model that uses a custom cuBLAS kernel for symmetric matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.symm_matmul = symm_matmul

    def forward(self, A, B):
        """
        Performs matrix multiplication of two symmetric matrices using a custom kernel.

        Args:
            A (torch.Tensor): Input matrix A, shape (N, N), symmetric.
            B (torch.Tensor): Input matrix B, shape (N, N), symmetric.

        Returns:
            torch.Tensor: Output matrix C, shape (N, N).
        """
        return self.symm_matmul.symm_matmul_cuda(A, B)

N = 4096

def get_inputs():
    """
    Generates a pair of random symmetric matrices for testing.

    Returns:
        list: List containing two symmetric tensors A and B.
    """
    A = torch.rand(N, N, device='cuda', dtype=torch.float32)
    A = (A + A.T) / 2  # Ensure symmetry
    B = torch.rand(N, N, device='cuda', dtype=torch.float32)
    B = (B + B.T) / 2  # Ensure symmetry
    return [A, B]

def get_init_inputs():
    """
    No specific initialization inputs needed for this model.

    Returns:
        list: Empty list.
    """
    return []