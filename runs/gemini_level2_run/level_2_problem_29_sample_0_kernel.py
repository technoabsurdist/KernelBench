import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear -> mish -> mish
fused_linear_mish_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_DIM 32

// Device function for the Mish activation
__device__ inline float mish_activation(float x) {
    // Mish(x) = x * tanh(softplus(x))
    // softplus(x) = log(1 + exp(x))
    return x * tanhf(logf(1.0f + expf(x)));
}

// Fused kernel for Linear (MatMul + Bias) -> Mish -> Mish
__global__ void linear_mish_mish_kernel(
    const float* A, const float* B, const float* bias, float* C,
    int M, int N, int K)
{
    // Shared memory for tiles of A and B
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global row and column for this thread's output element
    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    // Accumulator for the output element, stored in a register
    float C_val = 0.0f;

    // Loop over tiles in the K dimension
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Load tile of A into shared memory
        int a_row = by * TILE_DIM + ty;
        int a_col = t * TILE_DIM + tx;
        if (a_row < M && a_col < K) {
            sA[ty][tx] = A[a_row * K + a_col];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        int b_row = t * TILE_DIM + ty;
        int b_col = bx * TILE_DIM + tx;
        if (b_row < K && b_col < N) {
            sB[ty][tx] = B[b_row * N + b_col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        // Synchronize to make sure all data is loaded into shared memory
        __syncthreads();

        // Multiply the tiles from shared memory
        for (int k = 0; k < TILE_DIM; ++k) {
            C_val += sA[ty][k] * sB[k][tx];
        }

        // Synchronize before loading the next tiles
        __syncthreads();
    }

    // Check bounds before writing to global memory
    if (row < M && col < N) {
        // Fused epilogue: bias add -> mish -> mish
        float result = C_val + bias[col];
        result = mish_activation(result);
        result = mish_activation(result);
        C[row * N + col] = result;
    }
}

// C++ wrapper function to be called from Python
torch::Tensor linear_mish_mish_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Input weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Input bias must be a CUDA tensor");

    TORCH_CHECK(x.dim() == 2, "Input x must be 2D");
    TORCH_CHECK(weight.dim() == 2, "Input weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "Input bias must be 1D");

    // x: [M, K], weight: [N, K], bias: [N]
    // We need to compute x @ weight.T + bias
    // A = x, B = weight.T
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);

    TORCH_CHECK(weight.size(1) == K, "Dimension mismatch between x and weight");
    TORCH_CHECK(bias.size(0) == N, "Dimension mismatch between weight and bias");

    // PyTorch's nn.Linear weight is [out_features, in_features].
    // The matmul is x @ W.T. So we need to transpose the weight.
    // .contiguous() is important for memory layout.
    auto weight_t = weight.transpose(0, 1).contiguous();

    // Create output tensor
    auto out = torch::empty({M, N}, x.options());

    // Define grid and block dimensions for the CUDA kernel
    const dim3 threads(TILE_DIM, TILE_DIM);
    const dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // Launch the kernel
    linear_mish_mish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight_t.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N, K
    );

    return out;
}
"""

fused_linear_mish_mish_cpp_source = (
    "torch::Tensor linear_mish_mish_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_op_module = load_inline(
    name="fused_op_module",
    cpp_sources=fused_linear_mish_mish_cpp_source,
    cuda_sources=fused_linear_mish_mish_source,
    functions=["linear_mish_mish_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses the matrix multiplication, bias add, and two Mish activations
    into a single custom CUDA kernel.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # We still need nn.Linear to store and manage the weight and bias parameters.
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Call the custom fused CUDA kernel with the input tensor and the linear layer's parameters.
        return fused_op_module.linear_mish_mish_cuda(x, self.linear.weight, self.linear.bias)