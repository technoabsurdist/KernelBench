import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Fused CUDA kernel for Linear -> Swish -> Bias
# This kernel performs the operation: (sigmoid(X @ W.T + b1) * (X @ W.T + b1)) + b2
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// TILE_DIM must be a power of 2, 32 is a common choice
#define TILE_DIM 32

__global__ void fused_linear_swish_bias_kernel(
    const float* x,
    const float* weight,
    const float* linear_bias,
    const float* custom_bias,
    float* out,
    int M, int N, int K) {

    // Each block computes a TILE_DIM x TILE_DIM sub-matrix of the output.
    // Each thread in the block computes one element of this sub-matrix.
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    // Shared memory to hold tiles of the input matrices.
    // This reduces global memory bandwidth usage.
    __shared__ float x_s[TILE_DIM][TILE_DIM];
    __shared__ float w_s[TILE_DIM][TILE_DIM];

    float acc = 0.0f;

    // Loop over the tiles of the input matrices to compute the matrix multiplication.
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Cooperatively load a tile of x into shared memory.
        // Each thread loads one element.
        int x_k = t * TILE_DIM + threadIdx.x;
        if (row < M && x_k < K) {
            x_s[threadIdx.y][threadIdx.x] = x[row * K + x_k];
        } else {
            x_s[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Cooperatively load a tile of the weight matrix into shared memory.
        // The access pattern here is designed to match the multiplication logic below.
        // We are computing x @ W.T, which is sum_k(x[row][k] * W[col][k]).
        int w_row_idx = col;
        int w_col_idx = t * TILE_DIM + threadIdx.y;
        if (w_row_idx < N && w_col_idx < K) {
            w_s[threadIdx.y][threadIdx.x] = weight[w_row_idx * K + w_col_idx];
        } else {
            w_s[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute the dot product for the current tiles.
        // Each thread accumulates its partial sum.
        for (int i = 0; i < TILE_DIM; ++i) {
            acc += x_s[threadIdx.y][i] * w_s[i][threadIdx.x];
        }

        __syncthreads();
    }

    // After computing the matmul, apply the rest of the fused operations.
    // This is only done for threads within the output matrix bounds.
    if (row < M && col < N) {
        // Add the bias from the linear layer.
        acc += linear_bias[col];

        // Apply the Swish activation function: sigmoid(x) * x.
        // expf is the single-precision float version of exp.
        float swish_val = (1.0f / (1.0f + expf(-acc))) * acc;

        // Add the second, custom bias.
        swish_val += custom_bias[col];

        // Write the final result to the output matrix in global memory.
        out[row * N + col] = swish_val;
    }
}

torch::Tensor fused_linear_swish_bias_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor linear_bias,
    torch::Tensor custom_bias) {

    // Input validation checks.
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(linear_bias.is_cuda(), "linear_bias must be a CUDA tensor");
    TORCH_CHECK(custom_bias.is_cuda(), "custom_bias must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");

    // Get matrix dimensions.
    // x: (M, K), weight: (N, K) -> out: (M, N)
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);

    TORCH_CHECK(weight.size(1) == K, "weight dimension mismatch with x");
    TORCH_CHECK(linear_bias.size(0) == N, "linear_bias dimension mismatch");
    TORCH_CHECK(custom_bias.size(0) == N, "custom_bias dimension mismatch");

    // Create the output tensor.
    auto out = torch::zeros({M, N}, x.options());

    // Configure and launch the CUDA kernel.
    dim3 block_dim(TILE_DIM, TILE_DIM);
    dim3 grid_dim(
        (N + TILE_DIM - 1) / TILE_DIM,
        (M + TILE_DIM - 1) / TILE_DIM
    );

    fused_linear_swish_bias_kernel<<<grid_dim, block_dim>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        linear_bias.data_ptr<float>(),
        custom_bias.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N, K
    );
    
    // Check for any errors during kernel execution.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
"""

# C++ source for the function signature.
fused_kernel_cpp_source = """
torch::Tensor fused_linear_swish_bias_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor linear_bias,
    torch::Tensor custom_bias);
"""

# Use torch's JIT compiler to build the custom CUDA operator.
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_linear_swish_bias_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    An optimized model that fuses the matrix multiplication, Swish activation, and bias additions
    into a single custom CUDA kernel. The GroupNorm layer remains a standard PyTorch operator.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        
        # Store the compiled custom operator.
        self.fused_op = fused_op

        # Replicate the parameters from the original model structure.
        # These will be automatically registered as learnable parameters of this module.
        # We create a temporary nn.Linear layer to leverage its default parameter initialization.
        temp_linear = nn.Linear(in_features, out_features)
        self.weight = temp_linear.weight
        self.linear_bias = temp_linear.bias
        
        self.custom_bias = nn.Parameter(torch.randn(bias_shape))
        
        # The GroupNorm layer is kept as is, as its fusion is more complex and may not
        # yield significant benefits without extensive tuning.
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Call the custom fused CUDA kernel.
        x = self.fused_op.fused_linear_swish_bias_cuda(
            x, self.weight, self.linear_bias, self.custom_bias
        )
        # Apply the standard GroupNorm operation.
        x = self.group_norm(x)
        return x