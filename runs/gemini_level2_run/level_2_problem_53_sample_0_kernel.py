import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation
fused_gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_DIM 32

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu_activation(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// Fused Kernel: GEMM (Y = X @ W^T + B), Scale, Hardtanh, GELU
// X: (M, K), W: (N, K), B: (N), Y: (M, N)
__global__ void fused_gemm_scale_hardtanh_gelu_kernel(
    const float* X, const float* W, const float* B, float* Y,
    int M, int N, int K,
    float scaling_factor, float hardtanh_min, float hardtanh_max) {

    // Each thread block computes one tile of the output matrix Y.
    // Each thread in the block computes one element of the tile.
    const int row = blockIdx.y * TILE_DIM + threadIdx.y;
    const int col = blockIdx.x * TILE_DIM + threadIdx.x;

    // Shared memory for tiles of X and W
    __shared__ float X_tile[TILE_DIM][TILE_DIM];
    __shared__ float W_tile[TILE_DIM][TILE_DIM];

    float accumulator = 0.0f;

    // Loop over tiles of X and W to compute the dot product
    for (int k_tile = 0; k_tile < (K + TILE_DIM - 1) / TILE_DIM; ++k_tile) {
        // Load a tile of X into shared memory
        const int x_idx = k_tile * TILE_DIM + threadIdx.x;
        if (row < M && x_idx < K) {
            X_tile[threadIdx.y][threadIdx.x] = X[row * K + x_idx];
        } else {
            X_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load a tile of W into shared memory (transposed access)
        const int w_idx = k_tile * TILE_DIM + threadIdx.y;
        if (col < N && w_idx < K) {
            W_tile[threadIdx.x][threadIdx.y] = W[col * K + w_idx];
        } else {
            W_tile[threadIdx.x][threadIdx.y] = 0.0f;
        }
        __syncthreads();

        // Compute the dot product for the tile
        for (int i = 0; i < TILE_DIM; ++i) {
            accumulator += X_tile[threadIdx.y][i] * W_tile[threadIdx.x][i];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        // Add bias
        accumulator += B[col];

        // --- Fused Epilogue ---
        // 1. Scaling
        accumulator *= scaling_factor;

        // 2. Hardtanh
        accumulator = fmaxf(hardtanh_min, fminf(hardtanh_max, accumulator));

        // 3. GELU
        accumulator = gelu_activation(accumulator);

        // Write the final result to the output matrix Y
        Y[row * N + col] = accumulator;
    }
}

torch::Tensor fused_gemm_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    float scaling_factor, float hardtanh_min, float hardtanh_max) {

    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Input weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Input bias must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input x must be 2D");
    TORCH_CHECK(weight.dim() == 2, "Input weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "Input bias must be 1D");
    TORCH_CHECK(x.size(1) == weight.size(1), "Input x and weight have incompatible dimensions");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Input weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Input bias must be contiguous");

    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);

    auto out = torch::empty({M, N}, x.options());

    const dim3 threads(TILE_DIM, TILE_DIM);
    const dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    fused_gemm_scale_hardtanh_gelu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N, K,
        scaling_factor, hardtanh_min, hardtanh_max
    );
    
    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return out;
}
"""

fused_gemm_cpp_source = """
torch::Tensor fused_gemm_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    float scaling_factor, float hardtanh_min, float hardtanh_max);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_gemm_cpp_source,
    cuda_sources=fused_gemm_source,
    functions=["fused_gemm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Model that performs a fused GEMM, scaling, hardtanh, and GELU activation
    using a custom CUDA kernel.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        # We still need the nn.Linear layer to hold the weight and bias parameters
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        
        # Store the compiled custom operator
        self.fused_op = fused_op

    def forward(self, x):
        # Ensure inputs are on the correct device and contiguous
        x = x.cuda().contiguous()
        weight = self.gemm.weight.contiguous()
        bias = self.gemm.bias.contiguous()

        # Call the custom fused CUDA kernel
        return self.fused_op.fused_gemm_cuda(
            x,
            weight,
            bias,
            self.scaling_factor,
            self.hardtanh_min,
            self.hardtanh_max
        )

batch_size = 2048
in_features = 8192
out_features = 8192
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]