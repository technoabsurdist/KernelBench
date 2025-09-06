import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: GEMM + BatchNorm + GELU + ReLU
fused_gemm_bn_gelu_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Using a tile size for shared memory optimization in GEMM
#define TILE_DIM 32

// Device function for GELU approximation
__device__ __forceinline__ float gelu_fn(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// The fused kernel
__global__ void fused_gemm_bn_gelu_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ bn_gamma,
    const float* __restrict__ bn_beta,
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    float* __restrict__ out,
    int M, int N, int K,
    float bn_eps) {

    // M: batch_size (rows of X and out)
    // N: out_features (cols of W and out)
    // K: in_features (cols of X and rows of W)

    __shared__ float x_tile[TILE_DIM][TILE_DIM];
    __shared__ float w_tile[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float acc = 0.0f;

    // Loop over tiles of X and W to compute the dot product
    for (int i = 0; i < (K + TILE_DIM - 1) / TILE_DIM; ++i) {
        // Load a tile of X into shared memory
        int x_k = i * TILE_DIM + threadIdx.x;
        if (row < M && x_k < K) {
            x_tile[threadIdx.y][threadIdx.x] = x[row * K + x_k];
        } else {
            x_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load a tile of W into shared memory
        int w_k = i * TILE_DIM + threadIdx.y;
        if (col < N && w_k < K) {
            // weight is stored as (out_features, in_features) -> (N, K)
            w_tile[threadIdx.y][threadIdx.x] = weight[col * K + w_k];
        } else {
            w_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute the dot product for the tile.
        // This check prevents threads outside the output matrix from doing computation.
        if (row < M && col < N) {
            for (int j = 0; j < TILE_DIM; ++j) {
                // acc += X[row, k_base+j] * W[col, k_base+j]
                acc += x_tile[threadIdx.y][j] * w_tile[j][threadIdx.x];
            }
        }

        __syncthreads();
    }

    // After GEMM, perform fused post-processing operations
    if (row < M && col < N) {
        // 1. GEMM: Add bias
        acc += bias[col];

        // 2. BatchNorm (inference mode)
        float inv_std = rsqrtf(bn_var[col] + bn_eps);
        float bn_out = bn_gamma[col] * (acc - bn_mean[col]) * inv_std + bn_beta[col];

        // 3. GELU
        float gelu_out = gelu_fn(bn_out);

        // 4. ReLU
        float relu_out = fmaxf(0.0f, gelu_out);

        // Write final result to output
        out[row * N + col] = relu_out;
    }
}

// C++ wrapper function to be called from Python
torch::Tensor fused_gemm_bn_gelu_relu_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_gamma,
    torch::Tensor bn_beta,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    double bn_eps) {

    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Input weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Input bias must be a CUDA tensor");
    TORCH_CHECK(bn_gamma.is_cuda(), "Input bn_gamma must be a CUDA tensor");
    TORCH_CHECK(bn_beta.is_cuda(), "Input bn_beta must be a CUDA tensor");
    TORCH_CHECK(bn_mean.is_cuda(), "Input bn_mean must be a CUDA tensor");
    TORCH_CHECK(bn_var.is_cuda(), "Input bn_var must be a CUDA tensor");
    
    TORCH_CHECK(x.dtype() == torch::kFloat32, "All input tensors must be float32");
    // Add checks for other tensors as well for robustness

    TORCH_CHECK(x.dim() == 2, "Input x must be 2D");
    TORCH_CHECK(weight.dim() == 2, "Input weight must be 2D");

    const int M = x.size(0); // batch_size
    const int K = x.size(1); // in_features
    const int N = weight.size(0); // out_features

    TORCH_CHECK(K == weight.size(1), "Dimension mismatch: x.size(1) != weight.size(1)");

    // Create output tensor
    auto out = torch::empty({M, N}, x.options());

    // Grid and block dimensions
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks(
        (N + TILE_DIM - 1) / TILE_DIM,
        (M + TILE_DIM - 1) / TILE_DIM
    );

    // Launch kernel
    fused_gemm_bn_gelu_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        bn_gamma.data_ptr<float>(),
        bn_beta.data_ptr<float>(),
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N, K,
        static_cast<float>(bn_eps)
    );
    
    // Check for CUDA errors after kernel launch for safety
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

fused_gemm_bn_gelu_relu_cpp_source = """
torch::Tensor fused_gemm_bn_gelu_relu_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_gamma,
    torch::Tensor bn_beta,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    double bn_eps);
"""

# JIT compile the inline CUDA code
fused_op = load_inline(
    name="fused_gemm_bn_gelu_relu_op",
    cpp_sources=fused_gemm_bn_gelu_relu_cpp_source,
    cuda_sources=fused_gemm_bn_gelu_relu_source,
    functions=["fused_gemm_bn_gelu_relu_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Model that performs a fused GEMM, BatchNorm, GELU, and ReLU in sequence
    using a custom CUDA kernel for inference. Falls back to the standard
    PyTorch implementation during training.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # These layers hold the parameters (weights, biases, etc.) and buffers
        # which will be passed to our custom kernel.
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # The custom kernel is optimized for inference.
        # During training, we need the standard PyTorch layers to compute
        # gradients and update batch norm statistics correctly.
        if self.training:
            x = self.gemm(x)
            x = self.batch_norm(x)
            x = torch.nn.functional.gelu(x)
            x = torch.relu(x)
            return x
        else:
            # In inference mode, use the high-performance fused kernel.
            return fused_op.fused_gemm_bn_gelu_relu_cuda(
                x,
                self.gemm.weight,
                self.gemm.bias,
                self.batch_norm.weight,
                self.batch_norm.bias,
                self.batch_norm.running_mean,
                self.batch_norm.running_var,
                self.batch_norm.eps
            )