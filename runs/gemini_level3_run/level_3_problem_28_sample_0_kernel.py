import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for Fused Linear + GELU
linear_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_DIM 16

// Device function for GELU activation using the error function, which is standard in PyTorch
__device__ float gelu_erf_impl(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f)); // 0.707... is 1/sqrt(2)
}

// CUDA kernel for tiled matrix multiplication (X @ W.T + B) fused with GELU activation
// It computes Y(M,N) = GELU(X(M,K) @ W.T(K,N) + B(N)), where W is stored as (N, K)
__global__ void linear_gelu_kernel(const float* X, const float* W, const float* bias, float* out, int M, int N, int K) {
    const int row = blockIdx.y * TILE_DIM + threadIdx.y;
    const int col = blockIdx.x * TILE_DIM + threadIdx.x;

    __shared__ float X_tile[TILE_DIM][TILE_DIM];
    __shared__ float W_tile[TILE_DIM][TILE_DIM];

    float acc = 0.0f;

    const int num_k_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_base = k_tile * TILE_DIM;

        // Load a tile of X from global to shared memory (coalesced access)
        const int x_k = k_base + threadIdx.x;
        if (row < M && x_k < K) {
            X_tile[threadIdx.y][threadIdx.x] = X[row * K + x_k];
        } else {
            X_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load a tile of W from global to shared memory (coalesced access)
        const int w_k = k_base + threadIdx.x;
        if (col < N && w_k < K) {
            W_tile[threadIdx.y][threadIdx.x] = W[col * K + w_k];
        } else {
            W_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product from shared memory
        for (int j = 0; j < TILE_DIM; ++j) {
            acc += X_tile[threadIdx.y][j] * W_tile[threadIdx.x][j];
        }

        __syncthreads();
    }

    // After all k-tiles, add bias, apply GELU, and write to output
    if (row < M && col < N) {
        acc += bias[col];
        out[row * N + col] = gelu_erf_impl(acc);
    }
}

// C++ wrapper function to be called from Python
torch::Tensor linear_gelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");

    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weight.size(0);

    TORCH_CHECK(weight.size(1) == K, "Weight shape mismatch: weight.size(1) != input.size(1)");
    TORCH_CHECK(bias.size(0) == N, "Bias shape mismatch: bias.size(0) != weight.size(0)");

    auto out = torch::empty({M, N}, input.options());

    const dim3 threads(TILE_DIM, TILE_DIM);
    const dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    linear_gelu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N, K
    );
    
    // Check for errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
"""

linear_gelu_cpp_source = "torch::Tensor linear_gelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# JIT compile the CUDA kernel using load_inline
linear_gelu = load_inline(
    name="linear_gelu",
    cpp_sources=linear_gelu_cpp_source,
    cuda_sources=linear_gelu_source,
    functions=["linear_gelu_cuda"],
    verbose=False,
)

class FusedLinearGELU(nn.Module):
    """
    A custom nn.Module that fuses a Linear layer and a GELU activation using a custom CUDA kernel.
    This is a drop-in replacement for nn.Sequential(nn.Linear(in_features, out_features), nn.GELU()).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize parameters with the same method as the standard PyTorch Linear layer
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The input tensor must be contiguous for data_ptr() to work correctly
        x_contiguous = x.contiguous()
        return linear_gelu.linear_gelu_cuda(x_contiguous, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        """
        Optimized Vision Transformer (ViT) model with a custom FusedLinearGELU operator.
        """
        super(ModelNew, self).__init__()
        
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # Use batch_first=True and GELU activation for a standard ViT implementation
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, activation='gelu', batch_first=True),
            num_layers=depth
        )
        
        self.to_cls_token = nn.Identity()
        
        # Replace the first Linear and GELU in the MLP head with our fused operator
        self.mlp_head = nn.Sequential(
            FusedLinearGELU(dim, mlp_dim),
            # The nn.GELU() is now fused into the custom layer above
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )
    
    def forward(self, img):
        """
        Forward pass of the Vision Transformer.
        """
        p = self.patch_size
        b, c, h, w = img.shape
        
        # Correctly reshape image into patches: (B, C, H, W) -> (B, N, P*P*C)
        x = img.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(b, -1, c * p * p)
        
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)