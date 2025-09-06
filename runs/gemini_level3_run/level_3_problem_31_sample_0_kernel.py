import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA source for the fused add + layernorm kernel
fused_add_layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// A LayerNorm kernel that also handles the preceding residual addition.
// This fusion reduces memory bandwidth by avoiding the need to write the
// result of the addition back to global memory before the LayerNorm.
//
// Template parameter T: data type (e.g., float, half)
// Kernel launch configuration:
//   - Grid: (N), where N is the number of rows (e.g., seq_len * batch_size)
//   - Block: (BlockSize), where BlockSize is a power of 2 (e.g., 256)
template <typename T>
__global__ void add_layernorm_kernel(
    const T* __restrict__ attn_output,
    const T* __restrict__ x,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T* __restrict__ out,
    int N,
    int D,
    float epsilon) {

    // N is the total number of vectors (e.g., seq_len * batch_size)
    // D is the feature dimension (embed_dim)
    int row_idx = blockIdx.x;
    if (row_idx >= N) return;

    // Shared memory for block-level reduction
    extern __shared__ float sdata[];
    float* s_mean = sdata;
    float* s_var = &sdata[blockDim.x];

    int tid = threadIdx.x;
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // --- 1. Calculate sum and sum of squares for the current row ---
    // Each thread processes multiple elements if D > blockDim.x
    for (int i = tid; i < D; i += blockDim.x) {
        int col_idx = i;
        int global_idx = row_idx * D + col_idx;
        // Fused operation: add residual `x` to `attn_output`
        float val = (float)attn_output[global_idx] + (float)x[global_idx];
        local_sum += val;
        local_sum_sq += val * val;
    }

    s_mean[tid] = local_sum;
    s_var[tid] = local_sum_sq;
    __syncthreads();

    // --- 2. Parallel Reduction in Shared Memory ---
    // Reduce the sums calculated by each thread in the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_mean[tid] += s_mean[tid + s];
            s_var[tid] += s_var[tid + s];
        }
        __syncthreads();
    }

    // --- 3. Calculate mean and variance (thread 0) ---
    if (tid == 0) {
        float mean = s_mean[0] / D;
        float var = s_var[0] / D - mean * mean;
        s_mean[0] = mean;
        s_var[0] = rsqrtf(var + epsilon); // Store 1/sqrt(var + eps) for efficiency
    }
    __syncthreads();

    float mean_val = s_mean[0];
    float rsqrt_var_val = s_var[0];

    // --- 4. Normalize and apply affine transform (gamma, beta) ---
    for (int i = tid; i < D; i += blockDim.x) {
        int col_idx = i;
        int global_idx = row_idx * D + col_idx;
        float val = (float)attn_output[global_idx] + (float)x[global_idx];
        float normalized_val = (val - mean_val) * rsqrt_var_val;
        out[global_idx] = (T)(normalized_val * (float)gamma[col_idx] + (float)beta[col_idx]);
    }
}

// C++ wrapper function to be bound with PyTorch
torch::Tensor add_layernorm_forward_cuda(
    torch::Tensor attn_output,
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    float epsilon) {

    // Input validation
    TORCH_CHECK(attn_output.is_cuda(), "attn_output must be a CUDA tensor");
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "beta must be a CUDA tensor");
    TORCH_CHECK(attn_output.is_contiguous(), "attn_output must be contiguous");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(beta.is_contiguous(), "beta must be contiguous");
    TORCH_CHECK(attn_output.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");

    auto out = torch::empty_like(attn_output);
    auto last_dim = attn_output.dim() - 1;
    auto D = attn_output.size(last_dim); // The dimension to normalize over
    auto N = attn_output.numel() / D;    // The number of vectors to normalize

    const int block_size = 256;
    const int num_blocks = N;
    // Shared memory for two float arrays of size block_size (for mean and var reduction)
    const int shared_mem_size = 2 * block_size * sizeof(float);

    add_layernorm_kernel<float><<<num_blocks, block_size, shared_mem_size>>>(
        attn_output.data_ptr<float>(),
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D,
        epsilon
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# C++ source for function signature, required by load_inline
fused_add_layernorm_cpp_source = """
torch::Tensor add_layernorm_forward_cuda(
    torch::Tensor attn_output,
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    float epsilon);
"""

# JIT compile the CUDA kernel
fused_add_layernorm = load_inline(
    name="fused_add_layernorm",
    cpp_sources=fused_add_layernorm_cpp_source,
    cuda_sources=fused_add_layernorm_source,
    functions=["add_layernorm_forward_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, eps=1e-5):
        """
        Attention Block using Multihead Self-Attention with a custom fused Add+LayerNorm kernel.
        :param embed_dim: Embedding dimension (the number of channels)
        :param num_heads: Number of attention heads
        :param eps: Epsilon for LayerNorm
        """
        super(ModelNew, self).__init__()
        # Use PyTorch's highly optimized MultiheadAttention implementation
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.eps = eps

        # Replace nn.LayerNorm with our custom fused kernel.
        # We still need the learnable affine parameters (gamma and beta).
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        """
        Forward pass of the AttentionBlock.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of the same shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        # Reshape and permute for attention: (B, C, H*W) -> (H*W, B, C)
        # This matches the expected input format for nn.MultiheadAttention(batch_first=False)
        x_reshaped = x.view(B, C, H * W).permute(2, 0, 1)

        # Apply multi-head attention
        attn_output, _ = self.attn(x_reshaped, x_reshaped, x_reshaped)

        # Apply our custom fused residual add + layernorm kernel.
        # The kernel expects contiguous tensors, so we ensure that.
        # x_reshaped is from a permute, so it's not contiguous.
        x_norm = fused_add_layernorm.add_layernorm_forward_cuda(
            attn_output.contiguous(),
            x_reshaped.contiguous(),
            self.gamma,
            self.beta,
            self.eps
        )

        # Reshape and permute back to original shape: (H*W, B, C) -> (B, C, H*W) -> (B, C, H, W)
        x_out = x_norm.permute(1, 2, 0).view(B, C, H, W)
        return x_out