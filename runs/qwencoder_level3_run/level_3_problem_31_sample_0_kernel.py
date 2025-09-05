import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused attention + layernorm
attention_layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

__global__ void layernorm_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const float* mean,
    const float* rstd,
    int N,
    int H
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int hidden_idx = threadIdx.x;
    
    int idx = batch_idx * N * H + seq_idx * H + hidden_idx;
    
    if (hidden_idx < H) {
        float val = (input[idx] - mean[batch_idx * N + seq_idx]) * rstd[batch_idx * N + seq_idx];
        output[idx] = val * weight[hidden_idx] + bias[hidden_idx];
    }
}

__global__ void compute_mean_var_kernel(
    const float* input,
    float* mean,
    float* var,
    int N,
    int H
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    
    int start_idx = batch_idx * N * H + seq_idx * H;
    
    float sum = 0.0f;
    for (int i = 0; i < H; i++) {
        sum += input[start_idx + i];
    }
    float m = sum / H;
    mean[batch_idx * N + seq_idx] = m;
    
    float sum_sq = 0.0f;
    for (int i = 0; i < H; i++) {
        float diff = input[start_idx + i] - m;
        sum_sq += diff * diff;
    }
    var[batch_idx * N + seq_idx] = sum_sq / H;
}

__global__ void residual_add_kernel(
    const float* a,
    const float* b,
    float* out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor fused_attention_layernorm_residual_cuda(
    torch::Tensor attn_output,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor bias
) {
    auto sizes = attn_output.sizes();
    int batch_size = sizes[1];
    int seq_len = sizes[0];
    int embed_dim = sizes[2];
    
    // Compute residual addition
    auto output = torch::zeros_like(attn_output);
    int total_elements = attn_output.numel();
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    residual_add_kernel<<<num_blocks, block_size>>>(
        attn_output.data_ptr<float>(),
        residual.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements
    );
    
    // Compute mean and variance for layernorm
    auto mean = torch::zeros({batch_size, seq_len}, torch::kCUDA);
    auto var = torch::zeros({batch_size, seq_len}, torch::kCUDA);
    
    dim3 grid_mean_var(batch_size, seq_len);
    dim3 block_mean_var(1);
    compute_mean_var_kernel<<<grid_mean_var, block_mean_var>>>(
        output.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        seq_len,
        embed_dim
    );
    
    // Compute reciprocal of std
    auto rstd = torch::rsqrt(var + 1e-5);
    
    // Apply layernorm
    auto norm_output = torch::zeros_like(output);
    dim3 grid_norm(batch_size, seq_len);
    dim3 block_norm(embed_dim);
    layernorm_kernel<<<grid_norm, block_norm>>>(
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        norm_output.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        seq_len,
        embed_dim
    );
    
    return norm_output;
}
"""

attention_layernorm_cpp_source = """
torch::Tensor fused_attention_layernorm_residual_cuda(
    torch::Tensor attn_output,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor bias
);
"""

# Compile the inline CUDA code
fused_attention_layernorm = load_inline(
    name="fused_attention_layernorm",
    cpp_sources=attention_layernorm_cpp_source,
    cuda_sources=attention_layernorm_source,
    functions=["fused_attention_layernorm_residual_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Optimized Attention Block using custom CUDA kernels.
        :param embed_dim: Embedding dimension (the number of channels)
        :param num_heads: Number of attention heads
        """
        super(ModelNew, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.fused_kernel = fused_attention_layernorm

    def forward(self, x):
        """
        Forward pass of the optimized AttentionBlock.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of the same shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(2, 0, 1)  # (seq_len, batch_size, embed_dim)
        attn_output, _ = self.attn(x, x, x)
        
        # Use custom fused kernel for residual connection + layernorm
        norm_output = self.fused_kernel.fused_attention_layernorm_residual_cuda(
            attn_output, 
            x, 
            self.norm.weight, 
            self.norm.bias
        )
        
        x = norm_output
        x = x.permute(1, 2, 0).view(B, C, H, W)
        return x