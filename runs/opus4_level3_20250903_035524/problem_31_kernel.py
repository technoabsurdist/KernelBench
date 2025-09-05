import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused LayerNorm + Residual
layernorm_residual_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void layernorm_residual_kernel(
    const T* __restrict__ input,
    const T* __restrict__ residual,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T* __restrict__ output,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int N,
    int D) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * D + tid;
    
    __shared__ float s_mean;
    __shared__ float s_variance;
    
    // Compute mean
    float local_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        local_sum += (float)(input[bid * D + i] + residual[bid * D + i]);
    }
    
    __shared__ float shared_sum[256];
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    if (tid < 128) shared_sum[tid] += shared_sum[tid + 128];
    __syncthreads();
    if (tid < 64) shared_sum[tid] += shared_sum[tid + 64];
    __syncthreads();
    if (tid < 32) shared_sum[tid] += shared_sum[tid + 32];
    __syncthreads();
    if (tid < 16) shared_sum[tid] += shared_sum[tid + 16];
    __syncthreads();
    if (tid < 8) shared_sum[tid] += shared_sum[tid + 8];
    __syncthreads();
    if (tid < 4) shared_sum[tid] += shared_sum[tid + 4];
    __syncthreads();
    if (tid < 2) shared_sum[tid] += shared_sum[tid + 2];
    __syncthreads();
    if (tid == 0) {
        s_mean = (shared_sum[0] + shared_sum[1]) / (float)D;
        mean[bid] = s_mean;
    }
    __syncthreads();
    
    // Compute variance
    float local_var = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float diff = (float)(input[bid * D + i] + residual[bid * D + i]) - s_mean;
        local_var += diff * diff;
    }
    
    shared_sum[tid] = local_var;
    __syncthreads();
    
    if (tid < 128) shared_sum[tid] += shared_sum[tid + 128];
    __syncthreads();
    if (tid < 64) shared_sum[tid] += shared_sum[tid + 64];
    __syncthreads();
    if (tid < 32) shared_sum[tid] += shared_sum[tid + 32];
    __syncthreads();
    if (tid < 16) shared_sum[tid] += shared_sum[tid + 16];
    __syncthreads();
    if (tid < 8) shared_sum[tid] += shared_sum[tid + 8];
    __syncthreads();
    if (tid < 4) shared_sum[tid] += shared_sum[tid + 4];
    __syncthreads();
    if (tid < 2) shared_sum[tid] += shared_sum[tid + 2];
    __syncthreads();
    if (tid == 0) {
        s_variance = (shared_sum[0] + shared_sum[1]) / (float)D;
        float rstd_val = rsqrtf(s_variance + 1e-5f);
        rstd[bid] = rstd_val;
        s_variance = rstd_val;
    }
    __syncthreads();
    
    // Apply normalization
    for (int i = tid; i < D; i += blockDim.x) {
        float val = (float)(input[bid * D + i] + residual[bid * D + i]);
        float normalized = (val - s_mean) * s_variance;
        output[bid * D + i] = (T)(normalized * (float)gamma[i] + (float)beta[i]);
    }
}

torch::Tensor layernorm_residual_cuda(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps) {
    
    const int N = input.size(0);
    const int D = input.size(1);
    
    auto output = torch::empty_like(input);
    auto mean = torch::empty({N}, input.options().dtype(torch::kFloat32));
    auto rstd = torch::empty({N}, input.options().dtype(torch::kFloat32));
    
    const int threads = 256;
    const int blocks = N;
    
    layernorm_residual_kernel<float><<<blocks, threads>>>(
        input.data_ptr<float>(),
        residual.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        N, D);
    
    return output;
}
"""

layernorm_residual_cpp_source = """
torch::Tensor layernorm_residual_cuda(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps);
"""

# Custom CUDA kernel for optimized attention
attention_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

__global__ void softmax_kernel(float* __restrict__ data, int seq_len, int batch_size) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    if (bid >= batch_size * seq_len) return;
    
    float* row = data + bid * seq_len;
    
    // Find max
    float max_val = -INFINITY;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        max_val = fmaxf(max_val, row[i]);
    }
    
    __shared__ float shared_max[256];
    shared_max[tid] = max_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    max_val = shared_max[0];
    
    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float exp_val = expf(row[i] - max_val);
        row[i] = exp_val;
        sum_exp += exp_val;
    }
    
    __shared__ float shared_sum[256];
    shared_sum[tid] = sum_exp;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    sum_exp = shared_sum[0];
    
    // Normalize
    for (int i = tid; i < seq_len; i += blockDim.x) {
        row[i] /= sum_exp;
    }
}

torch::Tensor attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, float scale) {
    const int batch_size = q.size(1);
    const int seq_len = q.size(0);
    const int head_dim = q.size(2);
    
    // Q @ K^T
    auto scores = torch::matmul(q.transpose(0, 1), k.transpose(0, 1).transpose(-2, -1));
    scores = scores * scale;
    
    // Softmax
    scores = scores.contiguous();
    const int threads = 256;
    const int blocks = batch_size * seq_len;
    softmax_kernel<<<blocks, threads>>>(scores.data_ptr<float>(), seq_len, batch_size);
    
    // Scores @ V
    auto output = torch::matmul(scores, v.transpose(0, 1));
    
    return output.transpose(0, 1);
}
"""

attention_cpp_source = """
torch::Tensor attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, float scale);
"""

# Compile the inline CUDA code
layernorm_residual_module = load_inline(
    name="layernorm_residual",
    cpp_sources=layernorm_residual_cpp_source,
    cuda_sources=layernorm_residual_source,
    functions=["layernorm_residual_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

attention_module = load_inline(
    name="attention",
    cpp_sources=attention_cpp_source,
    cuda_sources=attention_source,
    functions=["attention_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(ModelNew, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # LayerNorm parameters
        self.norm_weight = nn.Parameter(torch.ones(embed_dim))
        self.norm_bias = nn.Parameter(torch.zeros(embed_dim))
        
        self.layernorm_residual = layernorm_residual_module
        self.attention = attention_module

    def forward(self, x):
        B, C, H, W = x.shape
        seq_len = H * W
        
        # Reshape and transpose
        x = x.view(B, C, seq_len).permute(2, 0, 1).contiguous()
        
        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(seq_len, B, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Reshape for multi-head attention
        q = q.reshape(seq_len, B * self.num_heads, self.head_dim).contiguous()
        k = k.reshape(seq_len, B * self.num_heads, self.head_dim).contiguous()
        v = v.reshape(seq_len, B * self.num_heads, self.head_dim).contiguous()
        
        # Apply custom attention
        attn_output = self.attention.attention_cuda(q, k, v, self.scale)
        
        # Reshape back
        attn_output = attn_output.reshape(seq_len, B, self.num_heads, self.head_dim)
        attn_output = attn_output.permute(0, 1, 2, 3).reshape(seq_len, B, C)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        # Fused LayerNorm + Residual
        attn_output_flat = attn_output.reshape(seq_len * B, C).contiguous()
        x_flat = x.reshape(seq_len * B, C).contiguous()
        
        x = self.layernorm_residual.layernorm_residual_cuda(
            attn_output_flat, x_flat, self.norm_weight, self.norm_bias, 1e-5
        )
        
        # Reshape back to original shape
        x = x.reshape(seq_len, B, C)
        x = x.permute(1, 2, 0).view(B, C, H, W)
        
        return x

embed_dim = 128
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 128
image_width = 128

def get_inputs():
    return [torch.rand(batch_size, num_channels, image_height, image_width).cuda()]

def get_init_inputs():
    return [embed_dim, num_heads]