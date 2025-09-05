import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Fused LayerNorm + Residual CUDA kernel
fused_layernorm_residual_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_layernorm_residual_kernel(
    const float* __restrict__ input,
    const float* __restrict__ residual, 
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int N, int D, float eps) {
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    extern __shared__ float shared_mem[];
    float* s_mean = shared_mem;
    float* s_var = &shared_mem[1];
    
    if (tid == 0) {
        s_mean[0] = 0.0f;
        s_var[0] = 0.0f;
    }
    __syncthreads();
    
    // Compute mean
    float local_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        int idx = bid * D + i;
        float val = input[idx] + residual[idx];
        local_sum += val;
    }
    atomicAdd(s_mean, local_sum);
    __syncthreads();
    
    float batch_mean = s_mean[0] / D;
    if (tid == 0) {
        mean[bid] = batch_mean;
    }
    
    // Compute variance
    float local_var = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        int idx = bid * D + i;
        float val = input[idx] + residual[idx] - batch_mean;
        local_var += val * val;
    }
    atomicAdd(s_var, local_var);
    __syncthreads();
    
    float batch_var = s_var[0] / D;
    float batch_rstd = rsqrtf(batch_var + eps);
    if (tid == 0) {
        rstd[bid] = batch_rstd;
    }
    
    // Normalize and apply affine transform
    for (int i = tid; i < D; i += blockDim.x) {
        int idx = bid * D + i;
        float val = input[idx] + residual[idx];
        output[idx] = (val - batch_mean) * batch_rstd * gamma[i] + beta[i];
    }
}

torch::Tensor fused_layernorm_residual_cuda(
    torch::Tensor input, torch::Tensor residual, 
    torch::Tensor gamma, torch::Tensor beta, float eps) {
    
    auto N = input.size(0) * input.size(1);
    auto D = input.size(2);
    auto output = torch::empty_like(input);
    auto mean = torch::empty({N}, input.options());
    auto rstd = torch::empty({N}, input.options());
    
    input = input.contiguous().view({N, D});
    residual = residual.contiguous().view({N, D});
    output = output.view({N, D});
    
    const int threads = 256;
    const int blocks = N;
    size_t shared_size = 2 * sizeof(float);
    
    fused_layernorm_residual_kernel<<<blocks, threads, shared_size>>>(
        input.data_ptr<float>(), residual.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        output.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        N, D, eps
    );
    
    return output.view(input.sizes());
}
"""

fused_layernorm_residual_cpp_source = "torch::Tensor fused_layernorm_residual_cuda(torch::Tensor input, torch::Tensor residual, torch::Tensor gamma, torch::Tensor beta, float eps);"

# Fused MLP with GELU activation
fused_mlp_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float gelu(float x) {
    const float c = 0.797884560803f; // sqrt(2/pi)
    const float a = 0.044715f;
    float x3 = x * x * x;
    float inner = c * (x + a * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void fused_mlp_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight1,
    const float* __restrict__ bias1,
    const float* __restrict__ weight2,
    const float* __restrict__ bias2,
    float* __restrict__ output,
    int batch_size, int seq_len, int d_model, int d_ff) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * d_model;
    
    if (idx < total_elements) {
        int b = idx / (seq_len * d_model);
        int s = (idx / d_model) % seq_len;
        int d = idx % d_model;
        
        // First linear layer + GELU
        float sum = 0.0f;
        for (int i = 0; i < d_ff; i++) {
            float val = bias1[i];
            for (int j = 0; j < d_model; j++) {
                int input_idx = b * seq_len * d_model + s * d_model + j;
                val += input[input_idx] * weight1[i * d_model + j];
            }
            // Apply GELU and second linear layer
            float gelu_out = gelu(val);
            sum += gelu_out * weight2[d * d_ff + i];
        }
        output[idx] = sum + bias2[d];
    }
}

torch::Tensor fused_mlp_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight1, torch::Tensor bias1,
    torch::Tensor weight2, torch::Tensor bias2) {
    
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto d_model = input.size(2);
    auto d_ff = weight1.size(0);
    
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int total_elements = batch_size * seq_len * d_model;
    const int blocks = (total_elements + threads - 1) / threads;
    
    fused_mlp_gelu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight1.data_ptr<float>(), bias1.data_ptr<float>(),
        weight2.data_ptr<float>(), bias2.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, seq_len, d_model, d_ff
    );
    
    return output;
}
"""

fused_mlp_gelu_cpp_source = "torch::Tensor fused_mlp_gelu_cuda(torch::Tensor input, torch::Tensor weight1, torch::Tensor bias1, torch::Tensor weight2, torch::Tensor bias2);"

# Compile CUDA kernels
fused_layernorm_residual = load_inline(
    name="fused_layernorm_residual",
    cpp_sources=fused_layernorm_residual_cpp_source,
    cuda_sources=fused_layernorm_residual_source,
    functions=["fused_layernorm_residual_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

fused_mlp_gelu = load_inline(
    name="fused_mlp_gelu",
    cpp_sources=fused_mlp_gelu_cpp_source,
    cuda_sources=fused_mlp_gelu_source,
    functions=["fused_mlp_gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class OptimizedTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        
        # Multi-head attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # MLP components
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Custom CUDA kernels
        self.fused_layernorm_residual = fused_layernorm_residual
        self.fused_mlp_gelu = fused_mlp_gelu
        
    def forward(self, x):
        # Standard multi-head attention (keeping PyTorch implementation for stability)
        residual = x
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        batch_size, seq_len, _ = q.size()
        q = q.view(batch_size, seq_len, self.nhead, self.d_model // self.nhead).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.d_model // self.nhead).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.d_model // self.nhead).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model // self.nhead)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)
        
        # Fused LayerNorm + Residual
        x = self.fused_layernorm_residual.fused_layernorm_residual_cuda(
            attn_output, residual, 
            self.norm1.weight, self.norm1.bias, 
            self.norm1.eps
        )
        
        # Fused MLP with GELU
        residual = x
        mlp_out = self.fused_mlp_gelu.fused_mlp_gelu_cuda(
            x,
            self.linear1.weight.t().contiguous(),
            self.linear1.bias,
            self.linear2.weight.t().contiguous(), 
            self.linear2.bias
        )
        
        # Final LayerNorm + Residual
        x = self.fused_layernorm_residual.fused_layernorm_residual_cuda(
            mlp_out, residual,
            self.norm2.weight, self.norm2.bias,
            self.norm2.eps
        )
        
        return x


class ModelNew(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6, 
                 mlp_ratio=4.0, patch_size=4, in_channels=3, image_size=32):
        super(ModelNew, self).__init__()
        
        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim
        
        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.linear_proj = nn.Linear(embed_dim * num_patches, embed_dim)
        
        self.transformer_layers = nn.ModuleList([
            OptimizedTransformerLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0
            ) for _ in range(num_layers)
        ])
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc_out = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B = x.size(0)
        x = self.conv1(x)
        x = x.flatten(start_dim=1)
        x = self.linear_proj(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        return self.fc_out(x[:, 0])