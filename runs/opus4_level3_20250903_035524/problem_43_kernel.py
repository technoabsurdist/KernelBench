import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Fused attention kernel with causal masking and online softmax
fused_attention_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

__global__ void fused_attention_kernel(
    const float* q, const float* k, const float* v,
    float* out,
    const int B, const int nh, const int T, const int hs,
    const float scale) {
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int row_idx = blockIdx.x;
    const int col_idx = threadIdx.x;
    
    if (batch_idx >= B || head_idx >= nh || row_idx >= T) return;
    
    extern __shared__ float shared_mem[];
    float* row_max = shared_mem;
    float* row_sum = &shared_mem[1];
    float* att_row = &shared_mem[2];
    
    // Initialize shared memory
    if (col_idx == 0) {
        row_max[0] = -FLT_MAX;
        row_sum[0] = 0.0f;
    }
    __syncthreads();
    
    // Compute attention scores for this row with causal masking
    float local_max = -FLT_MAX;
    for (int t = col_idx; t <= row_idx && t < T; t += blockDim.x) {
        float score = 0.0f;
        for (int d = 0; d < hs; d++) {
            int q_idx = batch_idx * nh * T * hs + head_idx * T * hs + row_idx * hs + d;
            int k_idx = batch_idx * nh * T * hs + head_idx * T * hs + t * hs + d;
            score += q[q_idx] * k[k_idx];
        }
        score *= scale;
        att_row[t] = score;
        local_max = fmaxf(local_max, score);
    }
    
    // Reduce to find row max
    atomicMax(reinterpret_cast<int*>(row_max), __float_as_int(local_max));
    __syncthreads();
    
    // Compute exp and sum for softmax
    float local_sum = 0.0f;
    for (int t = col_idx; t <= row_idx && t < T; t += blockDim.x) {
        float exp_val = expf(att_row[t] - row_max[0]);
        att_row[t] = exp_val;
        local_sum += exp_val;
    }
    
    // Reduce sum
    atomicAdd(row_sum, local_sum);
    __syncthreads();
    
    // Normalize and compute output
    for (int d = col_idx; d < hs; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t <= row_idx && t < T; t++) {
            int v_idx = batch_idx * nh * T * hs + head_idx * T * hs + t * hs + d;
            acc += (att_row[t] / row_sum[0]) * v[v_idx];
        }
        int out_idx = batch_idx * nh * T * hs + head_idx * T * hs + row_idx * hs + d;
        out[out_idx] = acc;
    }
}

torch::Tensor fused_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    const auto B = q.size(0);
    const auto nh = q.size(1);
    const auto T = q.size(2);
    const auto hs = q.size(3);
    
    auto out = torch::zeros_like(q);
    const float scale = 1.0f / sqrtf(static_cast<float>(hs));
    
    dim3 blocks(T, nh, B);
    int threads = min(256, T);
    int shared_mem_size = (2 + T) * sizeof(float);
    
    fused_attention_kernel<<<blocks, threads, shared_mem_size>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        out.data_ptr<float>(),
        B, nh, T, hs, scale
    );
    
    return out;
}
"""

fused_attention_cpp_source = "torch::Tensor fused_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v);"

# Fused linear projection kernel
fused_linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void add_bias_kernel(float* out, const float* bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = idx % N;
    if (idx < M * N) {
        out[idx] += bias[col];
    }
}

torch::Tensor fused_linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    const int B = input.size(0);
    const int T = input.size(1);
    const int in_features = input.size(2);
    const int out_features = weight.size(0);
    
    auto input_2d = input.view({B * T, in_features});
    auto output = torch::zeros({B * T, out_features}, input.options());
    
    // Use cuBLAS for matrix multiplication
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, B * T, in_features,
                &alpha,
                weight.data_ptr<float>(), in_features,
                input_2d.data_ptr<float>(), in_features,
                &beta,
                output.data_ptr<float>(), out_features);
    
    // Add bias
    int threads = 256;
    int blocks = (B * T * out_features + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(), bias.data_ptr<float>(), B * T, out_features
    );
    
    cublasDestroy(handle);
    
    return output.view({B, T, out_features});
}
"""

fused_linear_cpp_source = "torch::Tensor fused_linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Load inline CUDA kernels
fused_attention = load_inline(
    name="fused_attention",
    cpp_sources=fused_attention_cpp_source,
    cuda_sources=fused_attention_source,
    functions=["fused_attention_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""]
)

fused_linear = load_inline(
    name="fused_linear",
    cpp_sources=fused_linear_cpp_source,
    cuda_sources=fused_linear_source,
    functions=["fused_linear_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"]
)


class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        self.fused_attention = fused_attention
        self.fused_linear = fused_linear

    def forward(self, x):
        B, T, C = x.size()
        
        # QKV projection using fused linear
        qkv = self.fused_linear.fused_linear_cuda(
            x, self.c_attn.weight, self.c_attn.bias
        )
        
        # Split and reshape
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Fused attention computation
        y = self.fused_attention.fused_attention_cuda(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.fused_linear.fused_linear_cuda(
            y, self.c_proj.weight, self.c_proj.bias
        )
        y = self.resid_dropout(y)
        
        return y

batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [torch.rand(batch_size, seq_len, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]