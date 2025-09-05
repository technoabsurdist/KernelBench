import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused attention computation: (Q @ K^T) * scale + mask + ReLU
fused_attention_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void fused_attention_kernel(
    const float* q, 
    const float* k, 
    const float* mask,
    float* out, 
    int B, int nh, int T, int hs,
    float scale) {
    
    // Each block handles one (batch, head, query_pos, key_pos) element
    int batch = blockIdx.x / (nh * T);
    int head = (blockIdx.x / T) % nh;
    int query_pos = blockIdx.x % T;
    int key_pos = threadIdx.x;
    
    if (batch >= B || head >= nh || query_pos >= T || key_pos >= T) return;
    
    extern __shared__ float shared_mem[];
    float* q_shared = shared_mem;
    float* k_shared = &shared_mem[hs];
    
    // Load query vector for this position into shared memory
    if (key_pos < hs) {
        q_shared[key_pos] = q[batch * nh * T * hs + head * T * hs + query_pos * hs + key_pos];
    }
    __syncthreads();
    
    // Load key vector for this position
    if (key_pos < hs) {
        k_shared[key_pos] = k[batch * nh * T * hs + head * T * hs + key_pos * hs + threadIdx.x];
    }
    
    // Compute dot product
    float sum = 0.0f;
    for (int i = 0; i < hs; i++) {
        if (key_pos < T) {
            sum += q_shared[i] * k[batch * nh * T * hs + head * T * hs + key_pos * hs + i];
        }
    }
    
    // Apply scale
    sum *= scale;
    
    // Apply mask
    float mask_val = mask[query_pos * T + key_pos];
    if (mask_val == 0) {
        sum = -1e9f;
    }
    
    // Apply ReLU
    sum = fmaxf(0.0f, sum);
    
    // Write output
    if (key_pos < T) {
        out[batch * nh * T * T + head * T * T + query_pos * T + key_pos] = sum;
    }
}

torch::Tensor fused_attention_cuda(
    torch::Tensor q,
    torch::Tensor k, 
    torch::Tensor mask,
    float scale) {
    
    auto B = q.size(0);
    auto nh = q.size(1);
    auto T = q.size(2);
    auto hs = q.size(3);
    
    auto out = torch::zeros({B, nh, T, T}, q.options());
    
    dim3 blocks(B * nh * T);
    dim3 threads(T);
    size_t shared_size = 2 * hs * sizeof(float);
    
    fused_attention_kernel<<<blocks, threads, shared_size>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        mask.data_ptr<float>(),
        out.data_ptr<float>(),
        B, nh, T, hs, scale
    );
    
    return out;
}
"""

fused_attention_cpp_source = "torch::Tensor fused_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor mask, float scale);"

# Custom CUDA kernel for optimized batched matrix multiplication
batched_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

torch::Tensor batched_matmul_cuda(torch::Tensor a, torch::Tensor b) {
    auto batch_size = a.size(0) * a.size(1);
    auto m = a.size(2);
    auto k = a.size(3);
    auto n = b.size(3);
    
    auto c = torch::zeros({a.size(0), a.size(1), m, n}, a.options());
    
    // Use cuBLAS for efficient batched matrix multiplication
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        b.data_ptr<float>(), n, k * n,
        a.data_ptr<float>(), k, m * k,
        &beta,
        c.data_ptr<float>(), n, m * n,
        batch_size
    );
    
    cublasDestroy(handle);
    return c;
}
"""

batched_matmul_cpp_source = "torch::Tensor batched_matmul_cuda(torch::Tensor a, torch::Tensor b);"

# Compile the inline CUDA code
fused_attention = load_inline(
    name="fused_attention",
    cpp_sources=fused_attention_cpp_source,
    cuda_sources=fused_attention_source,
    functions=["fused_attention_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

batched_matmul = load_inline(
    name="batched_matmul",
    cpp_sources=batched_matmul_cpp_source,
    cuda_sources=batched_matmul_source,
    functions=["batched_matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class NewGELU(nn.Module):
    def __init__(self):
        super(NewGELU, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        self.fused_attention = fused_attention
        self.batched_matmul = batched_matmul

    def forward(self, x):
        B, T, C = x.size()
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Use fused attention kernel
        scale = 1.0 / math.sqrt(k.size(-1))
        att = self.fused_attention.fused_attention_cuda(
            q.contiguous(), 
            k.contiguous(), 
            self.bias[:,:,:T,:T].squeeze(0).squeeze(0).contiguous(),
            scale
        )
        
        # Use optimized batched matmul
        y = self.batched_matmul.batched_matmul_cuda(att.contiguous(), v.contiguous())
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return y

batch_size = 16
max_seqlen = 1024
n_embd = 768
n_head = 12

def get_inputs():
    return [torch.rand(batch_size, max_seqlen, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, max_seqlen]