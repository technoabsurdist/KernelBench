import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for NewGELU activation
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void gelu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (val + 0.044715f * powf(val, 3.0f))));
        out[idx] = val * cdf;
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    
    return out;
}
"""

gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor x);"

# Custom CUDA kernel for fused attention QKV projection
qkv_proj_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void qkv_proj_kernel(
    const float* x,
    const float* w_qkv,
    const float* b_qkv,
    float* q_out,
    float* k_out,
    float* v_out,
    int batch_size,
    int seq_len,
    int n_embd,
    int hidden_size
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int dim_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || dim_idx >= n_embd) return;
    
    int x_idx = batch_idx * seq_len * n_embd + seq_idx * n_embd + dim_idx;
    float x_val = x[x_idx];
    
    // Compute Q
    float q_val = b_qkv[dim_idx];
    for (int i = 0; i < n_embd; i++) {
        int w_idx = dim_idx * 3 * n_embd + i;
        q_val += x_val * w_qkv[w_idx];
    }
    q_out[batch_idx * seq_len * n_embd + seq_idx * n_embd + dim_idx] = q_val;
    
    // Compute K
    float k_val = b_qkv[n_embd + dim_idx];
    for (int i = 0; i < n_embd; i++) {
        int w_idx = dim_idx * 3 * n_embd + n_embd + i;
        k_val += x_val * w_qkv[w_idx];
    }
    k_out[batch_idx * seq_len * n_embd + seq_idx * n_embd + dim_idx] = k_val;
    
    // Compute V
    float v_val = b_qkv[2 * n_embd + dim_idx];
    for (int i = 0; i < n_embd; i++) {
        int w_idx = dim_idx * 3 * n_embd + 2 * n_embd + i;
        v_val += x_val * w_qkv[w_idx];
    }
    v_out[batch_idx * seq_len * n_embd + seq_idx * n_embd + dim_idx] = v_val;
}

torch::Tensor fused_qkv_projection(
    torch::Tensor x,
    torch::Tensor w_qkv,
    torch::Tensor b_qkv
) {
    int batch_size = x.size(0);
    int seq_len = x.size(1);
    int n_embd = x.size(2);
    
    auto q = torch::zeros({batch_size, seq_len, n_embd}, x.options());
    auto k = torch::zeros({batch_size, seq_len, n_embd}, x.options());
    auto v = torch::zeros({batch_size, seq_len, n_embd}, x.options());
    
    dim3 grid(batch_size, seq_len);
    dim3 block(n_embd);
    
    qkv_proj_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w_qkv.data_ptr<float>(),
        b_qkv.data_ptr<float>(),
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        batch_size,
        seq_len,
        n_embd,
        n_embd
    );
    
    auto result = torch::cat({q, k, v}, -1);
    return result;
}
"""

qkv_proj_cpp_source = "torch::Tensor fused_qkv_projection(torch::Tensor x, torch::Tensor w_qkv, torch::Tensor b_qkv);"

# Compile the inline CUDA code
gelu = load_inline(
    name="gelu",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=False
)

qkv_proj = load_inline(
    name="qkv_proj",
    cpp_sources=qkv_proj_cpp_source,
    cuda_sources=qkv_proj_source,
    functions=["fused_qkv_projection"],
    verbose=False
)

class NewGELU(nn.Module):
    def __init__(self):
        super(NewGELU, self).__init__()
        self.gelu_func = gelu
    
    def forward(self, x):
        return self.gelu_func.gelu_cuda(x)

class CausalSelfAttention(nn.Module):
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
        self.qkv_proj_func = qkv_proj

    def forward(self, x):
        B, T, C = x.size()

        # Use custom fused QKV projection
        qkv = self.qkv_proj_func.fused_qkv_projection(
            x,
            self.c_attn.weight.t().contiguous(),
            self.c_attn.bias
        )
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y

class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [torch.rand(batch_size, seq_len, n_embd).cuda()]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]