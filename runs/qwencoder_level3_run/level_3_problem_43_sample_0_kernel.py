import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused attention
attention_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

__global__ void fused_qkv_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w_qkv,
    const float* __restrict__ b_qkv,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    int B, int T, int C, int OC) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * T * OC;
    
    if (idx < total_elements) {
        int b = idx / (T * OC);
        int t = (idx / OC) % T;
        int oc = idx % OC;
        
        float sum = b_qkv[oc];
        for (int c = 0; c < C; c++) {
            sum += x[b * T * C + t * C + c] * w_qkv[c * OC + oc];
        }
        int base_idx = b * T * OC + t * OC + oc;
        if (oc < OC / 3) {
            q[base_idx] = sum;
        } else if (oc < 2 * OC / 3) {
            k[base_idx] = sum;
        } else {
            v[base_idx] = sum;
        }
    }
}

__global__ void transpose_and_scale_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    float* __restrict__ q_transposed,
    float* __restrict__ k_transposed,
    float scale,
    int B, int nh, int T, int hs) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * nh * T * hs;
    
    if (idx < total_elements) {
        int b = idx / (nh * T * hs);
        int h = (idx / (T * hs)) % nh;
        int t = (idx / hs) % T;
        int s = idx % hs;
        
        int src_idx = b * (nh * T * hs) + t * (nh * hs) + h * hs + s;
        int dst_idx = b * (nh * T * hs) + h * (T * hs) + t * hs + s;
        
        if (q) q_transposed[dst_idx] = q[src_idx] * scale;
        if (k) k_transposed[dst_idx] = k[src_idx];
    }
}

__global__ void causal_mask_kernel(
    float* __restrict__ att,
    const int* __restrict__ mask,
    int B, int nh, int T) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * nh * T * T;
    
    if (idx < total_elements) {
        int b = idx / (nh * T * T);
        int h = (idx / (T * T)) % nh;
        int i = (idx / T) % T;
        int j = idx % T;
        
        if (mask[i * T + j] == 0) {
            att[idx] = -1e10f;
        }
    }
}

__global__ void softmax_kernel(
    float* __restrict__ att,
    int B, int nh, int T) {
    
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int row_idx = threadIdx.x;
    
    if (batch_idx >= B || head_idx >= nh || row_idx >= T) return;
    
    int offset = batch_idx * (nh * T * T) + head_idx * (T * T) + row_idx * T;
    
    // Find max
    float max_val = att[offset];
    for (int i = 1; i < T; i++) {
        max_val = fmaxf(max_val, att[offset + i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < T; i++) {
        float val = expf(att[offset + i] - max_val);
        att[offset + i] = val;
        sum += val;
    }
    
    // Normalize
    for (int i = 0; i < T; i++) {
        att[offset + i] /= sum;
    }
}

__global__ void output_projection_kernel(
    const float* __restrict__ y,
    const float* __restrict__ w_out,
    const float* __restrict__ b_out,
    float* __restrict__ out,
    int B, int T, int C) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * T * C;
    
    if (idx < total_elements) {
        int b = idx / (T * C);
        int t = (idx / C) % T;
        int c = idx % C;
        
        float sum = b_out[c];
        for (int i = 0; i < C; i++) {
            sum += y[b * T * C + t * C + i] * w_out[i * C + c];
        }
        out[idx] = sum;
    }
}

torch::Tensor fused_attention_cuda(
    torch::Tensor x,
    torch::Tensor w_qkv,
    torch::Tensor b_qkv,
    torch::Tensor w_out,
    torch::Tensor b_out,
    torch::Tensor mask,
    int n_head,
    float dropout_p) {
    
    auto B = x.size(0);
    auto T = x.size(1);
    auto C = x.size(2);
    auto OC = w_qkv.size(1);
    
    // Allocate intermediate tensors
    auto q = torch::zeros({B, T, OC}, x.options());
    auto k = torch::zeros({B, T, OC}, x.options());
    auto v = torch::zeros({B, T, OC}, x.options());
    
    // Step 1: Fused QKV projection
    const int block_size = 256;
    const int num_blocks_qkv = CEIL_DIV(B * T * OC, block_size);
    fused_qkv_kernel<<<num_blocks_qkv, block_size>>>(
        x.data_ptr<float>(),
        w_qkv.data_ptr<float>(),
        b_qkv.data_ptr<float>(),
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        B, T, C, OC
    );
    
    auto nh = n_head;
    auto hs = C / nh;
    
    // Reshape and transpose
    q = q.view({B, T, nh, hs}).transpose(1, 2); // (B, nh, T, hs)
    k = k.view({B, T, nh, hs}).transpose(1, 2); // (B, nh, T, hs)
    v = v.view({B, T, nh, hs}).transpose(1, 2); // (B, nh, T, hs)
    
    // Step 2: Attention computation
    auto k_t = k.transpose(-2, -1); // (B, nh, hs, T)
    auto att = torch::matmul(q, k_t); // (B, nh, T, T)
    auto scale = 1.0f / sqrtf(static_cast<float>(hs));
    att = att * scale;
    
    // Apply causal mask
    auto mask_view = mask.slice(2, 0, T).slice(3, 0, T).expand_as(att);
    att = att.masked_fill(mask_view == 0, -1e10f);
    
    // Softmax
    att = torch::softmax(att, -1);
    
    // Dropout
    if (dropout_p > 0.0f) {
        att = torch::dropout(att, dropout_p, true);
    }
    
    // Step 3: Value multiplication
    auto y = torch::matmul(att, v); // (B, nh, T, hs)
    y = y.transpose(1, 2).contiguous().view({B, T, C}); // (B, T, C)
    
    // Step 4: Output projection
    auto out = torch::zeros({B, T, C}, x.options());
    const int num_blocks_out = CEIL_DIV(B * T * C, block_size);
    output_projection_kernel<<<num_blocks_out, block_size>>>(
        y.data_ptr<float>(),
        w_out.data_ptr<float>(),
        b_out.data_ptr<float>(),
        out.data_ptr<float>(),
        B, T, C
    );
    
    return out;
}
"""

attention_cpp_source = """
torch::Tensor fused_attention_cuda(
    torch::Tensor x,
    torch::Tensor w_qkv,
    torch::Tensor b_qkv,
    torch::Tensor w_out,
    torch::Tensor b_out,
    torch::Tensor mask,
    int n_head,
    float dropout_p);
"""

# Compile the inline CUDA code
fused_attention = load_inline(
    name="fused_attention",
    cpp_sources=attention_cpp_source,
    cuda_sources=attention_source,
    functions=["fused_attention_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized multi-head masked self-attention layer with custom CUDA kernels
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # Combined key, query, value projections for all heads
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        self.fused_attention = fused_attention

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Use custom fused attention kernel
        if x.is_cuda and hasattr(self, 'fused_attention'):
            # Prepare weights and biases
            w_qkv = self.c_attn.weight.t().contiguous()
            b_qkv = self.c_attn.bias
            w_out = self.c_proj.weight.t().contiguous()
            b_out = self.c_proj.bias
            
            # Call custom CUDA kernel
            y = self.fused_attention.fused_attention_cuda(
                x, w_qkv, b_qkv, w_out, b_out, self.bias, self.n_head, self.attn_dropout.p
            )
            return self.resid_dropout(y)
        else:
            # Fallback to original implementation
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

            # output projection
            y = self.resid_dropout(self.c_proj(y))
            return y

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