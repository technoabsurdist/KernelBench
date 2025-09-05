import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for LayerNorm
layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int batch_size,
    int hidden_size,
    float eps) {
    
    int idx = blockIdx.x;
    if (idx >= batch_size) return;
    
    const float* input_row = input + idx * hidden_size;
    float* output_row = output + idx * hidden_size;
    
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        sum += input_row[i];
    }
    
    __shared__ float shared_sum[32];
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) {
        mean[idx] = sum / hidden_size;
    }
    __syncthreads();
    
    float m = mean[idx];
    float variance = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = input_row[i] - m;
        variance += diff * diff;
    }
    
    variance = blockReduceSum(variance);
    if (threadIdx.x == 0) {
        rstd[idx] = rsqrtf(variance / hidden_size + eps);
    }
    __syncthreads();
    
    float s = rstd[idx];
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (input_row[i] - m) * s;
        output_row[i] = normalized * gamma[i] + beta[i];
    }
}

__device__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

torch::Tensor layernorm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps) {
    auto batch_size = input.size(0) * input.size(1);
    auto hidden_size = input.size(2);
    auto output = torch::zeros_like(input);
    auto mean = torch::zeros({batch_size}, input.options());
    auto rstd = torch::zeros({batch_size}, input.options());
    
    input = input.contiguous().view({batch_size, hidden_size});
    output = output.view({batch_size, hidden_size});
    
    const int threads = 256;
    const int blocks = batch_size;
    
    layernorm_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        batch_size,
        hidden_size,
        eps
    );
    
    return output.view({input.size(0), input.size(1), hidden_size});
}
"""

layernorm_cpp_source = "torch::Tensor layernorm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps);"

# Custom CUDA kernel for GELU
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void gelu_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float a = 0.044715f;
        float x_cubed = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + a * x_cubed);
        float tanh_result = tanhf(tanh_arg);
        output[idx] = 0.5f * x * (1.0f + tanh_result);
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    gelu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor input);"

# Custom CUDA kernel for fused attention
attention_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

__global__ void fused_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ mask,
    float* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    float dropout_prob) {
    
    extern __shared__ float shared_mem[];
    
    int batch_head = blockIdx.y;
    int row = blockIdx.x;
    int batch = batch_head / num_heads;
    int head = batch_head % num_heads;
    
    if (batch >= batch_size || row >= seq_len) return;
    
    const float* q_ptr = q + batch * num_heads * seq_len * head_dim + head * seq_len * head_dim + row * head_dim;
    const float* k_ptr = k + batch * num_heads * seq_len * head_dim + head * seq_len * head_dim;
    const float* v_ptr = v + batch * num_heads * seq_len * head_dim + head * seq_len * head_dim;
    float* out_ptr = output + batch * num_heads * seq_len * head_dim + head * seq_len * head_dim + row * head_dim;
    
    float max_score = -1e20f;
    for (int col = threadIdx.x; col <= row && col < seq_len; col += blockDim.x) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_ptr[d] * k_ptr[col * head_dim + d];
        }
        score *= scale;
        
        if (mask[row * seq_len + col] == 0) {
            score = -1e20f;
        }
        
        shared_mem[col] = score;
        max_score = fmaxf(max_score, score);
    }
    __syncthreads();
    
    float sum_exp = 0.0f;
    for (int col = threadIdx.x; col <= row && col < seq_len; col += blockDim.x) {
        if (mask[row * seq_len + col] != 0) {
            float exp_score = expf(shared_mem[col] - max_score);
            shared_mem[col] = exp_score;
            sum_exp += exp_score;
        }
    }
    __syncthreads();
    
    for (int col = threadIdx.x; col <= row && col < seq_len; col += blockDim.x) {
        if (mask[row * seq_len + col] != 0) {
            shared_mem[col] /= sum_exp;
        }
    }
    __syncthreads();
    
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int col = 0; col <= row && col < seq_len; col++) {
            if (mask[row * seq_len + col] != 0) {
                acc += shared_mem[col] * v_ptr[col * head_dim + d];
            }
        }
        out_ptr[d] = acc;
    }
}

torch::Tensor fused_attention_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, 
    torch::Tensor mask, float scale, float dropout_prob) {
    
    auto batch_size = q.size(0);
    auto num_heads = q.size(1);
    auto seq_len = q.size(2);
    auto head_dim = q.size(3);
    
    auto output = torch::zeros_like(q);
    
    dim3 blocks(seq_len, batch_size * num_heads);
    dim3 threads(min(256, (int)seq_len));
    size_t shared_mem_size = seq_len * sizeof(float);
    
    fused_attention_kernel<<<blocks, threads, shared_mem_size>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        mask.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        scale,
        dropout_prob
    );
    
    return output;
}
"""

attention_cpp_source = "torch::Tensor fused_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor mask, float scale, float dropout_prob);"

# Load custom kernels
layernorm_kernel = load_inline(
    name="layernorm_kernel",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_source,
    functions=["layernorm_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

gelu_kernel = load_inline(
    name="gelu_kernel",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

attention_kernel = load_inline(
    name="attention_kernel",
    cpp_sources=attention_cpp_source,
    cuda_sources=attention_source,
    functions=["fused_attention_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        
    def forward(self, x):
        return layernorm_kernel.layernorm_cuda(x, self.weight, self.bias, self.eps)

class CustomGELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return gelu_kernel.gelu_cuda(x)

class CausalSelfAttentionNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        self.scale = 1.0 / math.sqrt(n_embd // n_head)
        self.attn_pdrop = attn_pdrop

    def forward(self, x):
        B, T, C = x.size()
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        mask = self.bias[:,:,:T,:T].squeeze(0).squeeze(0)
        
        if self.training and self.attn_pdrop > 0:
            att = (q @ k.transpose(-2, -1)) * self.scale
            att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        else:
            y = attention_kernel.fused_attention_cuda(q, k, v, mask, self.scale, self.attn_pdrop)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = CustomLayerNorm(n_embd)
        self.attn = CausalSelfAttentionNew(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = CustomLayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = CustomGELU(),
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
    return [torch.rand(batch_size, seq_len, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]