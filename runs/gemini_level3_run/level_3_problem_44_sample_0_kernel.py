import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# CUDA Kernels and C++ Wrappers
# -----------------------------------------------------------------------------

cuda_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

// -------------------
// --- NewGELU Kernel ---
// -------------------

__global__ void new_gelu_kernel(const float* __restrict__ x, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = 0.5f * val * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (val + 0.044715f * powf(val, 3.0f))));
    }
}

// -------------------
// --- LayerNorm Kernel ---
// -------------------
// A single-pass LayerNorm kernel. Each block processes one row (embedding vector).
template <typename T>
__global__ void layer_norm_kernel(const T* __restrict__ x, const T* __restrict__ gamma, const T* __restrict__ beta, T* __restrict__ out, int B, int T, int C, float epsilon) {
    int row_idx = blockIdx.x;
    if (row_idx >= B * T) return;

    extern __shared__ float sdata[];
    float* s_mean = sdata;
    float* s_var = &sdata[blockDim.x];

    const T* row_x = x + row_idx * C;
    T* row_out = out + row_idx * C;

    // Step 1: Compute sum and sum_sq in parallel using shared memory
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float val = (float)row_x[i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    s_mean[threadIdx.x] = local_sum;
    s_var[threadIdx.x] = local_sum_sq;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_mean[threadIdx.x] += s_mean[threadIdx.x + s];
            s_var[threadIdx.x] += s_var[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Final mean and variance
    if (threadIdx.x == 0) {
        float mean = s_mean[0] / C;
        float var = s_var[0] / C - mean * mean;
        s_mean[0] = mean;
        s_var[0] = rsqrtf(var + epsilon); // Store 1/sqrt(var + eps)
    }
    __syncthreads();

    float mean = s_mean[0];
    float inv_std = s_var[0];

    // Step 2: Apply normalization
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        row_out[i] = (T)(((float)row_x[i] - mean) * inv_std * (float)gamma[i] + (float)beta[i]);
    }
}


// -------------------
// --- Fused Attention Softmax Kernel ---
// -------------------
// Fuses scaling, causal masking, and softmax.
// Each block processes one row of the attention matrix (B * n_head * T rows in total).
template <typename T>
__global__ void fused_scale_mask_softmax_kernel(T* __restrict__ att, int B, int n_head, int T_seq, float scale) {
    int row_glob_idx = blockIdx.x;
    if (row_glob_idx >= B * n_head * T_seq) return;

    extern __shared__ float sdata[];

    int t_query = row_glob_idx % T_seq; // which query token this row corresponds to

    T* row_att = att + row_glob_idx * T_seq;

    // Step 1: Find max value for stable softmax
    float max_val = -1e20f;
    for (int i = threadIdx.x; i < T_seq; i += blockDim.x) {
        if (i <= t_query) { // Apply causal mask
            float val = (float)row_att[i] * scale;
            if (val > max_val) {
                max_val = val;
            }
        }
    }
    sdata[threadIdx.x] = max_val;
    __syncthreads();

    // Reduction for max_val
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // Step 2: Compute exp and sum
    float sum_val = 0.0f;
    for (int i = threadIdx.x; i < T_seq; i += blockDim.x) {
        if (i <= t_query) {
            float val = expf((float)row_att[i] * scale - max_val);
            sum_val += val;
            row_att[i] = (T)val; // Store intermediate exp value in place
        } else {
            row_att[i] = (T)0.0f; // Store 0 for masked out values
        }
    }
    sdata[threadIdx.x] = sum_val;
    __syncthreads();

    // Reduction for sum_val
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum_val = sdata[0];
    __syncthreads();

    // Step 3: Normalize
    float inv_sum = 1.0f / (sum_val + 1e-8f);
    for (int i = threadIdx.x; i < T_seq; i += blockDim.x) {
        row_att[i] = (T)((float)row_att[i] * inv_sum);
    }
}


// -------------------
// --- C++ Wrappers ---
// -------------------

torch::Tensor new_gelu_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");

    auto out = torch::empty_like(x);
    auto size = x.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    new_gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

torch::Tensor layer_norm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float epsilon) {
    TORCH_CHECK(x.is_cuda() && gamma.is_cuda() && beta.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && gamma.is_contiguous() && beta.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(x.dim() == 3, "Input x must be a 3D tensor (B, T, C)");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input x must be a float32 tensor");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32, "Input gamma must be a float32 tensor");
    TORCH_CHECK(beta.scalar_type() == torch::kFloat32, "Input beta must be a float32 tensor");

    const auto dims = x.sizes();
    const int B = dims[0];
    const int T = dims[1];
    const int C = dims[2];

    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = B * T;
    const int shared_mem_size = 2 * block_size * sizeof(float);

    layer_norm_kernel<float><<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        B, T, C, epsilon
    );
    return out;
}

torch::Tensor fused_scale_mask_softmax_cuda(torch::Tensor att, float scale) {
    TORCH_CHECK(att.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(att.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(att.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(att.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");

    const auto dims = att.sizes();
    const int B = dims[0];
    const int n_head = dims[1];
    const int T_seq = dims[2];
    TORCH_CHECK(T_seq == dims[3], "Input must be a square matrix in the last two dimensions");

    // The kernel will operate in-place on the input tensor
    auto out = att;

    const int num_rows = B * n_head * T_seq;
    const int block_size = 256;
    const int num_blocks = num_rows;
    const int shared_mem_size = block_size * sizeof(float);

    fused_scale_mask_softmax_kernel<float><<<num_blocks, block_size, shared_mem_size>>>(
        out.data_ptr<float>(), B, n_head, T_seq, scale
    );
    return out;
}
"""

cpp_source = """
#include <torch/extension.h>

torch::Tensor new_gelu_cuda(torch::Tensor x);
torch::Tensor layer_norm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float epsilon);
torch::Tensor fused_scale_mask_softmax_cuda(torch::Tensor att, float scale);
"""

transformer_kernels = load_inline(
    name="transformer_kernels",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["new_gelu_cuda", "layer_norm_cuda", "fused_scale_mask_softmax_cuda"],
    verbose=True,
)

# -----------------------------------------------------------------------------
# Custom nn.Modules using the CUDA kernels
# -----------------------------------------------------------------------------

class CustomNewGELU(nn.Module):
    """
    Custom GELU activation using our CUDA kernel.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return transformer_kernels.new_gelu_cuda(x)

class CustomLayerNorm(nn.Module):
    """
    Custom LayerNorm using our CUDA kernel.
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x):
        return transformer_kernels.layer_norm_cuda(x, self.weight, self.bias, self.eps)

class CausalSelfAttentionNew(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    This version uses a custom fused CUDA kernel for scale, mask, and softmax.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # Use the custom fused kernel
        att = q @ k.transpose(-2, -1)
        scale = 1.0 / math.sqrt(k.size(-1))
        att = transformer_kernels.fused_scale_mask_softmax_cuda(att, scale)
        
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class ModelNew(nn.Module):
    """ an unassuming Transformer block with custom CUDA kernels """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = CustomLayerNorm(n_embd)
        # Note: max_seqlen is no longer needed by CausalSelfAttentionNew
        self.attn = CausalSelfAttentionNew(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.ln_2 = CustomLayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = CustomNewGELU(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x