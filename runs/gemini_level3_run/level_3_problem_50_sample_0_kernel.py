import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# From https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(NewGELU, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

# Define the custom CUDA kernel for fused attention
# This kernel fuses the Q@K.T, scaling, causal masking, ReLU, and the final @V operation.
# This avoids materializing the large (B, H, T, T) attention matrix, which is the primary
# source of memory bandwidth bottleneck in standard attention implementations.
fused_attention_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

// CUDA Kernel for Fused Causal Attention with ReLU
// Computes: y = (ReLU(Mask(Scale(Q @ K.T)))) @ V
__global__ void fused_attention_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    float scale,
    int T,
    int C_per_head) {

    // This kernel is specialized for C_per_head = 64, which matches the problem spec.
    if (C_per_head != 64) return;

    // Grid and Block dimensions
    // Grid: (T, B*H), Block: (64)
    // Each block computes one row of the output y.
    // Each thread in the block computes one element of that row.
    int t_row = blockIdx.x;
    int batch_head_idx = blockIdx.y;
    int head_col = threadIdx.x;

    // Pointers to the start of the data for the current batch and head
    const float* q_ptr_bh = q + batch_head_idx * T * C_per_head;
    const float* k_ptr_bh = k + batch_head_idx * T * C_per_head;
    const float* v_ptr_bh = v + batch_head_idx * T * C_per_head;
    float* out_ptr_bh = out + batch_head_idx * T * C_per_head;

    // Shared memory for efficient dot product calculation
    __shared__ float q_row_s[64];
    __shared__ float k_row_s[64];
    __shared__ float partial_dots[64];

    // Accumulator for the output element this thread is responsible for
    float out_val = 0.0f;

    // Load the q vector for the current row (t_row) into shared memory.
    // All threads in the block work on the same t_row.
    q_row_s[head_col] = q_ptr_bh[t_row * C_per_head + head_col];
    __syncthreads();

    // Loop over j (columns of the implicit attention matrix / rows of k and v)
    for (int j = 0; j < T; ++j) {
        // Apply causal mask: only attend to positions j <= t_row
        if (j > t_row) {
            continue; // The attention score would be 0 after ReLU, so we can skip.
        }

        // Load the k vector for row j into shared memory
        k_row_s[head_col] = k_ptr_bh[j * C_per_head + head_col];
        __syncthreads();

        // --- Parallel Reduction for Dot Product (s_ij = q_row . k_row) ---
        partial_dots[head_col] = q_row_s[head_col] * k_row_s[head_col];
        __syncthreads();

        // In-place reduction in shared memory.
        if (head_col < 32) partial_dots[head_col] += partial_dots[head_col + 32]; __syncthreads();
        if (head_col < 16) partial_dots[head_col] += partial_dots[head_col + 16]; __syncthreads();
        if (head_col < 8)  partial_dots[head_col] += partial_dots[head_col + 8];  __syncthreads();
        if (head_col < 4)  partial_dots[head_col] += partial_dots[head_col + 4];  __syncthreads();
        if (head_col < 2)  partial_dots[head_col] += partial_dots[head_col + 2];  __syncthreads();
        if (head_col < 1)  partial_dots[head_col] += partial_dots[head_col + 1];  __syncthreads();

        // The final dot product is in partial_dots[0], accessible to all threads in the block.
        float s_ij = partial_dots[0];
        __syncthreads(); // Ensure all threads read s_ij after reduction is complete.

        // Apply scale and ReLU to get the attention score p_ij
        float p_ij = fmaxf(0.0f, s_ij * scale);

        // Accumulate output: out_val += p_ij * v_j[head_col]
        out_val += p_ij * v_ptr_bh[j * C_per_head + head_col];
    }

    // Write the final accumulated value to global memory
    out_ptr_bh[t_row * C_per_head + head_col] = out_val;
}

// C++ Wrapper function that launches the CUDA kernel
torch::Tensor fused_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, float scale) {
    // Input validation
    TORCH_CHECK(q.dim() == 4, "q must be a 4D tensor");
    TORCH_CHECK(k.dim() == 4, "k must be a 4D tensor");
    TORCH_CHECK(v.dim() == 4, "v must be a 4D tensor");
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");

    const auto B = q.size(0);
    const auto H = q.size(1);
    const auto T = q.size(2);
    const auto C_per_head = q.size(3);

    TORCH_CHECK(C_per_head == 64, "This kernel is specialized for head size 64");

    // Prepare output tensor
    auto out = torch::empty_like(q);

    // Reshape tensors for kernel launch: (B, H, T, C_ph) -> (B*H, T, C_ph)
    auto q_reshaped = q.reshape({B * H, T, C_per_head});
    auto k_reshaped = k.reshape({B * H, T, C_per_head});
    auto v_reshaped = v.reshape({B * H, T, C_per_head});
    auto out_reshaped = out.reshape({B * H, T, C_per_head});

    // Kernel launch configuration
    dim3 threads(C_per_head);
    dim3 blocks(T, B * H);

    fused_attention_kernel<<<blocks, threads>>>(
        q_reshaped.data_ptr<float>(),
        k_reshaped.data_ptr<float>(),
        v_reshaped.data_ptr<float>(),
        out_reshaped.data_ptr<float>(),
        scale,
        T,
        C_per_head
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

fused_attention_cpp_source = (
    "torch::Tensor fused_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, float scale);"
)

# Compile the inline CUDA code
fused_attention = load_inline(
    name="fused_attention",
    cpp_sources=fused_attention_cpp_source,
    cuda_sources=fused_attention_source,
    functions=["fused_attention_forward"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    An optimized version of the multi-head masked self-attention layer.
    The core attention computation is replaced by a single, fused CUDA kernel.
    """

    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # The causal mask is now handled inside the fused CUDA kernel, so the 'bias' buffer is no longer needed.
        self.n_head = n_head
        self.n_embd = n_embd
        # Store the compiled custom operator
        self.fused_attention = fused_attention

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Ensure tensors are contiguous for the CUDA kernel
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Calculate the scaling factor
        scale = 1.0 / math.sqrt(k.size(-1))
        
        # Call the custom fused CUDA kernel for attention
        y = self.fused_attention.fused_attention_forward(q, k, v, scale)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Note: The original model does not use the final projection, so we return y directly.
        # A complete transformer block would typically apply self.c_proj(y) here.
        return y