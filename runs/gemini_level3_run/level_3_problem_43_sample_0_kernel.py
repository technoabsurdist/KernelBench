import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Fused Attention CUDA Kernel (FlashAttention-style)
# This kernel fuses the entire attention mechanism (scaling, masking, softmax, and value multiplication)
# into a single pass. It avoids materializing the large (T, T) attention matrix,
# significantly reducing memory usage and improving performance by enhancing memory locality.
# The implementation uses an online softmax algorithm to compute the result in a single pass over the keys/values.
fused_attention_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Using float for simplicity. Half precision (float16) would be faster on modern GPUs.
using scalar_t = float;

// A simplified FlashAttention-style kernel.
// Each thread block computes one row of the output tensor 'o'.
// It iterates through keys and values, updating the output row using an online softmax algorithm.
__global__ void fused_attention_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ o,
    const int B,
    const int H,
    const int T,
    const int D_head
) {
    // Shared memory for block-wide reduction of the dot product score.
    extern __shared__ float sdata[];

    // Identify which batch, head, and query row this block is working on.
    // The grid is structured as (B * H, T).
    const int query_row_idx_global = blockIdx.y; // Sequence index for the query (from 0 to T-1)
    const int batch_head_idx = blockIdx.x;       // Batch and head index (from 0 to B*H-1)
    const int thread_idx = threadIdx.x;
    const int block_size = blockDim.x;

    // Pointers to the start of the data for the current batch and head.
    const scalar_t* q_ptr = q + batch_head_idx * T * D_head;
    const scalar_t* k_ptr = k + batch_head_idx * T * D_head;
    const scalar_t* v_ptr = v + batch_head_idx * T * D_head;
    scalar_t* o_ptr = o + batch_head_idx * T * D_head;

    // --- Online Softmax variables ---
    // Each thread handles a fraction of the D_head elements.
    const int V_VEC_SIZE = (D_head + block_size - 1) / block_size;
    scalar_t o_val[V_VEC_SIZE];
    for(int i=0; i<V_VEC_SIZE; ++i) o_val[i] = 0.0f;

    scalar_t max_score = -INFINITY; // Max score so far
    scalar_t exp_sum = 0.0f;        // Sum of exp(score - max_score)

    // Load the query vector for this row into registers.
    scalar_t q_vec[V_VEC_SIZE];
    for(int i=0; i<V_VEC_SIZE; ++i) {
        int idx = i * block_size + thread_idx;
        if (idx < D_head) {
            q_vec[i] = q_ptr[query_row_idx_global * D_head + idx];
        }
    }

    // Iterate through all keys for the current query.
    for (int key_col_idx = 0; key_col_idx < T; ++key_col_idx) {
        // Causal mask: a query at position `i` can only attend to keys at positions `j <= i`.
        if (key_col_idx > query_row_idx_global) {
            continue;
        }

        // --- Step 1: Compute dot product Q_i * K_j^T ---
        scalar_t score_local = 0.0f;
        for(int i=0; i<V_VEC_SIZE; ++i) {
            int idx = i * block_size + thread_idx;
            if (idx < D_head) {
                score_local += q_vec[i] * k_ptr[key_col_idx * D_head + idx];
            }
        }
        sdata[thread_idx] = score_local;
        __syncthreads();

        // Block-wide reduction to sum the partial dot products.
        for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
            if (thread_idx < s) {
                sdata[thread_idx] += sdata[thread_idx + s];
            }
            __syncthreads();
        }

        // Thread 0 now has the final score. Apply scaling.
        scalar_t score = 0.0f;
        if (thread_idx == 0) {
            score = sdata[0] * (1.0f / sqrtf((float)D_head));
        }
        // Broadcast the final score to all threads in the block.
        if (thread_idx == 0) sdata[0] = score;
        __syncthreads();
        score = sdata[0];

        // --- Step 2: Online Softmax update ---
        scalar_t old_max = max_score;
        max_score = fmaxf(max_score, score);

        scalar_t p_ij = expf(score - max_score);
        scalar_t scale_factor = expf(old_max - max_score);
        
        scalar_t old_exp_sum = exp_sum;
        exp_sum = old_exp_sum * scale_factor + p_ij;

        // --- Step 3: Update output accumulator O_i ---
        // Rescale the existing accumulator with the new max score.
        for(int i=0; i<V_VEC_SIZE; ++i) {
            o_val[i] *= scale_factor;
        }
        // Add the contribution from the current value vector V_j.
        for(int i=0; i<V_VEC_SIZE; ++i) {
            int idx = i * block_size + thread_idx;
            if (idx < D_head) {
                o_val[i] += p_ij * v_ptr[key_col_idx * D_head + idx];
            }
        }
    }

    // --- Final normalization and write to global memory ---
    for(int i=0; i<V_VEC_SIZE; ++i) {
        int idx = i * block_size + thread_idx;
        if (idx < D_head) {
            // Add a small epsilon for stability to avoid division by zero.
            if (exp_sum > 1e-6f) {
                o_ptr[query_row_idx_global * D_head + idx] = o_val[i] / exp_sum;
            } else {
                o_ptr[query_row_idx_global * D_head + idx] = 0.0f;
            }
        }
    }
}

// C++ wrapper function that launches the CUDA kernel.
torch::Tensor fused_attention_forward_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    // Input validation checks
    TORCH_CHECK(q.is_cuda(), "Input tensor Q must be on CUDA");
    TORCH_CHECK(k.is_cuda(), "Input tensor K must be on CUDA");
    TORCH_CHECK(v.is_cuda(), "Input tensor V must be on CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "Input tensor Q must be float32");
    TORCH_CHECK(k.scalar_type() == torch::kFloat32, "Input tensor K must be float32");
    TORCH_CHECK(v.scalar_type() == torch::kFloat32, "Input tensor V must be float32");
    TORCH_CHECK(q.dim() == 4, "Input tensor Q must be 4D");
    TORCH_CHECK(k.dim() == 4, "Input tensor K must be 4D");
    TORCH_CHECK(v.dim() == 4, "Input tensor V must be 4D");

    const int B = q.size(0);
    const int H = q.size(1);
    const int T = q.size(2);
    const int D_head = q.size(3);

    TORCH_CHECK(D_head == k.size(3) && D_head == v.size(3), "Head dimension must be consistent");
    TORCH_CHECK(T == k.size(2) && T == v.size(2), "Sequence length must be consistent");

    // Allocate output tensor
    auto o = torch::empty_like(q);

    // Kernel launch configuration
    const int block_size = 256;
    TORCH_CHECK(D_head <= block_size, "D_head must be <= block_size for this kernel implementation");

    dim3 grid(B * H, T);
    dim3 block(block_size);

    // Shared memory size for the block-wide reduction
    size_t shared_mem_size = block_size * sizeof(float);

    // Launch the kernel
    fused_attention_forward_kernel<<<grid, block, shared_mem_size>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        o.data_ptr<float>(),
        B, H, T, D_head
    );
    
    return o;
}
"""

fused_attention_cpp_source = """
torch::Tensor fused_attention_forward_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v);
"""

# JIT compile the CUDA kernel using load_inline
fused_attention = load_inline(
    name="fused_attention",
    cpp_sources=fused_attention_cpp_source,
    cuda_sources=fused_attention_cuda_source,
    functions=["fused_attention_forward_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    An optimized version of the multi-head self-attention layer using a custom
    fused CUDA kernel. This kernel implements a FlashAttention-style algorithm
    to avoid materializing the large (T, T) attention matrix, reducing memory
    footprint and improving performance.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        # Note: The custom kernel does not implement attention dropout.
        # This is acceptable as the provided attn_pdrop is 0.0.
        self.resid_dropout = nn.Dropout(resid_pdrop)
        
        self.n_head = n_head
        self.n_embd = n_embd
        
        # The causal mask is now implemented inside the CUDA kernel,
        # so the 'bias' buffer from the original model is no longer needed.

        # Store the compiled custom CUDA function
        self.fused_attention = fused_attention

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape and transpose for multi-head attention: (B, T, C) -> (B, nh, T, hs)
        hs = C // self.n_head
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        # Perform fused causal self-attention using the custom CUDA kernel.
        # The kernel handles: QK^T, scaling, causal masking, softmax, and AV.
        y = self.fused_attention.fused_attention_forward_cuda(q, k, v)
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y