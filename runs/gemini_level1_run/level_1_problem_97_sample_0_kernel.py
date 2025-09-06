import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for a fused scaled dot-product attention.
# This implementation fuses all operations (MatMul, Scale, Softmax, MatMul) into a single kernel.
# It processes the attention row-by-row to avoid materializing the large (N, N) attention matrix
# in global memory, which is the primary bottleneck for large sequence lengths.
# While not as optimized as state-of-the-art implementations like FlashAttention (which uses tiling),
# this approach demonstrates the core concept of operator fusion for memory-bound operations.
fused_attention_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <limits>

// Helper for reduction operations inside a thread block
template <typename T>
__device__ T block_reduce_max(T val, T* shared_mem) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    shared_mem[tid] = val;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] = max(shared_mem[tid], shared_mem[tid + s]);
        }
        __syncthreads();
    }
    return shared_mem[0];
}

template <typename T>
__device__ T block_reduce_sum(T val, T* shared_mem) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    shared_mem[tid] = val;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    return shared_mem[0];
}


__global__ void fused_attention_row_kernel(
    const __half* q_ptr,
    const __half* k_ptr,
    const __half* v_ptr,
    __half* out_ptr,
    int B, int H, int N, int D,
    float scale) {

    // --- Block and Thread Indexing ---
    // Grid is (B * H, N), so each block computes one row of the output.
    int bh_idx = blockIdx.x;
    int row_idx = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // --- Memory Pointers ---
    const __half* q_row_ptr = q_ptr + (bh_idx * N + row_idx) * D;
    const __half* k_base_ptr = k_ptr + bh_idx * N * D;
    const __half* v_base_ptr = v_ptr + bh_idx * N * D;
    __half* out_row_ptr = out_ptr + (bh_idx * N + row_idx) * D;

    // --- Local Storage ---
    // Shared memory to store the attention scores (S) and later probabilities (P) for this row.
    // Also used for block-wide reductions.
    extern __shared__ float shared_mem[];
    float* s_scores = shared_mem; // size N
    float* reduce_mem = shared_mem + N; // size block_size for reductions

    // --- Step 1: Compute S[row_idx, :] = (Q[row_idx, :] @ K^T) * scale ---
    // Each thread computes a subset of the N dot products.
    for (int j = tid; j < N; j += block_size) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += __half2float(q_row_ptr[d]) * __half2float(k_base_ptr[j * D + d]);
        }
        s_scores[j] = dot * scale;
    }
    __syncthreads();

    // --- Step 2: Stable Softmax on s_scores (in shared memory) ---
    // 2a. Find max value for numerical stability
    float thread_max = -std::numeric_limits<float>::infinity();
    for (int j = tid; j < N; j += block_size) {
        thread_max = max(thread_max, s_scores[j]);
    }
    float max_val = block_reduce_max(thread_max, reduce_mem);
    __syncthreads();

    // 2b. Compute exp and sum
    float thread_sum = 0.0f;
    for (int j = tid; j < N; j += block_size) {
        s_scores[j] = expf(s_scores[j] - max_val);
        thread_sum += s_scores[j];
    }
    float sum_exp = block_reduce_sum(thread_sum, reduce_mem);
    __syncthreads();

    // 2c. Normalize to get probabilities (P)
    float inv_sum_exp = 1.0f / (sum_exp + 1e-6); // Add epsilon for safety
    for (int j = tid; j < N; j += block_size) {
        s_scores[j] *= inv_sum_exp;
    }
    __syncthreads();
    // Now s_scores contains the probability vector P[row_idx, :]

    // --- Step 3: Compute O[row_idx, :] = P[row_idx, :] @ V ---
    // Each thread computes a part of the output vector (D elements).
    for (int d = tid; d < D; d += block_size) {
        float o_val = 0.0f;
        for (int j = 0; j < N; ++j) {
            o_val += s_scores[j] * __half2float(v_base_ptr[j * D + d]);
        }
        out_row_ptr[d] = __float2half(o_val);
    }
}

torch::Tensor fused_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v) {

    // Input validation
    TORCH_CHECK(q.is_cuda(), "Input tensors must be on CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16, "Input tensors must be float16");
    TORCH_CHECK(q.is_contiguous(), "Input Q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "Input K must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "Input V must be contiguous");
    TORCH_CHECK(q.dim() == 4, "Input Q must be 4D");
    TORCH_CHECK(k.dim() == 4, "Input K must be 4D");
    TORCH_CHECK(v.dim() == 4, "Input V must be 4D");

    // Get dimensions
    const auto B = q.size(0);
    const auto H = q.size(1);
    const auto N = q.size(2);
    const auto D = q.size(3);

    // Create output tensor
    auto out = torch::empty_like(q);

    // Kernel configuration
    const int THREADS = 256;
    dim3 grid(B * H, N);
    dim3 block(THREADS);

    // Shared memory size: N floats for the score vector + THREADS floats for reduction
    size_t shared_mem_size = (N + THREADS) * sizeof(float);

    float scale = 1.0f / sqrtf(static_cast<float>(D));

    // Launch kernel
    fused_attention_row_kernel<<<grid, block, shared_mem_size>>>(
        (const __half*)q.data_ptr(),
        (const __half*)k.data_ptr(),
        (const __half*)v.data_ptr(),
        (__half*)out.data_ptr(),
        B, H, N, D,
        scale
    );
    
    // Check for any CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
"""

fused_attention_cpp_source = (
    "torch::Tensor fused_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);"
)

# Compile the inline CUDA code
fused_attention = load_inline(
    name="fused_attention",
    cpp_sources=fused_attention_cpp_source,
    cuda_sources=fused_attention_source,
    functions=["fused_attention_forward"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        # Store the custom operator
        self.fused_attention = fused_attention.fused_attention_forward

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Call the custom fused CUDA kernel
        return self.fused_attention(Q, K, V)

batch_size = 32
num_heads = 32
sequence_length = 512
embedding_dimension = 1024

def get_inputs():
    Q = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension, device='cuda', dtype=torch.float16)
    K = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension, device='cuda', dtype=torch.float16)
    V = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension, device='cuda', dtype=torch.float16)
    return [Q, K, V]

def get_init_inputs():
    return []