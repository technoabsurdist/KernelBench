import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for scaled dot-product attention
scaled_dot_product_attention_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

__global__ void softmax_kernel(float* attn_scores, const int seq_len, const float scale) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int batch_head_idx = row;

    // Shared memory for reduction
    extern __shared__ float shared_data[];
    float* shared_max = shared_data;
    float* shared_sum = shared_data + WARP_SIZE;

    // Initialize shared memory
    if (tid < WARP_SIZE) {
        shared_max[tid] = -INFINITY;
        shared_sum[tid] = 0.0f;
    }
    __syncthreads();

    // Load data and find max
    float thread_max = -INFINITY;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        attn_scores[batch_head_idx * seq_len * seq_len + blockIdx.y * seq_len + i] *= scale;
        thread_max = fmaxf(thread_max, attn_scores[batch_head_idx * seq_len * seq_len + blockIdx.y * seq_len + i]);
    }

    // Warp-level reduction for max
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xFFFFFFFF, thread_max, offset));
    }

    // Write to shared memory
    if (tid % WARP_SIZE == 0) {
        shared_max[tid / WARP_SIZE] = thread_max;
    }
    __syncthreads();

    // Reduce shared memory
    if (tid < WARP_SIZE) {
        thread_max = shared_max[tid];
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            thread_max = fmaxf(thread_max, shared_max[tid + offset]);
        }
        shared_max[0] = thread_max;
    }
    __syncthreads();
    float max_val = shared_max[0];

    // Compute exp and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float val = expf(attn_scores[batch_head_idx * seq_len * seq_len + blockIdx.y * seq_len + i] - max_val);
        attn_scores[batch_head_idx * seq_len * seq_len + blockIdx.y * seq_len + i] = val;
        thread_sum += val;
    }

    // Warp-level reduction for sum
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
    }

    // Write to shared memory
    if (tid % WARP_SIZE == 0) {
        shared_sum[tid / WARP_SIZE] = thread_sum;
    }
    __syncthreads();

    // Reduce shared memory
    if (tid < WARP_SIZE) {
        thread_sum = shared_sum[tid];
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            thread_sum += shared_sum[tid + offset];
        }
        shared_sum[0] = thread_sum;
    }
    __syncthreads();
    float sum_val = shared_sum[0];

    // Normalize
    for (int i = tid; i < seq_len; i += blockDim.x) {
        attn_scores[batch_head_idx * seq_len * seq_len + blockIdx.y * seq_len + i] /= sum_val;
    }
}

torch::Tensor scaled_dot_product_attention_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int embed_dim = Q.size(3);
    
    // Create output tensor
    auto options = torch::TensorOptions().dtype(Q.dtype()).device(Q.device());
    auto attn_output = torch::zeros({batch_size, num_heads, seq_len, embed_dim}, options);
    auto attn_scores = torch::zeros({batch_size, num_heads, seq_len, seq_len}, options);
    
    // Get raw pointers
    float* q_ptr = Q.data_ptr<float>();
    float* k_ptr = K.data_ptr<float>();
    float* v_ptr = V.data_ptr<float>();
    float* scores_ptr = attn_scores.data_ptr<float>();
    float* out_ptr = attn_output.data_ptr<float>();
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Constants for cuBLAS
    const float alpha = 1.0f / sqrtf(static_cast<float>(embed_dim));
    const float beta = 0.0f;
    
    // Compute Q @ K^T
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            cublasSgemm(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                seq_len, seq_len, embed_dim,
                &alpha,
                k_ptr + (b * num_heads + h) * seq_len * embed_dim, embed_dim,
                q_ptr + (b * num_heads + h) * seq_len * embed_dim, embed_dim,
                &beta,
                scores_ptr + (b * num_heads + h) * seq_len * seq_len, seq_len
            );
        }
    }
    
    // Apply softmax
    const int threads_per_block = min(MAX_THREADS_PER_BLOCK, seq_len);
    const int shared_mem_size = 2 * WARP_SIZE * sizeof(float);
    softmax_kernel<<<batch_size * num_heads * seq_len, threads_per_block, shared_mem_size>>>(
        scores_ptr, seq_len, 1.0f
    );
    
    // Compute softmax(QK^T/sqrt(d)) @ V
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            const float beta_zero = 0.0f;
            cublasSgemm(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                embed_dim, seq_len, seq_len,
                &alpha,
                v_ptr + (b * num_heads + h) * seq_len * embed_dim, embed_dim,
                scores_ptr + (b * num_heads + h) * seq_len * seq_len, seq_len,
                &beta_zero,
                out_ptr + (b * num_heads + h) * seq_len * embed_dim, embed_dim
            );
        }
    }
    
    // Clean up cuBLAS handle
    cublasDestroy(handle);
    
    return attn_output;
}
"""

scaled_dot_product_attention_cpp_source = (
    "torch::Tensor scaled_dot_product_attention_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V);"
)

# Compile the inline CUDA code for scaled dot-product attention
scaled_dot_product_attention = load_inline(
    name="scaled_dot_product_attention",
    cpp_sources=scaled_dot_product_attention_cpp_source,
    cuda_sources=scaled_dot_product_attention_source,
    functions=["scaled_dot_product_attention_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.scaled_dot_product_attention = scaled_dot_product_attention

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Convert to float32 for the custom kernel
        Q_f32 = Q.float()
        K_f32 = K.float()
        V_f32 = V.float()
        
        # Apply custom scaled dot-product attention
        out = self.scaled_dot_product_attention.scaled_dot_product_attention_cuda(Q_f32, K_f32, V_f32)
        
        # Convert back to float16
        return out.half()