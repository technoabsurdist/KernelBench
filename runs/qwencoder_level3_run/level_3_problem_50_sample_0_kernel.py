import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for GELU activation
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
        output[idx] = x * cdf;
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);
    
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    
    gelu_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    
    return output;
}
"""

gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor input);"

# Custom CUDA kernel for fused attention computation
attention_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void attention_kernel(
    const float* q, 
    const float* k, 
    const float* v,
    const bool* mask,
    float* output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int total_seq_len
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int row = threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || row >= seq_len) return;
    
    const int qkv_stride = num_heads * head_dim;
    const int head_offset = batch_idx * seq_len * qkv_stride + head_idx * head_dim;
    const int out_batch_offset = batch_idx * num_heads * seq_len * head_dim;
    const int out_head_offset = out_batch_offset + head_idx * seq_len * head_dim;
    
    // Shared memory for intermediate results
    extern __shared__ float shared_mem[];
    float* att_scores = shared_mem; // seq_len * seq_len
    
    // Compute attention scores
    for (int i = 0; i < seq_len; i++) {
        float sum = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            int q_idx = head_offset + row * qkv_stride + j;
            int k_idx = head_offset + i * qkv_stride + j;
            sum += q[q_idx] * k[k_idx];
        }
        sum /= sqrtf(static_cast<float>(head_dim));
        
        // Apply mask
        int mask_idx = batch_idx * total_seq_len * total_seq_len + row * total_seq_len + i;
        if (row < seq_len && i < seq_len && !mask[mask_idx]) {
            sum = 0.0f; // Use 0 instead of -inf for ReLU
        } else {
            sum = fmaxf(0.0f, sum); // ReLU activation
        }
        
        att_scores[row * seq_len + i] = sum;
    }
    
    __syncthreads();
    
    // Compute output
    for (int i = 0; i < head_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            sum += att_scores[row * seq_len + j] * v[head_offset + j * qkv_stride + i];
        }
        
        int out_idx = out_head_offset + row * head_dim + i;
        output[out_idx] = sum;
    }
}

torch::Tensor fused_attention_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor mask
) {
    int batch_size = q.size(0);
    int num_heads = q.size(1);
    int seq_len = q.size(2);
    int head_dim = q.size(3);
    int total_seq_len = mask.size(2);
    
    auto output = torch::zeros_like(v);
    
    dim3 grid(batch_size, num_heads);
    dim3 block(seq_len);
    int shared_mem_size = seq_len * seq_len * sizeof(float);
    
    attention_kernel<<<grid, block, shared_mem_size>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        mask.data_ptr<bool>(),
        output.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        total_seq_len
    );
    
    return output;
}
"""

attention_cpp_source = "torch::Tensor fused_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor mask);"

# Compile the inline CUDA code
gelu = load_inline(
    name="gelu",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=False
)

fused_attention = load_inline(
    name="fused_attention",
    cpp_sources=attention_cpp_source,
    cuda_sources=attention_source,
    functions=["fused_attention_cuda"],
    verbose=False
)

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(NewGELU, self).__init__()
        self.gelu_func = gelu
    
    def forward(self, x):
        if x.is_cuda:
            return self.gelu_func.gelu_cuda(x)
        else:
            return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class ModelNew(nn.Module):
    """
    A multi-head masked self-attention layer with a projection at the end that uses ReLU instead of Softmax.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        self.attention_func = fused_attention

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Use custom CUDA kernel for fused attention
        if x.is_cuda:
            # Prepare mask
            mask = self.bias[:, :, :T, :T].expand(B, 1, T, T).contiguous()
            mask = mask.view(B, 1, T, T).expand(B, self.n_head, T, T).contiguous()
            mask = mask.bool()
            
            y = self.attention_func.fused_attention_cuda(q, k, v, mask)
        else:
            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.relu(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        return y