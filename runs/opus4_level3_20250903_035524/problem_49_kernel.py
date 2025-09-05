import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for segsum + exp fusion
segsum_exp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void segsum_exp_kernel(
    const float* x, 
    float* out, 
    int batch_size, 
    int heads, 
    int chunks, 
    int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * heads * chunks * seq_len * seq_len;
    
    if (idx < total_elements) {
        int s2 = idx % seq_len;
        int s1 = (idx / seq_len) % seq_len;
        int c = (idx / (seq_len * seq_len)) % chunks;
        int h = (idx / (seq_len * seq_len * chunks)) % heads;
        int b = idx / (seq_len * seq_len * chunks * heads);
        
        if (s1 >= s2) {
            float sum = 0.0f;
            for (int k = s2; k <= s1; k++) {
                sum += x[b * heads * chunks * seq_len + h * chunks * seq_len + c * seq_len + k];
            }
            out[idx] = expf(sum);
        } else {
            out[idx] = 0.0f;
        }
    }
}

torch::Tensor segsum_exp_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto heads = x.size(1);
    auto chunks = x.size(2);
    auto seq_len = x.size(3);
    
    auto out = torch::zeros({batch_size, heads, chunks, seq_len, seq_len}, 
                           x.options());
    
    int total_elements = batch_size * heads * chunks * seq_len * seq_len;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    segsum_exp_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size, heads, chunks, seq_len
    );
    
    return out;
}
"""

segsum_exp_cpp_source = "torch::Tensor segsum_exp_cuda(torch::Tensor x);"

# Custom CUDA kernel for einsum operations
einsum_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void einsum_diagonal_kernel(
    const float* C, const float* B, const float* L, const float* X,
    float* Y, 
    int batch_size, int chunks, int seq_len, int heads, int d_head, int d_state
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * chunks * seq_len * heads * d_head;
    
    if (idx < total) {
        int p = idx % d_head;
        int h = (idx / d_head) % heads;
        int l = (idx / (d_head * heads)) % seq_len;
        int c = (idx / (d_head * heads * seq_len)) % chunks;
        int b = idx / (d_head * heads * seq_len * chunks);
        
        float sum = 0.0f;
        for (int s = 0; s <= l; s++) {
            for (int n = 0; n < d_state; n++) {
                sum += C[b*chunks*seq_len*heads*d_state + c*seq_len*heads*d_state + l*heads*d_state + h*d_state + n] *
                       B[b*chunks*seq_len*heads*d_state + c*seq_len*heads*d_state + s*heads*d_state + h*d_state + n] *
                       L[b*heads*chunks*seq_len*seq_len + h*chunks*seq_len*seq_len + c*seq_len*seq_len + l*seq_len + s] *
                       X[b*chunks*seq_len*heads*d_head + c*seq_len*heads*d_head + s*heads*d_head + h*d_head + p];
            }
        }
        Y[idx] = sum;
    }
}

torch::Tensor einsum_diagonal_cuda(
    torch::Tensor C, torch::Tensor B, torch::Tensor L, torch::Tensor X
) {
    auto batch_size = C.size(0);
    auto chunks = C.size(1);
    auto seq_len = C.size(2);
    auto heads = C.size(3);
    auto d_state = C.size(4);
    auto d_head = X.size(4);
    
    auto Y = torch::zeros({batch_size, chunks, seq_len, heads, d_head}, X.options());
    
    int total = batch_size * chunks * seq_len * heads * d_head;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    einsum_diagonal_kernel<<<num_blocks, block_size>>>(
        C.data_ptr<float>(), B.data_ptr<float>(), 
        L.data_ptr<float>(), X.data_ptr<float>(),
        Y.data_ptr<float>(),
        batch_size, chunks, seq_len, heads, d_head, d_state
    );
    
    return Y;
}
"""

einsum_cpp_source = """
torch::Tensor einsum_diagonal_cuda(
    torch::Tensor C, torch::Tensor B, torch::Tensor L, torch::Tensor X
);
"""

# Compile custom CUDA kernels
segsum_exp_module = load_inline(
    name="segsum_exp",
    cpp_sources=segsum_exp_cpp_source,
    cuda_sources=segsum_exp_source,
    functions=["segsum_exp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

einsum_module = load_inline(
    name="einsum_ops",
    cpp_sources=einsum_cpp_source,
    cuda_sources=einsum_kernel_source,
    functions=["einsum_diagonal_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        """
        Optimized Mamba SSM with custom CUDA kernels.
        
        :param batch_size: Size of the batch
        :param seq_length: Length of the input sequence
        :param n_heads: Number of attention heads
        :param d_head: Dimension of each head
        :param d_state: Dimension of the state space
        :param block_len: Length of each block for chunked computation
        """
        super(ModelNew, self).__init__()
        
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        
        # Initialize parameters
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        
        # Custom CUDA modules
        self.segsum_exp_module = segsum_exp_module
        self.einsum_module = einsum_module
    
    def segsum(self, x):
        """Original segsum for fallback."""
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum
    
    def forward(self, X, initial_states=None):
        """
        Optimized forward pass with custom CUDA kernels.
        
        :param X: Input tensor of shape (batch, length, n_heads, d_head)
        :param initial_states: Optional initial states
        :return: Output tensor Y and final state
        """
        # Rearrange into blocks/chunks
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]
        
        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_blocks, dim=-1)
        
        # 1. Compute diagonal block outputs with custom kernel
        L = self.segsum_exp_module.segsum_exp_cuda(A_blocks)
        Y_diag = self.einsum_module.einsum_diagonal_cuda(C_blocks, B_blocks, L, X_blocks)
        
        # 2. Compute intra-chunk states
        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", 
                            B_blocks, decay_states, X_blocks)
        
        # 3. Compute inter-chunk recurrence
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        
        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        return new_states[:, -1]