import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused segsum operation
segsum_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void segsum_kernel(
    const float* x, 
    float* output, 
    int batch_size, 
    int heads, 
    int chunks, 
    int length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * heads * chunks * length * length;
    
    if (idx >= total_elements) return;
    
    int l2 = idx % length;
    int l1 = (idx / length) % length;
    int c = (idx / (length * length)) % chunks;
    int h = (idx / (length * length * chunks)) % heads;
    int b = idx / (length * length * chunks * heads);
    
    if (l1 < l2) {
        output[idx] = -INFINITY;
    } else {
        float sum = 0.0f;
        int base_idx = b * heads * chunks * length + h * chunks * length + c * length;
        for (int i = l2; i <= l1; i++) {
            sum += x[base_idx + i];
        }
        output[idx] = sum;
    }
}

torch::Tensor segsum_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto heads = x.size(1);
    auto chunks = x.size(2);
    auto length = x.size(3);
    
    auto output = torch::zeros({batch_size, heads, chunks, length, length}, 
                              x.options());
    
    int total_elements = batch_size * heads * chunks * length * length;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    segsum_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, heads, chunks, length
    );
    
    return output;
}
"""

# Custom CUDA kernel for fused exp(segsum) operation
exp_segsum_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void exp_segsum_kernel(
    const float* x, 
    float* output, 
    int batch_size, 
    int heads, 
    int chunks, 
    int length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * heads * chunks * length * length;
    
    if (idx >= total_elements) return;
    
    int l2 = idx % length;
    int l1 = (idx / length) % length;
    int c = (idx / (length * length)) % chunks;
    int h = (idx / (length * length * chunks)) % heads;
    int b = idx / (length * length * chunks * heads);
    
    if (l1 < l2) {
        output[idx] = 0.0f;
    } else {
        float sum = 0.0f;
        int base_idx = b * heads * chunks * length + h * chunks * length + c * length;
        for (int i = l2; i <= l1; i++) {
            sum += x[base_idx + i];
        }
        output[idx] = expf(sum);
    }
}

torch::Tensor exp_segsum_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto heads = x.size(1);
    auto chunks = x.size(2);
    auto length = x.size(3);
    
    auto output = torch::zeros({batch_size, heads, chunks, length, length}, 
                              x.options());
    
    int total_elements = batch_size * heads * chunks * length * length;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    exp_segsum_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, heads, chunks, length
    );
    
    return output;
}
"""

# Custom CUDA kernel for decay states computation
decay_states_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void compute_decay_states_kernel(
    const float* A_cumsum,
    float* decay_states,
    int batch_size,
    int heads,
    int chunks,
    int length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * heads * chunks * length;
    
    if (idx >= total_elements) return;
    
    int l = idx % length;
    int c = (idx / length) % chunks;
    int h = (idx / (length * chunks)) % heads;
    int b = idx / (length * chunks * heads);
    
    int last_idx = b * heads * chunks * length + h * chunks * length + c * length + (length - 1);
    int curr_idx = idx;
    
    float last_val = A_cumsum[last_idx];
    float curr_val = A_cumsum[curr_idx];
    
    decay_states[idx] = expf(last_val - curr_val);
}

torch::Tensor compute_decay_states_cuda(torch::Tensor A_cumsum) {
    auto batch_size = A_cumsum.size(0);
    auto heads = A_cumsum.size(1);
    auto chunks = A_cumsum.size(2);
    auto length = A_cumsum.size(3);
    
    auto decay_states = torch::zeros_like(A_cumsum);
    
    int total_elements = batch_size * heads * chunks * length;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    compute_decay_states_kernel<<<blocks, threads>>>(
        A_cumsum.data_ptr<float>(),
        decay_states.data_ptr<float>(),
        batch_size, heads, chunks, length
    );
    
    return decay_states;
}
"""

segsum_cpp_source = "torch::Tensor segsum_cuda(torch::Tensor x);"
exp_segsum_cpp_source = "torch::Tensor exp_segsum_cuda(torch::Tensor x);"
decay_states_cpp_source = "torch::Tensor compute_decay_states_cuda(torch::Tensor A_cumsum);"

# Load custom CUDA kernels
segsum_module = load_inline(
    name="segsum_module",
    cpp_sources=segsum_cpp_source,
    cuda_sources=segsum_cuda_source,
    functions=["segsum_cuda"],
    verbose=False,
)

exp_segsum_module = load_inline(
    name="exp_segsum_module",
    cpp_sources=exp_segsum_cpp_source,
    cuda_sources=exp_segsum_cuda_source,
    functions=["exp_segsum_cuda"],
    verbose=False,
)

decay_states_module = load_inline(
    name="decay_states_module",
    cpp_sources=decay_states_cpp_source,
    cuda_sources=decay_states_cuda_source,
    functions=["compute_decay_states_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super(ModelNew, self).__init__()
        
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        
        self.segsum_cuda = segsum_module.segsum_cuda
        self.exp_segsum_cuda = exp_segsum_module.exp_segsum_cuda
        self.compute_decay_states_cuda = decay_states_module.compute_decay_states_cuda
    
    def forward(self, X, initial_states=None):
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]
        
        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_blocks, dim=-1)
        
        # Use custom CUDA kernel for exp(segsum)
        L = self.exp_segsum_cuda(A_blocks)
        Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", 
                             C_blocks, B_blocks, L, X_blocks)
        
        # Use custom CUDA kernel for decay states
        decay_states = self.compute_decay_states_cuda(A_cumsum)
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", 
                            B_blocks, decay_states, X_blocks)
        
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        
        # Use custom CUDA kernel for exp(segsum) on padded cumsum
        padded_cumsum = F.pad(A_cumsum[:, :, :, -1], (1, 0))
        decay_chunk = self.exp_segsum_cuda(padded_cumsum)
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states = new_states[:, :-1]
        
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', 
                           C_blocks, states, state_decay_out)
        
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        
        return Y

def get_inputs():
    return [torch.rand(2048, 128, 8, 64).cuda()]

def get_init_inputs():
    return [2048, 128, 8, 64, 16, 64]