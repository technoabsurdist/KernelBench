import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for segsum operation
segsum_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void segsum_kernel(const float* x, float* out, int T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < T && j < T) {
        if (j <= i) {
            float cumsum_i = 0.0f;
            float cumsum_j = 0.0f;
            
            for (int k = 0; k <= i; k++) {
                if (k < T) cumsum_i += x[k];
            }
            for (int k = 0; k < j; k++) {
                if (k < T) cumsum_j += x[k];
            }
            
            out[i * T + j] = cumsum_i - cumsum_j;
        } else {
            out[i * T + j] = -INFINITY;
        }
    }
}

torch::Tensor segsum_cuda(torch::Tensor x) {
    int T = x.size(-1);
    auto out = torch::full({T, T}, -INFINITY, x.options());
    
    const int block_size = 16;
    dim3 grid((T + block_size - 1) / block_size, (T + block_size - 1) / block_size);
    dim3 block(block_size, block_size);
    
    segsum_kernel<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(), T);
    
    return out;
}
"""

segsum_cpp_source = "torch::Tensor segsum_cuda(torch::Tensor x);"

# Compile the inline CUDA code for segsum
segsum_op = load_inline(
    name="segsum_op",
    cpp_sources=segsum_cpp_source,
    cuda_sources=segsum_cuda_source,
    functions=["segsum_cuda"],
    verbose=False,
)

# Custom CUDA kernel for element-wise exponentiation
exp_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void exp_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = expf(input[idx]);
    }
}

torch::Tensor exp_cuda(torch::Tensor input) {
    int size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    
    exp_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    
    return output;
}
"""

exp_cpp_source = "torch::Tensor exp_cuda(torch::Tensor input);"

# Compile the inline CUDA code for exp
exp_op = load_inline(
    name="exp_op",
    cpp_sources=exp_cpp_source,
    cuda_sources=exp_cuda_source,
    functions=["exp_cuda"],
    verbose=False,
)

# Custom CUDA kernel for cumsum
cumsum_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cumsum_kernel(const float* input, float* output, int size) {
    // Simple implementation - each thread handles one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = 0.0f;
        for (int i = 0; i <= idx; i++) {
            sum += input[i];
        }
        output[idx] = sum;
    }
}

torch::Tensor cumsum_cuda(torch::Tensor input) {
    int size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    
    cumsum_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    
    return output;
}
"""

cumsum_cpp_source = "torch::Tensor cumsum_cuda(torch::Tensor input);"

# Compile the inline CUDA code for cumsum
cumsum_op = load_inline(
    name="cumsum_op",
    cpp_sources=cumsum_cpp_source,
    cuda_sources=cumsum_cuda_source,
    functions=["cumsum_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        """
        Optimized Mamba Structured State Space model implementation with custom CUDA kernels.
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
        
        # Load custom operations
        self.segsum_op = segsum_op
        self.exp_op = exp_op
        self.cumsum_op = cumsum_op
        
    def segsum(self, x):
        """Optimized segment sum calculation using custom CUDA kernel."""
        T = x.size(-1)
        # Use our custom CUDA implementation
        x_flat = x.flatten(0, -2)  # Flatten all but last dimension
        results = []
        for i in range(x_flat.size(0)):
            result = self.segsum_op.segsum_cuda(x_flat[i])
            results.append(result)
        x_segsum = torch.stack(results).view(*x.shape[:-1], T, T)
        
        # Apply mask
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum
    
    def forward(self, X, initial_states=None):
        """
        Forward pass implementing the SSD operation with custom CUDA kernels.
        """
        # Rearrange into blocks/chunks
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]
        
        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
        
        # Use custom cumsum
        A_cumsum = self.cumsum_op.cumsum_cuda(A_blocks.flatten(0, -1)).view_as(A_blocks)
        
        # 1. Compute diagonal block outputs
        L = self.exp_op.exp_cuda(self.segsum(A_blocks))
        Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", 
                             C_blocks, B_blocks, L, X_blocks)
        
        # 2. Compute intra-chunk states
        decay_states = self.exp_op.exp_cuda((A_cumsum[:, :, :, -1:] - A_cumsum))
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", 
                            B_blocks, decay_states, X_blocks)
        
        # 3. Compute inter-chunk recurrence
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        
        padded_A_cumsum = F.pad(A_cumsum[:, :, :, -1], (1, 0))
        decay_chunk = self.exp_op.exp_cuda(self.segsum(padded_A_cumsum))
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states = new_states[:, :-1]
        
        # 4. Compute state-to-output conversion
        state_decay_out = self.exp_op.exp_cuda(A_cumsum)
        Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', 
                           C_blocks, states, state_decay_out)
        
        # Combine diagonal and off-diagonal terms
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        
        return Y

# Test parameters
batch_size = 2048
seq_length = 128
n_heads = 8
d_head = 64
d_state = 16
block_len = 64

def get_inputs():
    return [torch.rand(batch_size, seq_length, n_heads, d_head)]

def get_init_inputs():
    return [batch_size, seq_length, n_heads, d_head, d_state, block_len]