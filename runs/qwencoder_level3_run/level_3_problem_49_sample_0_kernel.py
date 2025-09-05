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
        if (j > i) {
            out[i * T + j] = -INFINITY;
        } else {
            float x_cumsum_i = 0.0f;
            float x_cumsum_j = 0.0f;
            
            for (int k = 0; k <= i; k++) {
                x_cumsum_i += x[k];
            }
            
            for (int k = 0; k <= j; k++) {
                x_cumsum_j += x[k];
            }
            
            out[i * T + j] = x_cumsum_i - x_cumsum_j;
        }
    }
}

torch::Tensor segsum_cuda(torch::Tensor x) {
    auto T = x.size(-1);
    auto out = torch::full({T, T}, -INFINITY, x.options());
    
    const int block_size = 16;
    dim3 block_dim(block_size, block_size);
    dim3 grid_dim((T + block_size - 1) / block_size, (T + block_size - 1) / block_size);
    
    segsum_kernel<<<grid_dim, block_dim>>>(x.data_ptr<float>(), out.data_ptr<float>(), T);
    
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

# Custom CUDA kernel for einsum operations
ssd_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void compute_Y_diag_kernel(
    const float* C_blocks, const float* B_blocks, const float* L, const float* X_blocks,
    float* Y_diag,
    int b, int c, int l, int h, int n, int p
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = b * c * l * h * p;
    
    if (idx < total_elements) {
        int b_idx = idx / (c * l * h * p);
        int remaining = idx % (c * l * h * p);
        int c_idx = remaining / (l * h * p);
        remaining = remaining % (l * h * p);
        int l_idx = remaining / (h * p);
        remaining = remaining % (h * p);
        int h_idx = remaining / p;
        int p_idx = remaining % p;
        
        float sum = 0.0f;
        for (int s = 0; s < l; s++) {
            for (int n_idx = 0; n_idx < n; n_idx++) {
                float c_val = C_blocks[(((b_idx * c + c_idx) * l + l_idx) * h + h_idx) * n + n_idx];
                float b_val = B_blocks[(((b_idx * c + c_idx) * l + s) * h + h_idx) * n + n_idx];
                float l_val = L[(((b_idx * h + h_idx) * c + c_idx) * l + l_idx) * l + s];
                float x_val = X_blocks[(((b_idx * c + c_idx) * l + s) * h + h_idx) * p + p_idx];
                sum += c_val * b_val * l_val * x_val;
            }
        }
        
        Y_diag[idx] = sum;
    }
}

torch::Tensor compute_Y_diag_cuda(
    torch::Tensor C_blocks, torch::Tensor B_blocks, torch::Tensor L, torch::Tensor X_blocks
) {
    auto b = C_blocks.size(0);
    auto c = C_blocks.size(1);
    auto l = C_blocks.size(2);
    auto h = C_blocks.size(3);
    auto n = C_blocks.size(4);
    auto p = X_blocks.size(4);
    
    auto Y_diag = torch::zeros({b, c, l, h, p}, X_blocks.options());
    
    int total_elements = b * c * l * h * p;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    compute_Y_diag_kernel<<<num_blocks, block_size>>>(
        C_blocks.data_ptr<float>(), B_blocks.data_ptr<float>(),
        L.data_ptr<float>(), X_blocks.data_ptr<float>(),
        Y_diag.data_ptr<float>(),
        b, c, l, h, n, p
    );
    
    return Y_diag;
}

__global__ void compute_states_kernel(
    const float* B_blocks, const float* decay_states, const float* X_blocks,
    float* states,
    int b, int c, int l, int h, int n, int p
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = b * c * h * n * p;
    
    if (idx < total_elements) {
        int b_idx = idx / (c * h * n * p);
        int remaining = idx % (c * h * n * p);
        int c_idx = remaining / (h * n * p);
        remaining = remaining % (h * n * p);
        int h_idx = remaining / (n * p);
        remaining = remaining % (n * p);
        int n_idx = remaining / p;
        int p_idx = remaining % p;
        
        float sum = 0.0f;
        for (int l_idx = 0; l_idx < l; l_idx++) {
            float b_val = B_blocks[(((b_idx * c + c_idx) * l + l_idx) * h + h_idx) * n + n_idx];
            float d_val = decay_states[((b_idx * c + c_idx) * h + h_idx) * l + l_idx];
            float x_val = X_blocks[(((b_idx * c + c_idx) * l + l_idx) * h + h_idx) * p + p_idx];
            sum += b_val * d_val * x_val;
        }
        
        states[(((b_idx * c + c_idx) * h + h_idx) * n + n_idx) * p + p_idx] = sum;
    }
}

torch::Tensor compute_states_cuda(
    torch::Tensor B_blocks, torch::Tensor decay_states, torch::Tensor X_blocks
) {
    auto b = B_blocks.size(0);
    auto c = B_blocks.size(1);
    auto l = B_blocks.size(2);
    auto h = B_blocks.size(3);
    auto n = B_blocks.size(4);
    auto p = X_blocks.size(4);
    
    auto states = torch::zeros({b, c, h, n, p}, X_blocks.options());
    
    int total_elements = b * c * h * n * p;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    compute_states_kernel<<<num_blocks, block_size>>>(
        B_blocks.data_ptr<float>(), decay_states.data_ptr<float>(),
        X_blocks.data_ptr<float>(),
        states.data_ptr<float>(),
        b, c, l, h, n, p
    );
    
    return states;
}

__global__ void compute_new_states_kernel(
    const float* decay_chunk, const float* states,
    float* new_states,
    int b, int z, int h, int c, int n, int p
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = b * z * h * n * p;
    
    if (idx < total_elements) {
        int b_idx = idx / (z * h * n * p);
        int remaining = idx % (z * h * n * p);
        int z_idx = remaining / (h * n * p);
        remaining = remaining % (h * n * p);
        int h_idx = remaining / (n * p);
        remaining = remaining % (n * p);
        int n_idx = remaining / p;
        int p_idx = remaining % p;
        
        float sum = 0.0f;
        for (int c_idx = 0; c_idx < c; c_idx++) {
            float d_val = decay_chunk[((b_idx * h + h_idx) * z + z_idx) * c + c_idx];
            float s_val = states[(((b_idx * c + c_idx) * h + h_idx) * n + n_idx) * p + p_idx];
            sum += d_val * s_val;
        }
        
        new_states[(((b_idx * z + z_idx) * h + h_idx) * n + n_idx) * p + p_idx] = sum;
    }
}

torch::Tensor compute_new_states_cuda(
    torch::Tensor decay_chunk, torch::Tensor states
) {
    auto b = decay_chunk.size(0);
    auto h = decay_chunk.size(1);
    auto z = decay_chunk.size(2);
    auto c = decay_chunk.size(3);
    auto n = states.size(3);
    auto p = states.size(4);
    
    auto new_states = torch::zeros({b, z, h, n, p}, states.options());
    
    int total_elements = b * z * h * n * p;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    compute_new_states_kernel<<<num_blocks, block_size>>>(
        decay_chunk.data_ptr<float>(), states.data_ptr<float>(),
        new_states.data_ptr<float>(),
        b, z, h, c, n, p
    );
    
    return new_states;
}
"""

ssd_cpp_source = """
torch::Tensor compute_Y_diag_cuda(torch::Tensor C_blocks, torch::Tensor B_blocks, torch::Tensor L, torch::Tensor X_blocks);
torch::Tensor compute_states_cuda(torch::Tensor B_blocks, torch::Tensor decay_states, torch::Tensor X_blocks);
torch::Tensor compute_new_states_cuda(torch::Tensor decay_chunk, torch::Tensor states);
"""

# Compile the inline CUDA code for SSD operations
ssd_ops = load_inline(
    name="ssd_ops",
    cpp_sources=ssd_cpp_source,
    cuda_sources=ssd_cuda_source,
    functions=["compute_Y_diag_cuda", "compute_states_cuda", "compute_new_states_cuda"],
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
        self.ssd_ops = ssd_ops
    
    def segsum(self, x):
        """Optimized segment sum calculation using custom CUDA kernel."""
        T = x.size(-1)
        x_flat = x.flatten(0, -2)  # Flatten batch and other dimensions
        results = []
        for i in range(x_flat.size(0)):
            result = self.segsum_op.segsum_cuda(x_flat[i])
            results.append(result)
        output = torch.stack(results).view(*x.shape[:-1], T, T)
        return output
    
    def forward(self, X, initial_states=None):
        """
        Optimized forward pass implementing the SSD operation with custom CUDA kernels.
        """
        # Rearrange into blocks/chunks
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]
        
        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_blocks, dim=-1)
        
        # 1. Compute diagonal block outputs
        L = torch.exp(self.segsum(A_blocks))
        # Use custom CUDA kernel for the einsum operation
        Y_diag = self.ssd_ops.compute_Y_diag_cuda(C_blocks, B_blocks, L, X_blocks)
        
        # 2. Compute intra-chunk states
        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        # Use custom CUDA kernel for the einsum operation
        states = self.ssd_ops.compute_states_cuda(B_blocks, decay_states.squeeze(-1), X_blocks)
        
        # 3. Compute inter-chunk recurrence
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        
        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        # Use custom CUDA kernel for the einsum operation
        new_states = self.ssd_ops.compute_new_states_cuda(decay_chunk, states)
        return new_states[:, -1]