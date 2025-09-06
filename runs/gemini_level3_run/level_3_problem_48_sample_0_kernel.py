import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for the Mamba block operations
mamba_kernels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Using float for simplicity, can be templated for other types
using scalar_t = float;

__global__ void mamba_diag_kernel(
    const scalar_t* __restrict__ A_cumsum_in, // (B, H, C, L)
    const scalar_t* __restrict__ B_in,        // (B, C, L, H, N)
    const scalar_t* __restrict__ C_in,        // (B, C, L, H, N)
    const scalar_t* __restrict__ X_in,        // (B, C, L, H, P)
    scalar_t* Y_out,                          // (B, C, L, H, P)
    const int B, const int C, const int L, const int H, const int N, const int P
) {
    // Grid: (B, H, C)
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int c = blockIdx.z;

    // Shared memory for one chunk
    // Total size: (L + L*N + L*N + L*P + L*L) * sizeof(scalar_t)
    // For L=64, N=16, P=64: (64 + 1024 + 1024 + 4096 + 4096) * 4 = 10284 * 4 = 41136 bytes.
    extern __shared__ scalar_t smem[];
    scalar_t* A_cumsum_sh = smem;
    scalar_t* B_sh = &smem[L];
    scalar_t* C_sh = &smem[L + L * N];
    scalar_t* X_sh = &smem[L + 2 * L * N];
    scalar_t* CB_sh = &smem[L + 2 * L * N + L * P];

    // Pointers to the start of the current chunk in global memory
    const scalar_t* A_cumsum_ptr = A_cumsum_in + b * H * C * L + h * C * L + c * L;
    const scalar_t* B_ptr = B_in + b * C * L * H * N + c * L * H * N;
    const scalar_t* C_ptr = C_in + b * C * L * H * N + c * L * H * N;
    const scalar_t* X_ptr = X_in + b * C * L * H * P + c * L * H * P;
    scalar_t* Y_ptr = Y_out + b * C * L * H * P + c * L * H * P;

    // Load data into shared memory using a 1D block
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        A_cumsum_sh[i] = A_cumsum_ptr[i];
    }
    for (int i = threadIdx.x; i < L * N; i += blockDim.x) {
        int l = i / N;
        int n = i % N;
        B_sh[i] = B_ptr[l * H * N + h * N + n];
        C_sh[i] = C_ptr[l * H * N + h * N + n];
    }
    for (int i = threadIdx.x; i < L * P; i += blockDim.x) {
        int l = i / P;
        int p = i % P;
        X_sh[i] = X_ptr[l * H * P + h * P + p];
    }
    __syncthreads();

    // Step 1: Compute CB = C @ B.T in shared memory
    // CB is (L, L). Each thread computes multiple elements.
    for (int i = threadIdx.x; i < L * L; i += blockDim.x) {
        int l = i / L;
        int s = i % L;
        scalar_t sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            sum += C_sh[l * N + n] * B_sh[s * N + n];
        }
        CB_sh[i] = sum;
    }
    __syncthreads();

    // Step 2: Compute Y_diag = (L * CB) @ X
    // Each thread computes one row of the output Y_diag for the current head
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        for (int p = 0; p < P; ++p) {
            scalar_t y_val = 0.0f;
            for (int s = 0; s <= l; ++s) {
                scalar_t L_ls = expf(A_cumsum_sh[l] - A_cumsum_sh[s]);
                scalar_t lcb_val = L_ls * CB_sh[l * L + s];
                y_val += lcb_val * X_sh[s * P + p];
            }
            Y_ptr[l * H * P + h * P + p] = y_val;
        }
    }
}

__global__ void mamba_states_kernel(
    const scalar_t* __restrict__ A_cumsum_in, // (B, H, C, L)
    const scalar_t* __restrict__ B_in,        // (B, C, L, H, N)
    const scalar_t* __restrict__ X_in,        // (B, C, L, H, P)
    scalar_t* states_out,                     // (B, C, H, P, N)
    const int B, const int C, const int L, const int H, const int N, const int P
) {
    // Grid: (B, C, H)
    const int b = blockIdx.x;
    const int c = blockIdx.y;
    const int h = blockIdx.z;

    // Shared memory for A_cumsum
    extern __shared__ scalar_t smem[];
    scalar_t* A_cumsum_sh = smem;

    // Pointers to global memory for the current chunk
    const scalar_t* A_cumsum_ptr = A_cumsum_in + b * H * C * L + h * C * L + c * L;
    const scalar_t* B_ptr = B_in + b * C * L * H * N + c * L * H * N;
    const scalar_t* X_ptr = X_in + b * C * L * H * P + c * L * H * P;
    scalar_t* states_ptr = states_out + b * C * H * P * N + c * H * P * N;

    // Load A_cumsum into shared memory
    if (threadIdx.x < L) {
        A_cumsum_sh[threadIdx.x] = A_cumsum_ptr[threadIdx.x];
    }
    __syncthreads();

    const scalar_t A_cumsum_last = A_cumsum_sh[L - 1];

    // Block is 2D: (P, N). Each thread computes one element of the output state matrix.
    const int p = threadIdx.x;
    const int n = threadIdx.y;

    if (p < P && n < N) {
        scalar_t state_val = 0.0f;
        for (int l = 0; l < L; ++l) {
            scalar_t decay = expf(A_cumsum_last - A_cumsum_sh[l]);
            scalar_t b_val = B_ptr[l * H * N + h * N + n];
            scalar_t x_val = X_ptr[l * H * P + h * P + p];
            state_val += b_val * decay * x_val;
        }
        // states_out is (B, C, H, P, N)
        states_ptr[h * P * N + p * N + n] = state_val;
    }
}

// Wrapper functions
torch::Tensor mamba_diag_cuda(
    torch::Tensor A_cumsum, torch::Tensor B_blocks,
    torch::Tensor C_blocks, torch::Tensor X_blocks
) {
    const auto B = X_blocks.size(0);
    const auto C = X_blocks.size(1);
    const auto L = X_blocks.size(2);
    const auto H = X_blocks.size(3);
    const auto P = X_blocks.size(4);
    const auto N = B_blocks.size(4);

    auto Y_diag = torch::empty_like(X_blocks);

    TORCH_CHECK(A_cumsum.is_contiguous(), "A_cumsum must be contiguous");
    TORCH_CHECK(B_blocks.is_contiguous(), "B_blocks must be contiguous");
    TORCH_CHECK(C_blocks.is_contiguous(), "C_blocks must be contiguous");
    TORCH_CHECK(X_blocks.is_contiguous(), "X_blocks must be contiguous");

    const dim3 grid_diag(B, H, C);
    const dim3 block_diag(256);
    const int smem_size_diag = (L + 2 * L * N + L * P + L * L) * sizeof(scalar_t);

    mamba_diag_kernel<<<grid_diag, block_diag, smem_size_diag>>>(
        A_cumsum.data_ptr<scalar_t>(), B_blocks.data_ptr<scalar_t>(),
        C_blocks.data_ptr<scalar_t>(), X_blocks.data_ptr<scalar_t>(),
        Y_diag.data_ptr<scalar_t>(),
        B, C, L, H, N, P
    );
    return Y_diag;
}

torch::Tensor mamba_states_cuda(
    torch::Tensor A_cumsum, torch::Tensor B_blocks, torch::Tensor X_blocks
) {
    const auto B = X_blocks.size(0);
    const auto C = X_blocks.size(1);
    const auto L = X_blocks.size(2);
    const auto H = X_blocks.size(3);
    const auto P = X_blocks.size(4);
    const auto N = B_blocks.size(4);

    auto states = torch::empty({B, C, H, P, N}, X_blocks.options());

    TORCH_CHECK(A_cumsum.is_contiguous(), "A_cumsum must be contiguous");
    TORCH_CHECK(B_blocks.is_contiguous(), "B_blocks must be contiguous");
    TORCH_CHECK(X_blocks.is_contiguous(), "X_blocks must be contiguous");

    const dim3 grid_states(B, C, H);
    const dim3 block_states(P, N);
    const int smem_size_states = L * sizeof(scalar_t);

    mamba_states_kernel<<<grid_states, block_states, smem_size_states>>>(
        A_cumsum.data_ptr<scalar_t>(), B_blocks.data_ptr<scalar_t>(),
        X_blocks.data_ptr<scalar_t>(), states.data_ptr<scalar_t>(),
        B, C, L, H, N, P
    );
    return states;
}
"""

mamba_kernels_cpp_source = """
torch::Tensor mamba_diag_cuda(torch::Tensor A_cumsum, torch::Tensor B_blocks, torch::Tensor C_blocks, torch::Tensor X_blocks);
torch::Tensor mamba_states_cuda(torch::Tensor A_cumsum, torch::Tensor B_blocks, torch::Tensor X_blocks);
"""

# JIT compile the CUDA kernels
mamba_kernels = load_inline(
    name="mamba_kernels",
    cpp_sources=mamba_kernels_cpp_source,
    cuda_sources=mamba_kernels_source,
    functions=["mamba_diag_cuda", "mamba_states_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        """
        Mamba Structured State Space model with custom CUDA kernel optimizations.
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
        
    def segsum(self, x):
        """Helper for the remaining PyTorch-based recurrence calculation."""
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def forward(self, X, initial_states=None):
        """
        Forward pass implementing the SSD operation with fused CUDA kernels.
        
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
        
        # 1. Compute diagonal block outputs using a fused CUDA kernel
        # This kernel replaces segsum and the first einsum
        Y_diag = mamba_kernels.mamba_diag_cuda(
            A_cumsum.contiguous(), 
            B_blocks.contiguous(), 
            C_blocks.contiguous(), 
            X_blocks.contiguous()
        )
        
        # 2. Compute intra-chunk states using a fused CUDA kernel
        # This kernel replaces the decay_states calculation and the second einsum
        states = mamba_kernels.mamba_states_cuda(
            A_cumsum.contiguous(),
            B_blocks.contiguous(),
            X_blocks.contiguous()
        )
        
        # 3. Compute inter-chunk recurrence (remaining in PyTorch)
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        
        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states = new_states[:, :-1]
        
        # 4. Compute state-to-output conversion (remaining in PyTorch)
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', 
                           C_blocks, states, state_decay_out)
        
        # Combine diagonal and off-diagonal terms
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        
        return Y