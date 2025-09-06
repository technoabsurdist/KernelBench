import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the Mamba chunked selective scan
mamba_chunk_scan_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel to perform the selective scan within each chunk.
// Fuses the computation of Y_diag and the final chunk state.
// Grid: (B, C, H)
// Block: (N, P) where N=d_state, P=d_head
__global__ void mamba_chunk_scan_kernel(
    const float* __restrict__ x_ptr,
    const float* __restrict__ a_ptr,
    const float* __restrict__ b_ptr,
    const float* __restrict__ c_ptr,
    float* __restrict__ y_ptr,
    float* __restrict__ state_ptr,
    const int B, const int C_chunks, const int L, const int H, const int P, const int N) {

    // Get block indices for batch, chunk, and head
    const int b = blockIdx.x;
    const int c = blockIdx.y;
    const int h = blockIdx.z;

    // Get thread indices for state and head dimensions
    const int n = threadIdx.x;
    const int p = threadIdx.y;

    // Shared memory for A values of the current chunk
    // L is block_len, which is small (e.g., 64)
    extern __shared__ float smem[];
    float* s_A = smem;
    float* s_Y_reduction = &smem[L];

    // Load A for the current chunk (b, c, h) into shared memory
    // Let the first warp handle this to avoid redundant reads
    if (threadIdx.x == 0 && threadIdx.y < L) {
        int l_idx = threadIdx.y;
        // A layout: (B, C, L, H)
        s_A[l_idx] = a_ptr[b * C_chunks * L * H + c * L * H + l_idx * H + h];
    }
    __syncthreads();

    // Each thread (n, p) computes a scalar recurrence
    float current_state_np = 0.0f;

    // Loop over the sequence length L within the chunk
    for (int l = 0; l < L; ++l) {
        // --- State Update ---
        // Get pointers to the current time step's data
        // X layout: (B, C, L, H, P)
        const int x_idx = b * C_chunks * L * H * P + c * L * H * P + l * H * P + h * P + p;
        // B layout: (B, C, L, H, N)
        const int b_idx = b * C_chunks * L * H * N + c * L * H * N + l * H * N + h * N + n;

        float a_val = s_A[l];
        float x_val = x_ptr[x_idx];
        float b_val = b_ptr[b_idx];

        current_state_np = expf(a_val) * current_state_np + b_val * x_val;

        // --- Output Calculation (Y) ---
        // C layout: (B, C, L, H, N)
        const int c_idx = b * C_chunks * L * H * N + c * L * H * N + l * H * N + h * N + n;
        float c_val = c_ptr[c_idx];

        float y_contribution = c_val * current_state_np;

        // Ensure all threads have calculated their contribution before starting reduction
        __syncthreads();

        // Initialize shared memory for this p
        if (threadIdx.x == 0) {
            s_Y_reduction[p] = 0.0f;
        }
        __syncthreads();

        // Perform reduction over n using atomicAdd on shared memory
        atomicAdd(&s_Y_reduction[p], y_contribution);
        
        // Wait for all threads in the block to finish the reduction
        __syncthreads();

        // The first thread for each p writes the final summed value to global memory
        if (threadIdx.x == 0) {
            // Y layout: (B, C, L, H, P)
            const int y_idx = b * C_chunks * L * H * P + c * L * H * P + l * H * P + h * P + p;
            y_ptr[y_idx] = s_Y_reduction[p];
        }
        // A final sync is needed before the next loop iteration because s_Y_reduction is reused
        __syncthreads();
    }

    // After the loop, store the final state of the chunk
    // state layout: (B, C, H, P, N)
    const int state_idx = b * C_chunks * H * P * N + c * H * P * N + h * P * N + p * N + n;
    state_ptr[state_idx] = current_state_np;
}

// C++ wrapper function
std::vector<torch::Tensor> mamba_chunk_scan_forward(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c) {

    // Input shapes:
    // x: (B, C, L, H, P)
    // a: (B, C, L, H)
    // b: (B, C, L, H, N)
    // c: (B, C, L, H, N)
    const auto B = x.size(0);
    const auto C_chunks = x.size(1);
    const auto L = x.size(2);
    const auto H = x.size(3);
    const auto P = x.size(4);
    const auto N = b.size(4);

    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(a.is_cuda(), "Input a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "Input b must be a CUDA tensor");
    TORCH_CHECK(c.is_cuda(), "Input c must be a CUDA tensor");
    
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(a.is_contiguous(), "Input a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Input b must be contiguous");
    TORCH_CHECK(c.is_contiguous(), "Input c must be contiguous");

    // Create output tensors
    auto y = torch::zeros_like(x);
    auto state = torch::zeros({B, C_chunks, H, P, N}, x.options());

    // Kernel launch configuration
    const dim3 grid(B, C_chunks, H);
    const dim3 block(N, P); // d_state, d_head
    
    const int shared_mem_size = (L + P) * sizeof(float);

    // Launch the kernel
    mamba_chunk_scan_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        y.data_ptr<float>(),
        state.data_ptr<float>(),
        B, C_chunks, L, H, P, N
    );
    
    return {y, state};
}
"""

mamba_chunk_scan_cpp_source = (
    "std::vector<torch::Tensor> mamba_chunk_scan_forward(torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor c);"
)

# Compile the inline CUDA code
mamba_ops = load_inline(
    name="mamba_ops",
    cpp_sources=mamba_chunk_scan_cpp_source,
    cuda_sources=mamba_chunk_scan_source,
    functions=["mamba_chunk_scan_forward"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        """
        Mamba Structured State Space model implementation with a custom CUDA kernel
        for the intra-chunk selective scan.
        
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
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads).cuda())
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state).cuda())
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state).cuda())
        
    def segsum(self, x):
        """Naive segment sum calculation, kept for the inter-chunk recurrence."""
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum
    
    def forward(self, X, initial_states=None):
        """
        Forward pass implementing the SSD operation with a custom CUDA kernel.
        
        :param X: Input tensor of shape (batch, length, n_heads, d_head)
        :param initial_states: Optional initial states
        :return: Output tensor Y and final state
        """
        # Ensure inputs are on the correct device
        X = X.cuda()

        # Rearrange into blocks/chunks
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]
        
        # 1. & 2. Compute diagonal block outputs and intra-chunk states with a fused CUDA kernel
        # This replaces the original segsum, exp, and two einsum operations for the intra-chunk part.
        # The kernel expects contiguous tensors.
        _, states = mamba_ops.mamba_chunk_scan_forward(
            X_blocks.contiguous(), 
            A_blocks.contiguous(), 
            B_blocks.contiguous(), 
            C_blocks.contiguous()
        )
        
        # 3. Compute inter-chunk recurrence (unchanged from original)
        # This part is sequential over chunks and small, so PyTorch is acceptable.
        A_blocks_rearranged = rearrange(A_blocks, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_blocks_rearranged, dim=-1)

        if initial_states is None:
            # The kernel output `states` has shape (b, c, h, p, n)
            initial_states = torch.zeros_like(states[:, :1])
        
        states = torch.cat([initial_states, states], dim=1)
        
        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        
        return new_states[:, -1]

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