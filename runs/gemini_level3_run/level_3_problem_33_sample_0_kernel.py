import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused RNN cell operation
# This kernel fuses: torch.cat, nn.Linear (matmul + bias), and nn.Tanh
fused_rnn_cell_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For tanhf

// Define the tile dimension for shared memory optimization
#define TILE_DIM 32

__global__ void fused_rnn_cell_kernel(
    const float* __restrict__ x,
    const float* __restrict__ hidden_prev,
    const float* __restrict__ W,
    const float* __restrict__ bias,
    float* __restrict__ hidden_new,
    int B, int I, int H) {

    const int K = I + H;

    // Shared memory for tiles of the input matrix (A) and weight matrix (B)
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Identify the portion of the output matrix this block will compute
    const int out_row_base = blockIdx.y * TILE_DIM;
    const int out_col_base = blockIdx.x * TILE_DIM;

    // Accumulator for the dot product, stored in registers
    float acc = 0.0f;

    // Loop over tiles of the K dimension (input_size + hidden_size)
    for (int k_tile = 0; k_tile < (K + TILE_DIM - 1) / TILE_DIM; ++k_tile) {
        const int k_base = k_tile * TILE_DIM;

        // --- Load a tile of A and B into shared memory ---
        
        // Load tile for A (from x and hidden_prev)
        const int a_row = out_row_base + ty;
        const int a_col = k_base + tx;
        if (a_row < B && a_col < K) {
            if (a_col < I) {
                s_A[ty][tx] = x[a_row * I + a_col];
            } else {
                s_A[ty][tx] = hidden_prev[a_row * H + (a_col - I)];
            }
        } else {
            s_A[ty][tx] = 0.0f;
        }

        // Load tile for B (from W, transposed)
        const int w_row = out_col_base + ty;
        const int w_col = k_base + tx;
        if (w_row < H && w_col < K) {
            s_B[ty][tx] = W[w_row * K + w_col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        // --- Compute the dot product for the tile ---
        for (int i = 0; i < TILE_DIM; ++i) {
            acc += s_A[ty][i] * s_B[tx][i];
        }

        __syncthreads();
    }

    // --- Finalize and write the result to the output matrix ---
    const int out_row = out_row_base + ty;
    const int out_col = out_col_base + tx;

    if (out_row < B && out_col < H) {
        acc += bias[out_col];
        hidden_new[out_row * H + out_col] = tanhf(acc);
    }
}

torch::Tensor fused_rnn_cell_cuda(
    torch::Tensor x,
    torch::Tensor hidden_prev,
    torch::Tensor W,
    torch::Tensor bias) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(hidden_prev.is_cuda(), "hidden_prev must be a CUDA tensor");
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(hidden_prev.is_contiguous(), "hidden_prev must be contiguous");
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(hidden_prev.dim() == 2, "hidden_prev must be 2D");
    TORCH_CHECK(W.dim() == 2, "W must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    const int B = x.size(0);
    const int I = x.size(1);
    const int H = hidden_prev.size(1);
    const int K = I + H;

    TORCH_CHECK(hidden_prev.size(0) == B, "Batch sizes of x and hidden_prev must match");
    TORCH_CHECK(W.size(0) == H, "W dimension 0 must be hidden_size");
    TORCH_CHECK(W.size(1) == K, "W dimension 1 must be input_size + hidden_size");
    TORCH_CHECK(bias.size(0) == H, "bias size must be hidden_size");

    auto hidden_new = torch::empty_like(hidden_prev);

    const dim3 threads(TILE_DIM, TILE_DIM);
    const dim3 blocks(
        (H + TILE_DIM - 1) / TILE_DIM,
        (B + TILE_DIM - 1) / TILE_DIM
    );

    fused_rnn_cell_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        hidden_prev.data_ptr<float>(),
        W.data_ptr<float>(),
        bias.data_ptr<float>(),
        hidden_new.data_ptr<float>(),
        B, I, H
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return hidden_new;
}
"""

fused_rnn_cell_cpp_source = (
    "torch::Tensor fused_rnn_cell_cuda(torch::Tensor x, torch::Tensor hidden_prev, torch::Tensor W, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_rnn_op = load_inline(
    name="fused_rnn_op",
    cpp_sources=fused_rnn_cell_cpp_source,
    cuda_sources=fused_rnn_cell_source,
    functions=["fused_rnn_cell_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the Vanilla RNN model with a custom fused CUDA kernel.
        
        :param input_size: The number of input features (int).
        :param hidden_size: The size of the hidden state (int).
        :param output_size: The number of output features (int).
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden = torch.randn((batch_size, hidden_size))
        
        # Define parameters for the fused i2h cell, replacing nn.Linear
        self.i2h_weight = nn.Parameter(torch.empty(hidden_size, input_size + hidden_size))
        self.i2h_bias = nn.Parameter(torch.empty(hidden_size))
        
        # Define the standard hidden to output layer
        self.h2o = nn.Linear(hidden_size, output_size)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Replicate the default initialization of nn.Linear
        nn.init.kaiming_uniform_(self.i2h_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.i2h_weight)
        if fan_in > 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.i2h_bias, -bound, bound)
        # h2o is initialized by its own class constructor
    
    def forward(self, x: torch.Tensor, initial_hidden=None) -> torch.Tensor:
        """
        Forward pass of the Vanilla RNN using the custom fused kernel.
        
        :param x: Input tensor of shape (batch_size, input_size).
        :param hidden: Hidden state tensor of shape (batch_size, hidden_size).
        :return: Output tensor of shape (batch_size, output_size).
        """
        if initial_hidden is not None:
            self.hidden.copy_(initial_hidden)
        self.hidden = self.hidden.to(x.device)
        
        # Call the custom fused kernel for the i2h and tanh operations
        self.hidden = fused_rnn_op.fused_rnn_cell_cuda(
            x, self.hidden, self.i2h_weight, self.i2h_bias
        )
        
        # Compute output using the standard linear layer
        output = self.h2o(self.hidden)
        return output

batch_size = 256
input_size = 16384
hidden_size = 16384
output_size = 8192
sequence_length = 256

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda().contiguous(), torch.rand(batch_size, hidden_size).cuda().contiguous()]

def get_init_inputs():
    return [input_size, hidden_size, output_size]