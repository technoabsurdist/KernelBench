import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused slicing and linear layer
fused_slice_linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void fused_slice_linear_kernel(const float* out_tensor, const float* weight, const float* bias, float* result,
                                          int B, int S, int H, int O) {
    // This kernel computes: result = out_tensor[:, -1, :] @ weight.T + bias
    // Let A = out_tensor[:, -1, :] of shape (B, H)
    // Let B = weight of shape (O, H)
    // We compute C = A @ B.T + bias. C has shape (B, O)

    // Shared memory for tiles of A and B
    __shared__ float a_tile[TILE_DIM][TILE_DIM];
    __shared__ float b_tile[TILE_DIM][TILE_DIM];

    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global row and column for the output element this thread will compute
    int row = by * TILE_DIM + ty; // Corresponds to batch dimension
    int col = bx * TILE_DIM + tx; // Corresponds to output feature dimension

    float C_val = 0.0f;

    // Loop over tiles of A and B to compute the dot product
    for (int t = 0; t < (H + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Load tile of A into shared memory
        // Coalesced read from out_tensor
        int a_col = t * TILE_DIM + tx;
        if (row < B && a_col < H) {
            // A is out_tensor[:, -1, :]. Indexing is out_tensor[row, S-1, a_col]
            a_tile[ty][tx] = out_tensor[row * S * H + (S - 1) * H + a_col];
        } else {
            a_tile[ty][tx] = 0.0f;
        }

        // Load tile of B (weight) into shared memory
        // Coalesced read from weight
        int b_row_in_tile = ty;
        int b_col_in_tile = tx;
        int b_row_global = bx * TILE_DIM + b_row_in_tile;
        int b_col_global = t * TILE_DIM + b_col_in_tile;
        if (b_row_global < O && b_col_global < H) {
            b_tile[b_row_in_tile][b_col_in_tile] = weight[b_row_global * H + b_col_global];
        } else {
            b_tile[b_row_in_tile][b_col_in_tile] = 0.0f;
        }

        __syncthreads();

        // Multiply the two tiles
        for (int k = 0; k < TILE_DIM; ++k) {
            C_val += a_tile[ty][k] * b_tile[tx][k];
        }

        __syncthreads();
    }

    // Write result to global memory
    if (row < B && col < O) {
        result[row * O + col] = C_val + bias[col];
    }
}

torch::Tensor fused_slice_linear_forward(
    torch::Tensor out_tensor,
    torch::Tensor weight,
    torch::Tensor bias) {

    TORCH_CHECK(out_tensor.is_cuda(), "out_tensor must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    out_tensor = out_tensor.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    const auto B = out_tensor.size(0);
    const auto S = out_tensor.size(1);
    const auto H = out_tensor.size(2);
    const auto O = weight.size(0);

    TORCH_CHECK(weight.size(1) == H, "weight dimension mismatch");
    TORCH_CHECK(bias.size(0) == O, "bias dimension mismatch");

    auto result = torch::empty({B, O}, out_tensor.options());

    dim3 block_size(TILE_DIM, TILE_DIM);
    dim3 grid_size((O + TILE_DIM - 1) / TILE_DIM, (B + TILE_DIM - 1) / TILE_DIM);

    fused_slice_linear_kernel<<<grid_size, block_size>>>(
        out_tensor.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        result.data_ptr<float>(),
        B, S, H, O
    );

    return result;
}
"""

fused_slice_linear_cpp_source = (
    "torch::Tensor fused_slice_linear_forward(torch::Tensor out_tensor, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_slice_linear_cpp_source,
    cuda_sources=fused_slice_linear_source,
    functions=["fused_slice_linear_forward"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model with a custom CUDA kernel for the final linear layer.

        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param num_layers: Number of recurrent layers
        :param output_size: The number of output features
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to `dropout`
        """
        super(ModelNew, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        
        # Replace nn.Linear with custom parameters for our kernel
        self.fc_weight = nn.Parameter(torch.empty(output_size, hidden_size))
        self.fc_bias = nn.Parameter(torch.empty(output_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Replicate the default initialization of nn.Linear
        nn.init.kaiming_uniform_(self.fc_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc_weight)
        if fan_in > 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.fc_bias, -bound, bound)
    
    def forward(self, x, h0, c0):
        """
        Forward pass through the LSTM model.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :return: The final cell state tensor, shape (num_layers, batch_size, hidden_size)
        """
        
        # Forward propagate LSTM
        out, state = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Call the custom fused kernel for slice + linear layer
        # The result is discarded to match the original model's behavior
        _ = fused_op.fused_slice_linear_forward(out, self.fc_weight, self.fc_bias)
        
        return state[1]