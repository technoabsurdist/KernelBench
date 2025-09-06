import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused slicing and linear layer
slice_linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void slice_and_linear_kernel(
    const float* lstm_out, const float* weight, const float* bias, float* output,
    int B, int S, int H, int O) {

    // This kernel computes the operation: output = lstm_out[:, -1, :] @ weight.T + bias
    // It fuses the slicing of the last time step from the LSTM output with the linear transformation.
    // It uses a tiled matrix multiplication approach for performance.

    // Shared memory for tiles of A (input_slice) and B (weight.T)
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row and column of the output matrix C to compute
    int row = blockIdx.y * TILE_DIM + ty; // Corresponds to batch index
    int col = blockIdx.x * TILE_DIM + tx; // Corresponds to output feature index

    float C_value = 0.0f;

    // Loop over the tiles of A and B required to compute the C_sub tile
    for (int i = 0; i < (H + TILE_DIM - 1) / TILE_DIM; ++i) {
        // Load a tile of A (input_slice) into shared memory
        // A has shape (B, H)
        int a_col = i * TILE_DIM + tx;
        if (row < B && a_col < H) {
            // A[row, a_col] is effectively lstm_out[row, S-1, a_col]
            s_A[ty][tx] = lstm_out[row * S * H + (S - 1) * H + a_col];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        // Load a tile of B (which is weight.T) into shared memory
        // B has shape (H, O). B[b_row, col] = weight.T[b_row, col] = weight[col, b_row]
        // weight has shape (O, H)
        int b_row = i * TILE_DIM + ty;
        if (col < O && b_row < H) {
            // B[b_row, col] = weight[col, b_row]
            s_B[ty][tx] = weight[col * H + b_row];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Multiply the two tiles from shared memory
        for (int k = 0; k < TILE_DIM; ++k) {
            C_value += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    // Write the final result to C (output) after adding the bias
    if (row < B && col < O) {
        output[row * O + col] = C_value + bias[col];
    }
}

torch::Tensor slice_and_linear_cuda(
    torch::Tensor lstm_out,
    torch::Tensor weight,
    torch::Tensor bias) {

    // Input validation
    TORCH_CHECK(lstm_out.is_cuda(), "lstm_out must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(lstm_out.dim() == 3, "lstm_out must be a 3D tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");

    const int B = lstm_out.size(0);
    const int S = lstm_out.size(1);
    const int H = lstm_out.size(2);
    const int O = weight.size(0);

    TORCH_CHECK(weight.size(1) == H, "weight and lstm_out have incompatible shapes");
    TORCH_CHECK(bias.size(0) == O, "bias and weight have incompatible shapes");

    // Create output tensor
    auto output = torch::zeros({B, O}, lstm_out.options());

    // Grid and block configuration
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks(
        (O + TILE_DIM - 1) / TILE_DIM,
        (B + TILE_DIM - 1) / TILE_DIM
    );

    // Launch kernel
    slice_and_linear_kernel<<<numBlocks, threadsPerBlock>>>(
        lstm_out.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, S, H, O
    );

    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

# C++ source for function signature
slice_linear_cpp_source = "torch::Tensor slice_and_linear_cuda(torch::Tensor lstm_out, torch::Tensor weight, torch::Tensor bias);"

# Compile the inline CUDA code
slice_linear = load_inline(
    name="slice_linear",
    cpp_sources=slice_linear_cpp_source,
    cuda_sources=slice_linear_source,
    functions=["slice_and_linear_cuda"],
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
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
        """
        super(ModelNew, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # The nn.LSTM layer is kept as is, since it's highly optimized with cuDNN.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=False)
        
        # We create the nn.Linear layer primarily to hold and manage the weight and bias parameters.
        # This ensures they are properly initialized, registered, and moved to the correct device.
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Store the custom fused CUDA function
        self.slice_linear_op = slice_linear.slice_and_linear_cuda

    def forward(self, x, h0=None, c0=None):
        """
        Forward pass through the LSTM model using the custom kernel.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :param h0: Optional initial hidden state (num_layers, batch_size, hidden_size)
        :param c0: Optional initial cell state (num_layers, batch_size, hidden_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        batch_size = x.size(0)

        if h0 is None:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=x.device)
        if c0 is None:
            c0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=x.device)

        # out: (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Instead of slicing and then calling self.fc, we call our custom kernel.
        # This kernel fuses the slicing (out[:, -1, :]) and the linear operation.
        # We pass the parameters from the self.fc layer to our custom function.
        out = self.slice_linear_op(out, self.fc.weight, self.fc.bias)

        return out