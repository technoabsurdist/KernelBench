import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused slicing and linear layer
fused_slice_linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused kernel to perform:
// 1. Slicing: out[:, -1, :]
// 2. Linear: matmul(slice, weight.T) + bias
__global__ void fused_slice_linear_kernel(
    const float* lstm_out,
    const float* weight,
    const float* bias,
    float* final_out,
    int B, // batch_size
    int S, // sequence_length
    int I, // input_features (hidden_size * 2)
    int O  // output_features
) {
    // Each thread computes one element of the output matrix (B x O)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // output feature index

    if (row < B && col < O) {
        // Pointer to the start of the relevant slice from lstm_out.
        // This corresponds to the last time step (S-1) for the current batch item (row).
        const float* x_slice = lstm_out + row * S * I + (S - 1) * I;

        // Pointer to the relevant row of the weight matrix.
        // This corresponds to the weights for the current output feature (col).
        const float* w_row = weight + col * I;

        // Compute dot product between the input slice and the weight row
        float sum = 0.0f;
        for (int k = 0; k < I; ++k) {
            sum += x_slice[k] * w_row[k];
        }

        // Add bias and store the result
        final_out[row * O + col] = sum + bias[col];
    }
}

torch::Tensor fused_slice_linear_cuda(
    torch::Tensor lstm_out, // (B, S, I)
    torch::Tensor weight,   // (O, I)
    torch::Tensor bias      // (O)
) {
    // Input validation
    TORCH_CHECK(lstm_out.is_cuda(), "lstm_out must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(lstm_out.scalar_type() == torch::kFloat32, "lstm_out must be a float32 tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be a float32 tensor");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be a float32 tensor");
    TORCH_CHECK(lstm_out.dim() == 3, "lstm_out must be 3-dimensional");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2-dimensional");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1-dimensional");
    TORCH_CHECK(lstm_out.size(2) == weight.size(1), "Dimension mismatch: lstm_out.size(2) != weight.size(1)");
    TORCH_CHECK(weight.size(0) == bias.size(0), "Dimension mismatch: weight.size(0) != bias.size(0)");

    const int B = lstm_out.size(0);
    const int S = lstm_out.size(1);
    const int I = lstm_out.size(2);
    const int O = weight.size(0);

    // Create the output tensor
    auto final_out = torch::empty({B, O}, lstm_out.options());

    // Define grid and block dimensions for a 2D launch
    const dim3 block_size(16, 16);
    const dim3 num_blocks(
        (O + block_size.x - 1) / block_size.x,
        (B + block_size.y - 1) / block_size.y
    );

    // Launch the kernel
    fused_slice_linear_kernel<<<num_blocks, block_size>>>(
        lstm_out.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        final_out.data_ptr<float>(),
        B, S, I, O
    );

    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return final_out;
}
"""

fused_slice_linear_cpp_source = """
torch::Tensor fused_slice_linear_cuda(torch::Tensor lstm_out, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
# This fuses the operation of taking the last time step output and applying a linear layer
fused_slice_linear = load_inline(
    name="fused_slice_linear",
    cpp_sources=fused_slice_linear_cpp_source,
    cuda_sources=fused_slice_linear_source,
    functions=["fused_slice_linear_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model with a custom fused CUDA kernel for the final layer.

        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param num_layers: Number of recurrent layers
        :param output_size: The number of output features
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
        """
        super(ModelNew, self).__init__()
        # The LSTM layer is kept as is, since it's already highly optimized (cuDNN)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        
        # We replace the nn.Linear layer with our custom fused operation.
        # We need to manually create and initialize the weight and bias parameters.
        fc_input_features = hidden_size * 2
        self.fc_weight = nn.Parameter(torch.empty(output_size, fc_input_features))
        self.fc_bias = nn.Parameter(torch.empty(output_size))
        
        # Initialize parameters to match the default nn.Linear initialization
        self.reset_parameters()

        # Store the compiled CUDA function
        self.fused_op = fused_slice_linear

    def reset_parameters(self) -> None:
        # Mimic the initialization of nn.Linear to ensure comparable behavior
        nn.init.kaiming_uniform_(self.fc_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc_weight)
        if fan_in > 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.fc_bias, -bound, bound)

    def forward(self, x, h0, c0):
        """
        Forward pass through the LSTM model using the custom CUDA kernel.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size * 2)
        out, hn = self.lstm(x, (h0, c0))
        
        # Use the custom fused kernel to slice the last time step and apply the linear transformation
        # This avoids creating an intermediate tensor from slicing and reduces kernel launch overhead.
        out = self.fused_op.fused_slice_linear_cuda(out, self.fc_weight, self.fc_bias)
        
        return out