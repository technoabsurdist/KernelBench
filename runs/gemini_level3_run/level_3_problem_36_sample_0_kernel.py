import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for a fused LSTM cell operation
# This kernel combines the gate activations (sigmoid, tanh) and the cell/hidden state updates
# into a single operation to reduce kernel launch overhead and memory bandwidth.
fused_lstm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Device function for sigmoid, __forceinline__ suggests the compiler to inline it
__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// The fused LSTM cell kernel
__global__ void fused_lstm_cell_kernel(
    const float* i_linear, const float* f_linear, const float* g_linear, const float* o_linear,
    const float* c_prev,
    float* h_next, float* c_next,
    int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Apply activations to the linear transformations of gates
        float i = sigmoidf(i_linear[idx]);
        float f = sigmoidf(f_linear[idx]);
        float g = tanhf(g_linear[idx]);
        float o = sigmoidf(o_linear[idx]);

        // Compute the next cell state
        // c_next = (f * c_prev) + (i * g)
        float c_n = f * c_prev[idx] + i * g;
        c_next[idx] = c_n;

        // Compute the next hidden state
        // h_next = o * tanh(c_next)
        h_next[idx] = o * tanhf(c_n);
    }
}

// C++ wrapper function to be called from Python
std::vector<torch::Tensor> fused_lstm_cell_cuda(
    torch::Tensor i_linear,
    torch::Tensor f_linear,
    torch::Tensor g_linear,
    torch::Tensor o_linear,
    torch::Tensor c_prev) {

    // Input validation
    TORCH_CHECK(i_linear.is_cuda(), "i_linear must be a CUDA tensor");
    TORCH_CHECK(f_linear.is_cuda(), "f_linear must be a CUDA tensor");
    TORCH_CHECK(g_linear.is_cuda(), "g_linear must be a CUDA tensor");
    TORCH_CHECK(o_linear.is_cuda(), "o_linear must be a CUDA tensor");
    TORCH_CHECK(c_prev.is_cuda(), "c_prev must be a CUDA tensor");

    TORCH_CHECK(i_linear.is_contiguous(), "i_linear must be contiguous");
    TORCH_CHECK(f_linear.is_contiguous(), "f_linear must be contiguous");
    TORCH_CHECK(g_linear.is_contiguous(), "g_linear must be contiguous");
    TORCH_CHECK(o_linear.is_contiguous(), "o_linear must be contiguous");
    TORCH_CHECK(c_prev.is_contiguous(), "c_prev must be contiguous");
    
    TORCH_CHECK(i_linear.sizes() == f_linear.sizes() &&
                i_linear.sizes() == g_linear.sizes() &&
                i_linear.sizes() == o_linear.sizes() &&
                i_linear.sizes() == c_prev.sizes(),
                "All input tensors must have the same size");

    const auto size = i_linear.numel();

    // Create output tensors
    auto h_next = torch::empty_like(i_linear);
    auto c_next = torch::empty_like(i_linear);

    // Kernel launch configuration
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Launch the kernel
    fused_lstm_cell_kernel<<<num_blocks, block_size>>>(
        i_linear.data_ptr<float>(),
        f_linear.data_ptr<float>(),
        g_linear.data_ptr<float>(),
        o_linear.data_ptr<float>(),
        c_prev.data_ptr<float>(),
        h_next.data_ptr<float>(),
        c_next.data_ptr<float>(),
        size
    );
    
    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return {h_next, c_next};
}
"""

# C++ source for the function signature, required by load_inline
fused_lstm_cpp_source = """
std::vector<torch::Tensor> fused_lstm_cell_cuda(
    torch::Tensor i_linear,
    torch::Tensor f_linear,
    torch::Tensor g_linear,
    torch::Tensor o_linear,
    torch::Tensor c_prev);
"""

# JIT compile the CUDA and C++ code
fused_lstm_cell = load_inline(
    name="fused_lstm_cell",
    cpp_sources=fused_lstm_cpp_source,
    cuda_sources=fused_lstm_source,
    functions=["fused_lstm_cell_cuda"],
    verbose=False,
)


class CustomLSTMLayer(nn.Module):
    """
    A single-layer LSTM implementation that replaces the standard PyTorch cell
    with our custom fused CUDA kernel. This is used as a building block in ModelNew.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTM weights and biases, matching the structure of nn.LSTM
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights and biases similar to nn.LSTM
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input_seq, state):
        h_prev, c_prev = state
        batch_size, seq_len, _ = input_seq.shape
        
        outputs = []

        # Pre-calculate the input-to-hidden transformation for the whole sequence.
        # This is more efficient than doing it inside the loop.
        x_gates_seq = torch.mm(input_seq.reshape(batch_size * seq_len, -1), self.weight_ih.t())
        x_gates_seq = x_gates_seq.view(batch_size, seq_len, -1)

        # Loop over the time steps
        for t in range(seq_len):
            # Hidden-to-hidden transformation
            h_gates = torch.mm(h_prev, self.weight_hh.t())
            
            # Combine with pre-calculated x_gates and add biases
            gates = x_gates_seq[:, t, :] + h_gates + self.bias_ih + self.bias_hh
            
            # Split the combined gates into four parts for the fused kernel
            i_linear, f_linear, g_linear, o_linear = gates.chunk(4, 1)

            # Call the custom fused CUDA kernel
            h_next, c_next = fused_lstm_cell.fused_lstm_cell_cuda(
                i_linear, f_linear, g_linear, o_linear, c_prev
            )

            outputs.append(h_next.unsqueeze(1))
            h_prev, c_prev = h_next, c_next

        output_seq = torch.cat(outputs, dim=1)
        return output_seq, (h_prev, c_prev)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model with custom CUDA kernels.
        This model replaces nn.LSTM with a stack of CustomLSTMLayer modules,
        each using a fused kernel for the element-wise operations within the LSTM cell.
        """
        super(ModelNew, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        # First layer's input size is the model's input_size
        self.layers.append(CustomLSTMLayer(input_size, hidden_size))
        # Subsequent layers' input size is the hidden_size
        for _ in range(num_layers - 1):
            self.layers.append(CustomLSTMLayer(hidden_size, hidden_size))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 and num_layers > 1 else None
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0, c0):
        """
        Forward pass through the custom LSTM model.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :param h0: Initial hidden state, shape (num_layers, batch_size, hidden_size)
        :param c0: Initial cell state, shape (num_layers, batch_size, hidden_size)
        :return: The final hidden state tensor, shape (num_layers, batch_size, hidden_size)
        """
        current_input = x
        next_h_states = []
        next_c_states = []

        for i, layer in enumerate(self.layers):
            # Get initial state for the current layer
            h_i = h0[i, :, :]
            c_i = c0[i, :, :]
            
            # Forward pass through the custom LSTM layer
            output_seq, (h_n, c_n) = layer(current_input, (h_i, c_i))
            
            # Apply dropout to the output of all but the last layer
            if self.dropout is not None and i < self.num_layers - 1:
                output_seq = self.dropout(output_seq)

            # The output of this layer is the input to the next
            current_input = output_seq
            
            # Store the final states of this layer
            next_h_states.append(h_n)
            next_c_states.append(c_n)

        # The final output of the LSTM stack is the output_seq from the last layer
        out = output_seq
        
        # Stack the final hidden states from all layers
        final_h = torch.stack(next_h_states, dim=0)
        
        # Replicate the original model's logic:
        # The fc layer is computed on the last time step's output, but its result is discarded.
        _ = self.fc(out[:, -1, :])
        
        # Return the final hidden state, matching the original model's return value.
        return final_h