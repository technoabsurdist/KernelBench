import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for the element-wise operations in a GRU cell
gru_cell_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Device function for sigmoid
__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void gru_cell_forward_kernel(
    const float* __restrict__ gates_ih,
    const float* __restrict__ gates_hh,
    const float* __restrict__ h_prev,
    float* __restrict__ h_next,
    const int batch_size,
    const int hidden_size) {

    // Each thread computes one element in the (batch, hidden) output matrix
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batch_size && col < hidden_size) {
        // Index for the current (batch, hidden) element
        const int idx = row * hidden_size + col;
        
        // Pointers to the start of the current batch item's gate values
        const float* gates_ih_row = gates_ih + row * 3 * hidden_size;
        const float* gates_hh_row = gates_hh + row * 3 * hidden_size;

        // Calculate reset gate (r)
        const float r_linear = gates_ih_row[col] + gates_hh_row[col];
        const float r = sigmoidf(r_linear);

        // Calculate update gate (z)
        const float z_linear = gates_ih_row[col + hidden_size] + gates_hh_row[col + hidden_size];
        const float z = sigmoidf(z_linear);

        // Calculate new gate (n)
        const float n_linear_ih = gates_ih_row[col + 2 * hidden_size];
        const float n_linear_hh = gates_hh_row[col + 2 * hidden_size];
        const float n = tanhf(n_linear_ih + r * n_linear_hh);

        // Calculate next hidden state
        h_next[idx] = (1.0f - z) * n + z * h_prev[idx];
    }
}

torch::Tensor gru_cell_forward_cuda(
    torch::Tensor gates_ih,
    torch::Tensor gates_hh,
    torch::Tensor h_prev) {

    TORCH_CHECK(gates_ih.is_cuda(), "gates_ih must be a CUDA tensor");
    TORCH_CHECK(gates_hh.is_cuda(), "gates_hh must be a CUDA tensor");
    TORCH_CHECK(h_prev.is_cuda(), "h_prev must be a CUDA tensor");

    TORCH_CHECK(gates_ih.is_contiguous(), "gates_ih must be contiguous");
    TORCH_CHECK(gates_hh.is_contiguous(), "gates_hh must be contiguous");
    TORCH_CHECK(h_prev.is_contiguous(), "h_prev must be contiguous");

    const auto batch_size = h_prev.size(0);
    const auto hidden_size = h_prev.size(1);

    auto h_next = torch::empty_like(h_prev);

    const dim3 threads(16, 16);
    const dim3 blocks(
        (hidden_size + threads.x - 1) / threads.x,
        (batch_size + threads.y - 1) / threads.y
    );

    gru_cell_forward_kernel<<<blocks, threads>>>(
        gates_ih.data_ptr<float>(),
        gates_hh.data_ptr<float>(),
        h_prev.data_ptr<float>(),
        h_next.data_ptr<float>(),
        batch_size,
        hidden_size
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return h_next;
}
"""

gru_cell_cpp_source = """
torch::Tensor gru_cell_forward_cuda(torch::Tensor gates_ih, torch::Tensor gates_hh, torch::Tensor h_prev);
"""

# JIT compile the custom CUDA kernel
gru_cell_op = load_inline(
    name="gru_cell_op",
    cpp_sources=gru_cell_cpp_source,
    cuda_sources=gru_cell_source,
    functions=["gru_cell_forward_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        Custom GRU implementation using a fused CUDA kernel for the cell's element-wise operations.
        The matrix multiplications are handled by torch.nn.functional.linear to leverage cuBLAS.
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        # Create parameters for each layer
        self.weights_ih = nn.ParameterList()
        self.weights_hh = nn.ParameterList()
        if bias:
            self.biases_ih = nn.ParameterList()
            self.biases_hh = nn.ParameterList()
        else:
            self.register_parameter('biases_ih', None)
            self.register_parameter('biases_hh', None)

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            
            # Input-to-hidden weights
            w_ih = nn.Parameter(torch.Tensor(3 * hidden_size, layer_input_size))
            self.weights_ih.append(w_ih)

            # Hidden-to-hidden weights
            w_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
            self.weights_hh.append(w_hh)

            if bias:
                b_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
                b_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
                self.biases_ih.append(b_ih)
                self.biases_hh.append(b_hh)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, h0):
        """
        :param x: The input tensor, shape (seq_len, batch_size, input_size) if batch_first=False, otherwise (batch_size, seq_len, input_size)
        :param h0: The initial hidden state for the input sequence, shape (num_layers, batch_size, hidden_size)
        :return: h_n
            - h_n: The hidden state for t = seq_len, shape (num_layers, batch_size, hidden_size)
        """
        if self.batch_first:
            # (batch, seq, feature) -> (seq, batch, feature)
            x = x.transpose(0, 1)

        seq_len = x.shape[0]
        
        # h0 is (num_layers, batch_size, hidden_size)
        hidden_states = list(torch.unbind(h0, dim=0))
        
        current_input = x
        final_hidden_states = []

        for i in range(self.num_layers):
            h_layer = hidden_states[i]
            layer_output_seq = []
            
            # Get weights for the current layer
            w_ih = self.weights_ih[i]
            w_hh = self.weights_hh[i]
            b_ih = self.biases_ih[i] if self.bias else None
            b_hh = self.biases_hh[i] if self.bias else None

            for t in range(seq_len):
                x_t = current_input[t]
                
                # Use F.linear for optimized GEMM (matrix multiplication)
                gates_ih = F.linear(x_t, w_ih, b_ih)
                gates_hh = F.linear(h_layer, w_hh, b_hh)
                
                # Call the custom fused kernel for element-wise operations
                h_layer = gru_cell_op.gru_cell_forward_cuda(gates_ih, gates_hh, h_layer)
                
                layer_output_seq.append(h_layer)
            
            # Stack outputs of this layer to be input for the next
            current_input = torch.stack(layer_output_seq, dim=0)
            final_hidden_states.append(h_layer)
            
        h_n = torch.stack(final_hidden_states, dim=0)
        
        return h_n