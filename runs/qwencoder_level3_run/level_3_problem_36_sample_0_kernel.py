import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused LSTM cell computation
lstm_cell_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void lstm_cell_kernel(
    const float* x, 
    const float* h_prev, 
    const float* c_prev,
    const float* w_ih,
    const float* w_hh,
    const float* b_ih,
    const float* b_hh,
    float* h_next,
    float* c_next,
    int input_size,
    int hidden_size,
    int batch_size
) {
    int batch_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || hidden_idx >= hidden_size) return;
    
    // Compute gates: i, f, g, o
    float i = 0.0f, f = 0.0f, g = 0.0f, o = 0.0f;
    
    // Input to hidden connections
    for (int k = 0; k < input_size; k++) {
        float x_val = x[batch_idx * input_size + k];
        i += x_val * w_ih[k * hidden_size * 4 + hidden_idx];
        f += x_val * w_ih[k * hidden_size * 4 + hidden_size + hidden_idx];
        g += x_val * w_ih[k * hidden_size * 4 + 2 * hidden_size + hidden_idx];
        o += x_val * w_ih[k * hidden_size * 4 + 3 * hidden_size + hidden_idx];
    }
    
    // Hidden to hidden connections
    for (int k = 0; k < hidden_size; k++) {
        float h_val = h_prev[batch_idx * hidden_size + k];
        i += h_val * w_hh[k * hidden_size * 4 + hidden_idx];
        f += h_val * w_hh[k * hidden_size * 4 + hidden_size + hidden_idx];
        g += h_val * w_hh[k * hidden_size * 4 + 2 * hidden_size + hidden_idx];
        o += h_val * w_hh[k * hidden_size * 4 + 3 * hidden_size + hidden_idx];
    }
    
    // Add biases
    i += b_ih[hidden_idx] + b_hh[hidden_idx];
    f += b_ih[hidden_size + hidden_idx] + b_hh[hidden_size + hidden_idx];
    g += b_ih[2 * hidden_size + hidden_idx] + b_hh[2 * hidden_size + hidden_idx];
    o += b_ih[3 * hidden_size + hidden_idx] + b_hh[3 * hidden_size + hidden_idx];
    
    // Apply activations
    i = 1.0f / (1.0f + expf(-i));  // sigmoid
    f = 1.0f / (1.0f + expf(-f));  // sigmoid
    g = tanhf(g);                   // tanh
    o = 1.0f / (1.0f + expf(-o));  // sigmoid
    
    // Compute cell state
    float c_prev_val = c_prev[batch_idx * hidden_size + hidden_idx];
    float c_next_val = f * c_prev_val + i * g;
    
    // Compute hidden state
    float h_next_val = o * tanhf(c_next_val);
    
    // Write outputs
    c_next[batch_idx * hidden_size + hidden_idx] = c_next_val;
    h_next[batch_idx * hidden_size + hidden_idx] = h_next_val;
}

torch::Tensor fused_lstm_cell_cuda(
    torch::Tensor x,
    torch::Tensor h_prev,
    torch::Tensor c_prev,
    torch::Tensor w_ih,
    torch::Tensor w_hh,
    torch::Tensor b_ih,
    torch::Tensor b_hh
) {
    auto batch_size = x.size(0);
    auto input_size = x.size(1);
    auto hidden_size = h_prev.size(1);
    
    auto h_next = torch::zeros_like(h_prev);
    auto c_next = torch::zeros_like(c_prev);
    
    dim3 grid(batch_size);
    dim3 block(hidden_size);
    
    lstm_cell_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        h_prev.data_ptr<float>(),
        c_prev.data_ptr<float>(),
        w_ih.data_ptr<float>(),
        w_hh.data_ptr<float>(),
        b_ih.data_ptr<float>(),
        b_hh.data_ptr<float>(),
        h_next.data_ptr<float>(),
        c_next.data_ptr<float>(),
        input_size,
        hidden_size,
        batch_size
    );
    
    return h_next;
}
"""

lstm_cell_cpp_source = """
torch::Tensor fused_lstm_cell_cuda(
    torch::Tensor x,
    torch::Tensor h_prev,
    torch::Tensor c_prev,
    torch::Tensor w_ih,
    torch::Tensor w_hh,
    torch::Tensor b_ih,
    torch::Tensor b_hh
);
"""

# Compile the inline CUDA code for fused LSTM cell
fused_lstm_cell = load_inline(
    name="fused_lstm_cell",
    cpp_sources=lstm_cell_cpp_source,
    cuda_sources=lstm_cell_source,
    functions=["fused_lstm_cell_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Optimized LSTM model with custom CUDA kernels.
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        
        # Create parameters for each layer
        self.lstm_weights = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            
            # Weight matrices
            w_ih = nn.Parameter(torch.randn(4 * hidden_size, layer_input_size))
            w_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
            
            # Bias vectors
            b_ih = nn.Parameter(torch.randn(4 * hidden_size))
            b_hh = nn.Parameter(torch.randn(4 * hidden_size))
            
            # Initialize parameters
            nn.init.xavier_uniform_(w_ih)
            nn.init.xavier_uniform_(w_hh)
            nn.init.zeros_(b_ih)
            nn.init.zeros_(b_hh)
            
            self.lstm_weights.append(nn.ParameterList([
                w_ih, w_hh, b_ih, b_hh
            ]))
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else None
        
        # Register custom CUDA function
        self.fused_lstm_cell = fused_lstm_cell
    
    def forward(self, x, h0, c0):
        """
        Forward pass through the optimized LSTM model.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :param h0: Initial hidden state, shape (num_layers, batch_size, hidden_size)
        :param c0: Initial cell state, shape (num_layers, batch_size, hidden_size)
        :return: The output tensor, shape (batch_size, sequence_length, output_size)
        """
        batch_size, seq_length, _ = x.shape
        
        # Initialize hidden and cell states
        h = h0.clone()
        c = c0.clone()
        
        # Process each time step
        for t in range(seq_length):
            x_t = x[:, t, :]  # Shape: (batch_size, input_size)
            
            # Process through each LSTM layer
            for layer in range(self.num_layers):
                layer_input = x_t if layer == 0 else h[layer-1]
                
                # Get layer parameters
                w_ih, w_hh, b_ih, b_hh = self.lstm_weights[layer]
                
                # Apply fused LSTM cell computation
                h[layer] = self.fused_lstm_cell.fused_lstm_cell_cuda(
                    layer_input, h[layer], c[layer], w_ih, w_hh, b_ih, b_hh
                )
                
                # Apply dropout if specified (except for last layer)
                if self.dropout_layer and layer < self.num_layers - 1:
                    h[layer] = self.dropout_layer(h[layer])
                
                # Update input for next layer
                x_t = h[layer]
        
        # Use the last time step's output from the last layer
        out = self.fc(h[-1])  # Shape: (batch_size, output_size)
        
        return h  # Return final hidden states