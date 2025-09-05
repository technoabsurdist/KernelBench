import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused LSTM cell computation
lstm_cell_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__global__ void lstm_cell_forward_kernel(
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
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_size) return;
    
    // Compute gates: [i, f, g, o] = x @ W_ih.T + h_prev @ W_hh.T + b_ih + b_hh
    float i = 0.0f, f = 0.0f, g = 0.0f, o = 0.0f;
    
    // Input gate
    for (int k = 0; k < input_size; ++k) {
        i += x[k] * w_ih[k * 4 * hidden_size + idx];
        f += x[k] * w_ih[k * 4 * hidden_size + hidden_size + idx];
        g += x[k] * w_ih[k * 4 * hidden_size + 2 * hidden_size + idx];
        o += x[k] * w_ih[k * 4 * hidden_size + 3 * hidden_size + idx];
    }
    
    for (int k = 0; k < hidden_size; ++k) {
        i += h_prev[k] * w_hh[k * 4 * hidden_size + idx];
        f += h_prev[k] * w_hh[k * 4 * hidden_size + hidden_size + idx];
        g += h_prev[k] * w_hh[k * 4 * hidden_size + 2 * hidden_size + idx];
        o += h_prev[k] * w_hh[k * 4 * hidden_size + 3 * hidden_size + idx];
    }
    
    i += b_ih[idx] + b_hh[idx];
    f += b_ih[hidden_size + idx] + b_hh[hidden_size + idx];
    g += b_ih[2 * hidden_size + idx] + b_hh[2 * hidden_size + idx];
    o += b_ih[3 * hidden_size + idx] + b_hh[3 * hidden_size + idx];
    
    // Apply activations
    i = 1.0f / (1.0f + expf(-i)); // sigmoid
    f = 1.0f / (1.0f + expf(-f)); // sigmoid
    g = tanhf(g);                 // tanh
    o = 1.0f / (1.0f + expf(-o)); // sigmoid
    
    // Cell state
    c_next[idx] = f * c_prev[idx] + i * g;
    
    // Hidden state
    h_next[idx] = o * tanhf(c_next[idx]);
}

torch::Tensor fused_lstm_cell_forward(
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
    
    auto h_next = torch::zeros({batch_size, hidden_size}, x.options());
    auto c_next = torch::zeros({batch_size, hidden_size}, x.options());
    
    const int block_size = 256;
    const int num_blocks = (hidden_size + block_size - 1) / block_size;
    
    for (int b = 0; b < batch_size; ++b) {
        lstm_cell_forward_kernel<<<num_blocks, block_size>>>(
            x[b].data_ptr<float>(),
            h_prev[b].data_ptr<float>(),
            c_prev[b].data_ptr<float>(),
            w_ih.data_ptr<float>(),
            w_hh.data_ptr<float>(),
            b_ih.data_ptr<float>(),
            b_hh.data_ptr<float>(),
            h_next[b].data_ptr<float>(),
            c_next[b].data_ptr<float>(),
            input_size,
            hidden_size
        );
    }
    
    return torch::stack({h_next, c_next}, 0);
}
"""

lstm_cell_cpp_source = """
torch::Tensor fused_lstm_cell_forward(
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
    functions=["fused_lstm_cell_forward"],
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
        self.hidden_size = hidden_size
        self.num_layers = num_layers
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
            
            self.lstm_weights.append(nn.ParameterList([w_ih, w_hh, b_ih, b_hh]))
        
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Register custom CUDA function
        self.fused_lstm_cell = fused_lstm_cell

    def forward(self, x, h0=None, c0=None):
        """
        Forward pass through the optimized LSTM model.
        """
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        if h0 is None:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=x.device)
        if c0 is None:
            c0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # Initialize hidden and cell states
        h = h0.clone()
        c = c0.clone()
        
        # Process sequence
        for t in range(seq_length):
            x_t = x[:, t, :]  # (batch_size, input_size)
            
            # Process through layers
            for layer in range(self.num_layers):
                layer_input = x_t if layer == 0 else h[layer]
                
                # Get layer parameters
                w_ih, w_hh, b_ih, b_hh = self.lstm_weights[layer]
                
                # Apply fused LSTM cell
                hc_next = self.fused_lstm_cell.fused_lstm_cell_forward(
                    layer_input, h[layer], c[layer], w_ih, w_hh, b_ih, b_hh
                )
                
                h[layer] = hc_next[0]
                c[layer] = hc_next[1]
                
                # Apply dropout if not last layer
                if layer < self.num_layers - 1 and self.dropout > 0:
                    h[layer] = torch.dropout(h[layer], self.dropout, self.training)
                
                x_t = h[layer]
        
        # Apply final linear layer to last time step
        out = self.fc(h[-1])
        
        return out