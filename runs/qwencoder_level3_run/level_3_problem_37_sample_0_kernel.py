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
        printf("CUDA error at %s %d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

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
    
    // Shared memory for intermediate computations
    extern __shared__ float shared_mem[];
    float* ifgo = &shared_mem[0]; // 4 * hidden_size per block
    
    // Compute gates for this batch item
    float ig = 0.0f, fg = 0.0f, gg = 0.0f, og = 0.0f;
    
    // Compute input gate
    for (int i = 0; i < input_size; i++) {
        ig += x[batch_idx * input_size + i] * w_ih[0 * hidden_size * input_size + hidden_idx * input_size + i];
        fg += x[batch_idx * input_size + i] * w_ih[1 * hidden_size * input_size + hidden_idx * input_size + i];
        gg += x[batch_idx * input_size + i] * w_ih[2 * hidden_size * input_size + hidden_idx * input_size + i];
        og += x[batch_idx * input_size + i] * w_ih[3 * hidden_size * input_size + hidden_idx * input_size + i];
    }
    
    // Compute recurrent connections
    for (int i = 0; i < hidden_size; i++) {
        ig += h_prev[batch_idx * hidden_size + i] * w_hh[0 * hidden_size * hidden_size + hidden_idx * hidden_size + i];
        fg += h_prev[batch_idx * hidden_size + i] * w_hh[1 * hidden_size * hidden_size + hidden_idx * hidden_size + i];
        gg += h_prev[batch_idx * hidden_size + i] * w_hh[2 * hidden_size * hidden_size + hidden_idx * hidden_size + i];
        og += h_prev[batch_idx * hidden_size + i] * w_hh[3 * hidden_size * hidden_size + hidden_idx * hidden_size + i];
    }
    
    // Add biases
    ig += b_ih[0 * hidden_size + hidden_idx] + b_hh[0 * hidden_size + hidden_idx];
    fg += b_ih[1 * hidden_size + hidden_idx] + b_hh[1 * hidden_size + hidden_idx];
    gg += b_ih[2 * hidden_size + hidden_idx] + b_hh[2 * hidden_size + hidden_idx];
    og += b_ih[3 * hidden_size + hidden_idx] + b_hh[3 * hidden_size + hidden_idx];
    
    // Store in shared memory
    ifgo[0 * hidden_size + hidden_idx] = ig;
    ifgo[1 * hidden_size + hidden_idx] = fg;
    ifgo[2 * hidden_size + hidden_idx] = gg;
    ifgo[3 * hidden_size + hidden_idx] = og;
    
    __syncthreads();
    
    // Apply activations
    float i_gate = 1.0f / (1.0f + expf(-ifgo[0 * hidden_size + hidden_idx]));
    float f_gate = 1.0f / (1.0f + expf(-ifgo[1 * hidden_size + hidden_idx]));
    float g_gate = tanhf(ifgo[2 * hidden_size + hidden_idx]);
    float o_gate = 1.0f / (1.0f + expf(-ifgo[3 * hidden_size + hidden_idx]));
    
    // Compute cell state
    float c_prev_val = c_prev[batch_idx * hidden_size + hidden_idx];
    float c_next_val = f_gate * c_prev_val + i_gate * g_gate;
    c_next[batch_idx * hidden_size + hidden_idx] = c_next_val;
    
    // Compute hidden state
    float h_next_val = o_gate * tanhf(c_next_val);
    h_next[batch_idx * hidden_size + hidden_idx] = h_next_val;
}

torch::Tensor fused_lstm_cell(
    torch::Tensor x,
    torch::Tensor h_prev,
    torch::Tensor c_prev,
    torch::Tensor w_ih,
    torch::Tensor w_hh,
    torch::Tensor b_ih,
    torch::Tensor b_hh
) {
    int batch_size = x.size(0);
    int input_size = x.size(1);
    int hidden_size = h_prev.size(1);
    
    auto h_next = torch::zeros_like(h_prev);
    auto c_next = torch::zeros_like(c_prev);
    
    const int threads_per_block = hidden_size;
    const int blocks_per_grid = batch_size;
    const int shared_mem_size = 4 * hidden_size * sizeof(float);
    
    lstm_cell_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
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
    
    CUDA_CHECK(cudaGetLastError());
    
    return torch::stack({h_next, c_next}, 0);
}
"""

lstm_cell_cpp_source = """
#include <torch/extension.h>

torch::Tensor fused_lstm_cell(
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
    functions=["fused_lstm_cell"],
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
            
            self.lstm_weights.append(nn.ParameterList([w_ih, w_hh, b_ih, b_hh]))
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else None
        
    def forward(self, x, h0, c0):
        """
        Forward pass through the optimized LSTM model.
        """
        batch_size, seq_length, _ = x.shape
        h_prev = h0
        c_prev = c0
        
        # Process through all layers
        for layer_idx in range(self.num_layers):
            w_ih, w_hh, b_ih, b_hh = self.lstm_weights[layer_idx]
            
            # Process sequence
            h_layer = []
            c_layer = []
            
            h_prev_layer = h_prev[layer_idx].unsqueeze(0).repeat(batch_size, 1)
            c_prev_layer = c_prev[layer_idx].unsqueeze(0).repeat(batch_size, 1)
            
            for t in range(seq_length):
                x_t = x[:, t, :] if layer_idx == 0 else h_layer[t]
                
                # Apply fused LSTM cell
                result = fused_lstm_cell.fused_lstm_cell(
                    x_t.contiguous(),
                    h_prev_layer.contiguous(),
                    c_prev_layer.contiguous(),
                    w_ih.contiguous(),
                    w_hh.contiguous(),
                    b_ih.contiguous(),
                    b_hh.contiguous()
                )
                
                h_prev_layer = result[0]
                c_prev_layer = result[1]
                
                h_layer.append(h_prev_layer)
                c_layer.append(c_prev_layer)
            
            # Stack outputs for next layer
            if layer_idx < self.num_layers - 1 and self.dropout_layer is not None:
                x = self.dropout_layer(torch.stack(h_layer, dim=1))
            else:
                x = torch.stack(h_layer, dim=1)
        
        # Return final cell state
        return c_prev_layer.unsqueeze(0).repeat(self.num_layers, 1, 1)