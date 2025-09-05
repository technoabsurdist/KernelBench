import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused LSTM cell computation
lstm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

__global__ void lstm_cell_forward_kernel(
    const float* input,
    const float* weight_ih,
    const float* weight_hh,
    const float* bias,
    const float* hx,
    const float* cx,
    float* hy,
    float* cy,
    int input_size,
    int hidden_size,
    int batch_size) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * hidden_size;
    
    if (tid < total_threads) {
        int batch_idx = tid / hidden_size;
        int hidden_idx = tid % hidden_size;
        
        // Compute input contributions (4 gates)
        float igate = 0.0f, fgate = 0.0f, ggate = 0.0f, ogate = 0.0f;
        
        // Input to hidden transformation
        for (int i = 0; i < input_size; i++) {
            float inp_val = input[batch_idx * input_size + i];
            igate += inp_val * weight_ih[i * 4 * hidden_size + hidden_idx];
            fgate += inp_val * weight_ih[i * 4 * hidden_size + hidden_size + hidden_idx];
            ggate += inp_val * weight_ih[i * 4 * hidden_size + 2 * hidden_size + hidden_idx];
            ogate += inp_val * weight_ih[i * 4 * hidden_size + 3 * hidden_size + hidden_idx];
        }
        
        // Hidden to hidden transformation
        for (int i = 0; i < hidden_size; i++) {
            float h_val = hx[batch_idx * hidden_size + i];
            igate += h_val * weight_hh[i * 4 * hidden_size + hidden_idx];
            fgate += h_val * weight_hh[i * 4 * hidden_size + hidden_size + hidden_idx];
            ggate += h_val * weight_hh[i * 4 * hidden_size + 2 * hidden_size + hidden_idx];
            ogate += h_val * weight_hh[i * 4 * hidden_size + 3 * hidden_size + hidden_idx];
        }
        
        // Add biases
        igate += bias[hidden_idx];
        fgate += bias[hidden_size + hidden_idx];
        ggate += bias[2 * hidden_size + hidden_idx];
        ogate += bias[3 * hidden_size + hidden_idx];
        
        // Apply activations
        float i_t = 1.0f / (1.0f + expf(-igate));  // sigmoid
        float f_t = 1.0f / (1.0f + expf(-fgate));  // sigmoid
        float g_t = tanhf(ggate);                  // tanh
        float o_t = 1.0f / (1.0f + expf(-ogate));  // sigmoid
        
        // Cell state update
        float c_prev = cx[batch_idx * hidden_size + hidden_idx];
        float c_t = f_t * c_prev + i_t * g_t;
        
        // Hidden state update
        float h_t = o_t * tanhf(c_t);
        
        // Write outputs
        hy[batch_idx * hidden_size + hidden_idx] = h_t;
        cy[batch_idx * hidden_size + hidden_idx] = c_t;
    }
}

torch::Tensor fused_lstm_cell_forward(
    torch::Tensor input,
    torch::Tensor weight_ih,
    torch::Tensor weight_hh,
    torch::Tensor bias,
    torch::Tensor hx,
    torch::Tensor cx) {
    
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto hidden_size = hx.size(1);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto hy = torch::zeros({batch_size, hidden_size}, options);
    auto cy = torch::zeros({batch_size, hidden_size}, options);
    
    const int block_size = 256;
    const int num_blocks = (batch_size * hidden_size + block_size - 1) / block_size;
    
    lstm_cell_forward_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight_ih.data_ptr<float>(),
        weight_hh.data_ptr<float>(),
        bias.data_ptr<float>(),
        hx.data_ptr<float>(),
        cx.data_ptr<float>(),
        hy.data_ptr<float>(),
        cy.data_ptr<float>(),
        input_size,
        hidden_size,
        batch_size
    );
    
    return torch::stack({hy, cy}, 0);
}
"""

lstm_cpp_source = """
torch::Tensor fused_lstm_cell_forward(
    torch::Tensor input,
    torch::Tensor weight_ih,
    torch::Tensor weight_hh,
    torch::Tensor bias,
    torch::Tensor hx,
    torch::Tensor cx);
"""

# Compile the inline CUDA code for LSTM
fused_lstm = load_inline(
    name="fused_lstm",
    cpp_sources=lstm_cpp_source,
    cuda_sources=lstm_source,
    functions=["fused_lstm_cell_forward"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class CustomLSTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights_ih, weights_hh, biases, h0, c0):
        # Save tensors for backward pass
        ctx.save_for_backward(input, weights_ih, weights_hh, biases, h0, c0)
        
        batch_size, seq_len, input_size = input.shape
        num_layers = len(weights_ih)
        hidden_size = h0.shape[2]
        
        # Initialize hidden and cell states
        h_prev = h0.clone()
        c_prev = c0.clone()
        
        # Output tensor to store all hidden states
        output = torch.zeros(batch_size, seq_len, hidden_size * 2, device=input.device)
        
        # Process each layer
        layer_input = input
        for layer in range(num_layers):
            # Process each time step
            layer_output = torch.zeros(batch_size, seq_len, hidden_size * 2, device=input.device)
            
            h_layer = h_prev[layer*2:(layer*2)+2]  # Forward and backward
            c_layer = c_prev[layer*2:(layer*2)+2]
            
            # Forward direction
            h_fwd = h_layer[0]
            c_fwd = c_layer[0]
            for t in range(seq_len):
                x_t = layer_input[:, t, :]
                result = fused_lstm.fused_lstm_cell_forward(
                    x_t.contiguous(),
                    weights_ih[layer][0],
                    weights_hh[layer][0],
                    biases[layer][0],
                    h_fwd.contiguous(),
                    c_fwd.contiguous()
                )
                h_fwd, c_fwd = result[0], result[1]
                layer_output[:, t, :hidden_size] = h_fwd
            
            # Backward direction
            h_bwd = h_layer[1]
            c_bwd = c_layer[1]
            for t in range(seq_len-1, -1, -1):
                x_t = layer_input[:, t, :]
                result = fused_lstm.fused_lstm_cell_forward(
                    x_t.contiguous(),
                    weights_ih[layer][1],
                    weights_hh[layer][1],
                    biases[layer][1],
                    h_bwd.contiguous(),
                    c_bwd.contiguous()
                )
                h_bwd, c_bwd = result[0], result[1]
                layer_output[:, t, hidden_size:] = h_bwd
            
            layer_input = layer_output
            output = layer_output
        
        return output, (h_prev, c_prev)

    @staticmethod
    def backward(ctx, grad_output, grad_hn):
        # Simplified backward - in practice, this would need full LSTM backward implementation
        input, weights_ih, weights_hh, biases, h0, c0 = ctx.saved_tensors
        return None, None, None, None, None, None

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, bidirectional=True):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Create parameters for each layer and direction
        self.weights_ih = nn.ParameterList()
        self.weights_hh = nn.ParameterList()
        self.biases = nn.ParameterList()
        
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            
            # Forward and backward directions
            for direction in range(self.num_directions):
                # Input to hidden weights (4 gates: input, forget, cell, output)
                w_ih = nn.Parameter(torch.randn(4 * hidden_size, layer_input_size))
                # Hidden to hidden weights
                w_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
                # Biases
                bias = nn.Parameter(torch.randn(4 * hidden_size))
                
                self.weights_ih.append(w_ih)
                self.weights_hh.append(w_hh)
                self.biases.append(bias)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for i in range(len(self.weights_ih)):
            nn.init.xavier_uniform_(self.weights_ih[i])
            nn.init.xavier_uniform_(self.weights_hh[i])
            nn.init.zeros_(self.biases[i])
    
    def forward(self, input, hx=None):
        batch_size = input.size(0)
        
        if hx is None:
            h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=input.device)
            c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=input.device)
        else:
            h0, c0 = hx
        
        # Reshape parameters for easier access
        weights_ih_layered = []
        weights_hh_layered = []
        biases_layered = []
        
        idx = 0
        for layer in range(self.num_layers):
            layer_weights_ih = []
            layer_weights_hh = []
            layer_biases = []
            
            for direction in range(self.num_directions):
                layer_weights_ih.append(self.weights_ih[idx])
                layer_weights_hh.append(self.weights_hh[idx])
                layer_biases.append(self.biases[idx])
                idx += 1
            
            weights_ih_layered.append(layer_weights_ih)
            weights_hh_layered.append(layer_weights_hh)
            biases_layered.append(layer_biases)
        
        return CustomLSTMFunction.apply(input, weights_ih_layered, weights_hh_layered, biases_layered, h0, c0)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.lstm = CustomLSTM(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x, h0, c0):
        out, hn = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out