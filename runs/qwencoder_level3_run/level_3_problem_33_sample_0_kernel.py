import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused RNN cell computation
rnn_cell_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void rnn_cell_kernel(
    const float* x,
    const float* hidden,
    const float* weight_ih,
    const float* weight_hh,
    const float* bias_ih,
    const float* bias_hh,
    const float* weight_ho,
    const float* bias_ho,
    float* new_hidden,
    float* output,
    int input_size,
    int hidden_size,
    int output_size,
    int batch_size
) {
    int batch_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || hidden_idx >= hidden_size) return;
    
    // Calculate linear combination for input to hidden
    float ih_sum = 0.0f;
    float hh_sum = 0.0f;
    
    // Process input to hidden (first hidden_size elements)
    for (int i = 0; i < input_size; ++i) {
        ih_sum += x[batch_idx * input_size + i] * weight_ih[hidden_idx * (input_size + hidden_size) + i];
    }
    
    // Process hidden to hidden (next hidden_size elements)
    for (int i = 0; i < hidden_size; ++i) {
        hh_sum += hidden[batch_idx * hidden_size + i] * 
                  weight_ih[hidden_idx * (input_size + hidden_size) + input_size + i];
    }
    
    // Add biases
    float ih_bias = bias_ih[hidden_idx];
    float hh_bias = bias_hh[hidden_idx];
    
    // Apply tanh activation
    float pre_tanh = ih_sum + hh_sum + ih_bias + hh_bias;
    float tanh_val = tanhf(pre_tanh);
    
    // Store new hidden state
    new_hidden[batch_idx * hidden_size + hidden_idx] = tanh_val;
    
    // Only thread 0 computes output for this batch
    if (hidden_idx == 0) {
        // Compute hidden to output
        for (int out_idx = 0; out_idx < output_size; ++out_idx) {
            float out_sum = 0.0f;
            for (int h_idx = 0; h_idx < hidden_size; ++h_idx) {
                out_sum += tanh_val * weight_ho[out_idx * hidden_size + h_idx];
            }
            output[batch_idx * output_size + out_idx] = out_sum + bias_ho[out_idx];
        }
    }
}

torch::Tensor fused_rnn_cell_forward(
    torch::Tensor x,
    torch::Tensor hidden,
    torch::Tensor weight_ih,
    torch::Tensor weight_hh,
    torch::Tensor bias_ih,
    torch::Tensor bias_hh,
    torch::Tensor weight_ho,
    torch::Tensor bias_ho
) {
    auto batch_size = x.size(0);
    auto input_size = x.size(1);
    auto hidden_size = hidden.size(1);
    auto output_size = weight_ho.size(0);
    
    auto new_hidden = torch::zeros_like(hidden);
    auto output = torch::zeros({batch_size, output_size}, x.options());
    
    dim3 grid(batch_size);
    dim3 block(hidden_size);
    
    rnn_cell_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        hidden.data_ptr<float>(),
        weight_ih.data_ptr<float>(),
        weight_hh.data_ptr<float>(),
        bias_ih.data_ptr<float>(),
        bias_hh.data_ptr<float>(),
        weight_ho.data_ptr<float>(),
        bias_ho.data_ptr<float>(),
        new_hidden.data_ptr<float>(),
        output.data_ptr<float>(),
        input_size,
        hidden_size,
        output_size,
        batch_size
    );
    
    return output;
}
"""

rnn_cell_cpp_source = """
torch::Tensor fused_rnn_cell_forward(
    torch::Tensor x,
    torch::Tensor hidden,
    torch::Tensor weight_ih,
    torch::Tensor weight_hh,
    torch::Tensor bias_ih,
    torch::Tensor bias_hh,
    torch::Tensor weight_ho,
    torch::Tensor bias_ho
);
"""

# Compile the inline CUDA code for fused RNN cell
fused_rnn_cell = load_inline(
    name="fused_rnn_cell",
    cpp_sources=rnn_cell_cpp_source,
    cuda_sources=rnn_cell_source,
    functions=["fused_rnn_cell_forward"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Optimized Vanilla RNN model with custom CUDA kernels.
        
        :param input_size: The number of input features (int).
        :param hidden_size: The size of the hidden state (int).
        :param output_size: The number of output features (int).
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Define the RNN cell components
        self.weight_ih = nn.Parameter(torch.randn(hidden_size, input_size + hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(hidden_size))
        self.weight_ho = nn.Parameter(torch.randn(output_size, hidden_size))
        self.bias_ho = nn.Parameter(torch.randn(output_size))
        
        # Initialize hidden state buffer
        self.register_buffer('hidden', torch.randn(256, hidden_size))
        
        # Load custom CUDA function
        self.fused_rnn_cell = fused_rnn_cell
    
    def forward(self, x: torch.Tensor, initial_hidden=None) -> torch.Tensor:
        """
        Forward pass of the optimized Vanilla RNN.
        
        :param x: Input tensor of shape (batch_size, input_size).
        :param initial_hidden: Initial hidden state tensor of shape (batch_size, hidden_size).
        :return: Output tensor of shape (batch_size, output_size).
        """
        if initial_hidden is not None:
            self.hidden[:x.size(0)].copy_(initial_hidden)
        
        # Ensure hidden state is on the same device as input
        self.hidden = self.hidden.to(x.device)
        
        # Split weight matrix for input and hidden parts
        weight_input = self.weight_ih[:, :self.input_size]
        weight_hidden = self.weight_ih[:, self.input_size:]
        bias_hidden = torch.zeros_like(self.bias_ih)
        
        # Call custom CUDA kernel
        output = self.fused_rnn_cell.fused_rnn_cell_forward(
            x,
            self.hidden[:x.size(0)],
            weight_input,
            weight_hidden,
            self.bias_ih,
            bias_hidden,
            self.weight_ho,
            self.bias_ho
        )
        
        return output