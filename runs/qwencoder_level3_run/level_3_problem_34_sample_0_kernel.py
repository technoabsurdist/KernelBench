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
    const float* x_t, 
    const float* h_prev, 
    const float* w_ih, 
    const float* b_ih, 
    const float* w_ho, 
    const float* b_ho, 
    float* h_next, 
    float* output,
    int input_size,
    int hidden_size,
    int output_size,
    int batch_size
) {
    int batch_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || hidden_idx >= hidden_size) return;
    
    // Calculate linear combination for hidden state
    float h_val = 0.0f;
    
    // Process input to hidden (first input_size elements)
    for (int i = 0; i < input_size; i++) {
        h_val += x_t[batch_idx * input_size + i] * w_ih[hidden_idx * (input_size + hidden_size) + i];
    }
    
    // Process hidden to hidden (next hidden_size elements)
    for (int i = 0; i < hidden_size; i++) {
        h_val += h_prev[batch_idx * hidden_size + i] * w_ih[hidden_idx * (input_size + hidden_size) + input_size + i];
    }
    
    // Add bias
    h_val += b_ih[hidden_idx];
    
    // Apply tanh activation
    h_val = tanhf(h_val);
    
    // Store new hidden state
    h_next[batch_idx * hidden_size + hidden_idx] = h_val;
    
    // Compute output if this thread is responsible for it
    if (hidden_idx < output_size) {
        float out_val = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            out_val += h_val * w_ho[hidden_idx * hidden_size + i];
        }
        out_val += b_ho[hidden_idx];
        output[batch_idx * output_size + hidden_idx] = out_val;
    }
}

torch::Tensor fused_rnn_cell_cuda(
    torch::Tensor x_t,
    torch::Tensor h_prev,
    torch::Tensor w_ih,
    torch::Tensor b_ih,
    torch::Tensor w_ho,
    torch::Tensor b_ho
) {
    auto batch_size = x_t.size(0);
    auto input_size = x_t.size(1);
    auto hidden_size = h_prev.size(1);
    auto output_size = w_ho.size(0);
    
    auto h_next = torch::zeros_like(h_prev);
    auto output = torch::zeros({batch_size, output_size}, torch::kCUDA);
    
    dim3 grid(batch_size);
    dim3 block(std::max(hidden_size, output_size));
    
    rnn_cell_kernel<<<grid, block>>>(
        x_t.data_ptr<float>(),
        h_prev.data_ptr<float>(),
        w_ih.data_ptr<float>(),
        b_ih.data_ptr<float>(),
        w_ho.data_ptr<float>(),
        b_ho.data_ptr<float>(),
        h_next.data_ptr<float>(),
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
torch::Tensor fused_rnn_cell_cuda(
    torch::Tensor x_t,
    torch::Tensor h_prev,
    torch::Tensor w_ih,
    torch::Tensor b_ih,
    torch::Tensor w_ho,
    torch::Tensor b_ho
);
"""

# Compile the inline CUDA code for fused RNN cell
fused_rnn_cell = load_inline(
    name="fused_rnn_cell",
    cpp_sources=rnn_cell_cpp_source,
    cuda_sources=rnn_cell_source,
    functions=["fused_rnn_cell_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Optimized Vanilla RNN model with custom CUDA kernels.
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Define the RNN cell components
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        
        # Register the custom CUDA function
        self.fused_rnn_cell = fused_rnn_cell

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the optimized Vanilla RNN.

        :param x: Input tensor of shape (seq_len, batch_size, input_size)
        :param h0: Initial hidden state tensor of shape (batch_size, hidden_size)
        :return: Output tensor of shape (seq_len, batch_size, output_size)
        """
        seq_len, batch_size, _ = x.size()
        hidden = h0.to(x.device)
        outputs = []

        # Get parameters once to avoid repeated access
        w_ih = self.i2h.weight
        b_ih = self.i2h.bias
        w_ho = self.h2o.weight
        b_ho = self.h2o.bias

        for t in range(seq_len):
            # Use custom CUDA kernel for fused computation
            output = self.fused_rnn_cell.fused_rnn_cell_cuda(
                x[t], hidden, w_ih, b_ih, w_ho, b_ho
            )
            # Update hidden state (last hidden_size elements of the concatenated input)
            hidden = torch.tanh(
                torch.matmul(
                    torch.cat((x[t], hidden), dim=1), 
                    w_ih.transpose(0, 1)
                ) + b_ih
            )
            outputs.append(output)

        return torch.stack(outputs, dim=0)