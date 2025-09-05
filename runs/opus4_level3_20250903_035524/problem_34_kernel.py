import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused concat + linear + tanh operation
fused_rnn_cell_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

__global__ void fused_concat_linear_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ hidden,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int input_size,
    int hidden_size) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / hidden_size;
    int out_idx = tid % hidden_size;
    
    if (batch_idx < batch_size && out_idx < hidden_size) {
        float sum = bias[out_idx];
        
        // Process input part
        for (int i = 0; i < input_size; i++) {
            sum += input[batch_idx * input_size + i] * weight[out_idx * (input_size + hidden_size) + i];
        }
        
        // Process hidden part
        for (int h = 0; h < hidden_size; h++) {
            sum += hidden[batch_idx * hidden_size + h] * weight[out_idx * (input_size + hidden_size) + input_size + h];
        }
        
        // Apply tanh activation
        output[batch_idx * hidden_size + out_idx] = tanhf(sum);
    }
}

__global__ void linear_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int input_size,
    int output_size) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / output_size;
    int out_idx = tid % output_size;
    
    if (batch_idx < batch_size && out_idx < output_size) {
        float sum = bias[out_idx];
        
        for (int i = 0; i < input_size; i++) {
            sum += input[batch_idx * input_size + i] * weight[out_idx * input_size + i];
        }
        
        output[batch_idx * output_size + out_idx] = sum;
    }
}

torch::Tensor fused_concat_linear_tanh_cuda(
    torch::Tensor input,
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto hidden_size = hidden.size(1);
    
    auto output = torch::zeros({batch_size, hidden_size}, input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * hidden_size + threads - 1) / threads;
    
    fused_concat_linear_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        hidden.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size);
    
    return output;
}

torch::Tensor linear_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto output_size = weight.size(0);
    
    auto output = torch::zeros({batch_size, output_size}, input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * output_size + threads - 1) / threads;
    
    linear_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        output_size);
    
    return output;
}
"""

fused_rnn_cell_cpp_source = """
torch::Tensor fused_concat_linear_tanh_cuda(
    torch::Tensor input,
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor bias);
    
torch::Tensor linear_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias);
"""

# Compile the inline CUDA code
rnn_ops = load_inline(
    name="rnn_ops",
    cpp_sources=fused_rnn_cell_cpp_source,
    cuda_sources=fused_rnn_cell_source,
    functions=["fused_concat_linear_tanh_cuda", "linear_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the Vanilla RNN model with custom CUDA kernels.

        :param input_size: The number of input features (int).
        :param hidden_size: The size of the hidden state (int).
        :param output_size: The number of output features (int).
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases for i2h
        self.i2h_weight = nn.Parameter(torch.randn(hidden_size, input_size + hidden_size) * 0.01)
        self.i2h_bias = nn.Parameter(torch.zeros(hidden_size))
        
        # Initialize weights and biases for h2o
        self.h2o_weight = nn.Parameter(torch.randn(output_size, hidden_size) * 0.01)
        self.h2o_bias = nn.Parameter(torch.zeros(output_size))
        
        self.rnn_ops = rnn_ops

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vanilla RNN with custom CUDA kernels.

        :param x: Input tensor of shape (seq_len, batch_size, input_size)
        :param h0: Initial hidden state tensor of shape (batch_size, hidden_size)
        :return: Output tensor of shape (seq_len, batch_size, output_size)
        """
        seq_len, batch_size, _ = x.size()
        hidden = h0.to(x.device)
        outputs = []
        
        # Ensure all tensors are contiguous and on the same device
        x = x.contiguous()
        hidden = hidden.contiguous()
        i2h_weight = self.i2h_weight.contiguous()
        i2h_bias = self.i2h_bias.contiguous()
        h2o_weight = self.h2o_weight.contiguous()
        h2o_bias = self.h2o_bias.contiguous()

        for t in range(seq_len):
            # Fused concat + linear + tanh for hidden state update
            hidden = self.rnn_ops.fused_concat_linear_tanh_cuda(
                x[t], hidden, i2h_weight, i2h_bias
            )
            
            # Linear transformation for output
            output = self.rnn_ops.linear_cuda(
                hidden, h2o_weight, h2o_bias
            )
            outputs.append(output)

        return torch.stack(outputs, dim=0)  # (seq_len, batch_size, output_size)

# === Test configuration ===
batch_size = 8
input_size = 1024
hidden_size = 256
output_size = 128
sequence_length = 256

def get_inputs():
    return [
        torch.rand(sequence_length, batch_size, input_size),
        torch.rand(batch_size, hidden_size)
    ]

def get_init_inputs():
    return [input_size, hidden_size, output_size]