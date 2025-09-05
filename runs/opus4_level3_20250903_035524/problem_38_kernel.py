import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for extracting last timestep and applying linear transformation
last_timestep_linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void extract_last_timestep_matmul_kernel(
    const float* lstm_out,  // (batch_size, seq_length, hidden_size*2)
    const float* weight,    // (output_size, hidden_size*2)
    const float* bias,      // (output_size)
    float* output,          // (batch_size, output_size)
    int batch_size,
    int seq_length,
    int hidden_size_2x,
    int output_size
) {
    int batch_idx = blockIdx.x;
    int out_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (batch_idx < batch_size && out_idx < output_size) {
        float sum = 0.0f;
        
        // Extract last timestep and compute matrix multiplication
        int last_timestep_offset = batch_idx * seq_length * hidden_size_2x + (seq_length - 1) * hidden_size_2x;
        
        for (int k = 0; k < hidden_size_2x; k++) {
            sum += lstm_out[last_timestep_offset + k] * weight[out_idx * hidden_size_2x + k];
        }
        
        // Add bias
        if (bias != nullptr) {
            sum += bias[out_idx];
        }
        
        output[batch_idx * output_size + out_idx] = sum;
    }
}

torch::Tensor last_timestep_linear_cuda(
    torch::Tensor lstm_out,
    torch::Tensor weight,
    torch::Tensor bias
) {
    auto batch_size = lstm_out.size(0);
    auto seq_length = lstm_out.size(1);
    auto hidden_size_2x = lstm_out.size(2);
    auto output_size = weight.size(0);
    
    auto output = torch::zeros({batch_size, output_size}, lstm_out.options());
    
    const int threads_per_block = 256;
    dim3 blocks(batch_size, (output_size + threads_per_block - 1) / threads_per_block);
    
    last_timestep_matmul_kernel<<<blocks, threads_per_block>>>(
        lstm_out.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        seq_length,
        hidden_size_2x,
        output_size
    );
    
    return output;
}
"""

last_timestep_linear_cpp_source = """
torch::Tensor last_timestep_linear_cuda(
    torch::Tensor lstm_out,
    torch::Tensor weight,
    torch::Tensor bias
);
"""

# Compile the inline CUDA code
last_timestep_linear = load_inline(
    name="last_timestep_linear",
    cpp_sources=last_timestep_linear_cpp_source,
    cuda_sources=last_timestep_linear_source,
    functions=["last_timestep_linear_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.last_timestep_linear = last_timestep_linear
    
    def forward(self, x, h0, c0):
        # Forward propagate LSTM
        out, hn = self.lstm(x, (h0, c0))
        
        # Use custom CUDA kernel for last timestep extraction and linear transformation
        out = self.last_timestep_linear.last_timestep_linear_cuda(
            out.cuda(),
            self.fc.weight.cuda(),
            self.fc.bias.cuda() if self.fc.bias is not None else torch.empty(0).cuda()
        )
        
        return out

# Test code
batch_size = 10
sequence_length = 512
input_size = 128
hidden_size = 256
num_layers = 6
output_size = 10
dropout = 0.0

def get_inputs():
    return [torch.rand(batch_size, sequence_length, input_size),torch.rand((num_layers*2, batch_size, hidden_size)),torch.rand((num_layers*2, batch_size, hidden_size))]

def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, dropout]