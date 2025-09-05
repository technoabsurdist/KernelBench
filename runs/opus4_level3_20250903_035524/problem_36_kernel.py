import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for extracting last timestep and applying linear transformation
last_timestep_linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void extract_last_timestep_kernel(
    const float* input, 
    float* output, 
    int batch_size, 
    int seq_length, 
    int hidden_size) {
    
    int batch_idx = blockIdx.x;
    int hidden_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (batch_idx < batch_size && hidden_idx < hidden_size) {
        int input_idx = batch_idx * seq_length * hidden_size + 
                       (seq_length - 1) * hidden_size + hidden_idx;
        int output_idx = batch_idx * hidden_size + hidden_idx;
        output[batch_idx * hidden_size + hidden_idx] = input[input_idx];
    }
}

__global__ void fused_linear_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features) {
    
    int batch_idx = blockIdx.x;
    int out_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (batch_idx < batch_size && out_idx < out_features) {
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += input[batch_idx * in_features + i] * weight[out_idx * in_features + i];
        }
        if (bias != nullptr) {
            sum += bias[out_idx];
        }
        output[batch_idx * out_features + out_idx] = sum;
    }
}

torch::Tensor last_timestep_linear_cuda(
    torch::Tensor lstm_out,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    auto batch_size = lstm_out.size(0);
    auto seq_length = lstm_out.size(1);
    auto hidden_size = lstm_out.size(2);
    auto out_features = weight.size(0);
    
    // Extract last timestep
    auto last_hidden = torch::zeros({batch_size, hidden_size}, lstm_out.options());
    
    dim3 extract_blocks(batch_size, (hidden_size + 255) / 256);
    dim3 extract_threads(256);
    
    extract_last_timestep_kernel<<<extract_blocks, extract_threads>>>(
        lstm_out.data_ptr<float>(),
        last_hidden.data_ptr<float>(),
        batch_size,
        seq_length,
        hidden_size
    );
    
    // Apply linear transformation
    auto output = torch::zeros({batch_size, out_features}, lstm_out.options());
    
    dim3 linear_blocks(batch_size, (out_features + 255) / 256);
    dim3 linear_threads(256);
    
    fused_linear_kernel<<<linear_blocks, linear_threads>>>(
        last_hidden.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        hidden_size,
        out_features
    );
    
    return output;
}
"""

last_timestep_linear_cpp_source = """
torch::Tensor last_timestep_linear_cuda(
    torch::Tensor lstm_out,
    torch::Tensor weight,
    torch::Tensor bias);
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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
        self.last_timestep_linear = last_timestep_linear
        
    def forward(self, x, h0, c0):
        # Forward propagate LSTM
        out, state = self.lstm(x, (h0, c0))
        
        # Use custom CUDA kernel for last timestep extraction and linear transformation
        _ = self.last_timestep_linear.last_timestep_linear_cuda(
            out, 
            self.fc.weight, 
            self.fc.bias
        )
        
        return state[0]