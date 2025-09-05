import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GRU cell computation
gru_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_NUM_THREADS 256
#define CUDA_GET_BLOCKS(N) (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void gru_kernel(
    const float* input,
    const float* hidden,
    const float* weight_ih,
    const float* weight_hh,
    const float* bias_ih,
    const float* bias_hh,
    float* output,
    int input_size,
    int hidden_size,
    int batch_size
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    int stride = blockDim.x * gridDim.x;
    
    for (int b = batch_idx; b < batch_size; b += stride) {
        // Compute gates for current batch
        for (int h = 0; h < hidden_size; h++) {
            float reset = 0.0f;
            float update = 0.0f;
            float new = 0.0f;
            
            // Compute reset gate (first hidden_size elements)
            for (int i = 0; i < input_size; i++) {
                reset += input[b * input_size + i] * weight_ih[h * input_size + i];
            }
            for (int i = 0; i < hidden_size; i++) {
                reset += hidden[b * hidden_size + i] * weight_hh[h * hidden_size + i];
            }
            reset += bias_ih[h] + bias_hh[h];
            reset = sigmoid(reset);
            
            // Compute update gate (second hidden_size elements)
            for (int i = 0; i < input_size; i++) {
                update += input[b * input_size + i] * weight_ih[(h + hidden_size) * input_size + i];
            }
            for (int i = 0; i < hidden_size; i++) {
                update += hidden[b * hidden_size + i] * weight_hh[(h + hidden_size) * hidden_size + i];
            }
            update += bias_ih[h + hidden_size] + bias_hh[h + hidden_size];
            update = sigmoid(update);
            
            // Compute new gate (third hidden_size elements)
            for (int i = 0; i < input_size; i++) {
                new += input[b * input_size + i] * weight_ih[(h + 2 * hidden_size) * input_size + i];
            }
            for (int i = 0; i < hidden_size; i++) {
                new += reset * hidden[b * hidden_size + i] * weight_hh[(h + 2 * hidden_size) * hidden_size + i];
            }
            new += bias_ih[h + 2 * hidden_size] + bias_hh[h + 2 * hidden_size];
            new = tanhf(new);
            
            // Compute output
            float h_prev = hidden[b * hidden_size + h];
            output[b * hidden_size + h] = update * h_prev + (1 - update) * new;
        }
    }
}

torch::Tensor gru_cell_cuda(
    torch::Tensor input,
    torch::Tensor hidden,
    torch::Tensor weight_ih,
    torch::Tensor weight_hh,
    torch::Tensor bias_ih,
    torch::Tensor bias_hh
) {
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto hidden_size = hidden.size(1);
    
    auto output = torch::zeros_like(hidden);
    
    const int threads = CUDA_NUM_THREADS;
    const int blocks = CUDA_GET_BLOCKS(batch_size);
    
    gru_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        hidden.data_ptr<float>(),
        weight_ih.data_ptr<float>(),
        weight_hh.data_ptr<float>(),
        bias_ih.data_ptr<float>(),
        bias_hh.data_ptr<float>(),
        output.data_ptr<float>(),
        input_size,
        hidden_size,
        batch_size
    );
    
    return output;
}
"""

gru_cpp_source = """
torch::Tensor gru_cell_cuda(
    torch::Tensor input,
    torch::Tensor hidden,
    torch::Tensor weight_ih,
    torch::Tensor weight_hh,
    torch::Tensor bias_ih,
    torch::Tensor bias_hh
);
"""

# Compile the inline CUDA code for GRU
gru_module = load_inline(
    name="gru_module",
    cpp_sources=gru_cpp_source,
    cuda_sources=gru_source,
    functions=["gru_cell_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        
        # Create parameters for each layer
        self.weight_ih_l = nn.ParameterList()
        self.weight_hh_l = nn.ParameterList()
        self.bias_ih_l = nn.ParameterList()
        self.bias_hh_l = nn.ParameterList()
        
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            
            # Weight matrices
            self.weight_ih_l.append(nn.Parameter(torch.randn(3 * hidden_size, layer_input_size)))
            self.weight_hh_l.append(nn.Parameter(torch.randn(3 * hidden_size, hidden_size)))
            
            # Bias vectors
            if bias:
                self.bias_ih_l.append(nn.Parameter(torch.randn(3 * hidden_size)))
                self.bias_hh_l.append(nn.Parameter(torch.randn(3 * hidden_size)))
            else:
                self.bias_ih_l.append(None)
                self.bias_hh_l.append(None)
        
        # Initialize parameters
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x, h0):
        if self.batch_first:
            x = x.transpose(0, 1)  # Convert to (seq_len, batch, input_size)
        
        seq_len, batch_size, _ = x.size()
        
        # Initialize hidden states
        h_prev = list(h0.unbind(0)) if h0 is not None else [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        
        outputs = []
        
        # Process each time step
        for t in range(seq_len):
            x_t = x[t]  # (batch_size, input_size)
            
            # Process through each layer
            for layer in range(self.num_layers):
                layer_input = x_t if layer == 0 else h_t
                
                weight_ih = self.weight_ih_l[layer]
                weight_hh = self.weight_hh_l[layer]
                
                if self.bias:
                    bias_ih = self.bias_ih_l[layer]
                    bias_hh = self.bias_hh_l[layer]
                else:
                    bias_ih = torch.zeros(3 * self.hidden_size, device=x.device)
                    bias_hh = torch.zeros(3 * self.hidden_size, device=x.device)
                
                # Apply custom GRU cell
                h_t = gru_module.gru_cell_cuda(
                    layer_input.contiguous(),
                    h_prev[layer].contiguous(),
                    weight_ih.contiguous(),
                    weight_hh.contiguous(),
                    bias_ih.contiguous(),
                    bias_hh.contiguous()
                )
                
                h_prev[layer] = h_t
            
            outputs.append(h_t)
        
        output = torch.stack(outputs, dim=0)  # (seq_len, batch_size, hidden_size)
        
        if self.batch_first:
            output = output.transpose(0, 1)  # Convert to (batch, seq_len, hidden_size)
        
        h_n = torch.stack(h_prev, dim=0)  # (num_layers, batch_size, hidden_size)
        
        return output, h_n