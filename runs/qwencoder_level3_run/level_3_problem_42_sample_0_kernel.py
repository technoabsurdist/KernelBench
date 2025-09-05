import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GRU cell computation
gru_source = """
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

__global__ void gru_forward_kernel(
    const float* input,
    const float* hidden,
    const float* weight_ih,
    const float* weight_hh,
    const float* bias_ih,
    const float* bias_hh,
    float* output,
    int batch_size,
    int input_size,
    int hidden_size
) {
    int batch_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || hidden_idx >= hidden_size) return;
    
    // Calculate indices
    int input_offset = batch_idx * input_size;
    int hidden_offset = batch_idx * hidden_size;
    int output_offset = batch_idx * hidden_size;
    
    // Load input and hidden states
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_hidden = shared_mem + input_size;
    
    // Load data into shared memory
    if (hidden_idx < input_size) {
        shared_input[hidden_idx] = input[input_offset + hidden_idx];
    }
    shared_hidden[hidden_idx] = hidden[hidden_offset + hidden_idx];
    __syncthreads();
    
    // Compute intermediate values
    float r = 0.0f, z = 0.0f, n = 0.0f;
    
    // Reset gate
    for (int i = 0; i < input_size; i++) {
        r += shared_input[i] * weight_ih[hidden_idx * input_size + i];
    }
    for (int i = 0; i < hidden_size; i++) {
        r += shared_hidden[i] * weight_hh[hidden_idx * hidden_size + i];
    }
    r += bias_ih[hidden_idx] + bias_hh[hidden_idx];
    r = 1.0f / (1.0f + expf(-r)); // sigmoid
    
    // Update gate
    float z_val = 0.0f;
    for (int i = 0; i < input_size; i++) {
        z_val += shared_input[i] * weight_ih[(hidden_size + hidden_idx) * input_size + i];
    }
    for (int i = 0; i < hidden_size; i++) {
        z_val += shared_hidden[i] * weight_hh[(hidden_size + hidden_idx) * hidden_size + i];
    }
    z_val += bias_ih[hidden_size + hidden_idx] + bias_hh[hidden_size + hidden_idx];
    z = 1.0f / (1.0f + expf(-z_val)); // sigmoid
    
    // New gate
    float n_val = 0.0f;
    for (int i = 0; i < input_size; i++) {
        n_val += shared_input[i] * weight_ih[(2 * hidden_size + hidden_idx) * input_size + i];
    }
    for (int i = 0; i < hidden_size; i++) {
        n_val += r * shared_hidden[i] * weight_hh[(2 * hidden_size + hidden_idx) * hidden_size + i];
    }
    n_val += bias_ih[2 * hidden_size + hidden_idx] + bias_hh[2 * hidden_size + hidden_idx];
    n = tanhf(n_val); // tanh
    
    // Final output
    output[output_offset + hidden_idx] = (1.0f - z) * n + z * shared_hidden[hidden_idx];
}

torch::Tensor gru_cell_forward(
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
    
    const int threads_per_block = 256;
    const int blocks = batch_size;
    const int shared_mem_size = (input_size + hidden_size) * sizeof(float);
    
    gru_forward_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        hidden.data_ptr<float>(),
        weight_ih.data_ptr<float>(),
        weight_hh.data_ptr<float>(),
        bias_ih.data_ptr<float>(),
        bias_hh.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    return output;
}
"""

gru_cpp_source = """
torch::Tensor gru_cell_forward(
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
    functions=["gru_cell_forward"],
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
        
        # First layer
        self.weight_ih_l.append(nn.Parameter(torch.randn(3 * hidden_size, input_size)))
        self.weight_hh_l.append(nn.Parameter(torch.randn(3 * hidden_size, hidden_size)))
        if bias:
            self.bias_ih_l.append(nn.Parameter(torch.randn(3 * hidden_size)))
            self.bias_hh_l.append(nn.Parameter(torch.randn(3 * hidden_size)))
        else:
            self.bias_ih_l.append(None)
            self.bias_hh_l.append(None)
        
        # Additional layers
        for i in range(1, num_layers * 2):  # *2 for bidirectional
            self.weight_ih_l.append(nn.Parameter(torch.randn(3 * hidden_size, hidden_size * 2)))  # *2 for bidirectional
            self.weight_hh_l.append(nn.Parameter(torch.randn(3 * hidden_size, hidden_size)))
            if bias:
                self.bias_ih_l.append(nn.Parameter(torch.randn(3 * hidden_size)))
                self.bias_hh_l.append(nn.Parameter(torch.randn(3 * hidden_size)))
            else:
                self.bias_ih_l.append(None)
                self.bias_hh_l.append(None)
        
        self.gru_fn = gru_module
        
        # Initialize parameters
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x, h0):
        if self.batch_first:
            x = x.transpose(0, 1)  # Convert to (seq_len, batch, feature)
        
        seq_len, batch_size, _ = x.size()
        num_directions = 2
        num_layers = self.num_layers
        
        # Prepare output
        output = torch.zeros(seq_len, batch_size, num_directions * self.hidden_size, device=x.device)
        h_n = torch.zeros(num_layers * num_directions, batch_size, self.hidden_size, device=x.device)
        
        # Process each layer
        for layer in range(num_layers):
            # Forward direction
            h_prev = h0[layer * num_directions, :, :]
            for t in range(seq_len):
                if layer == 0:
                    x_t = x[t]
                else:
                    x_t = output[t, :, :self.hidden_size]
                
                weight_ih = self.weight_ih_l[layer * num_directions]
                weight_hh = self.weight_hh_l[layer * num_directions]
                bias_ih = self.bias_ih_l[layer * num_directions] if self.bias else torch.zeros(3 * self.hidden_size, device=x.device)
                bias_hh = self.bias_hh_l[layer * num_directions] if self.bias else torch.zeros(3 * self.hidden_size, device=x.device)
                
                if not self.bias:
                    bias_ih = bias_ih.to(x.device)
                    bias_hh = bias_hh.to(x.device)
                
                h_t = self.gru_fn.gru_cell_forward(x_t, h_prev, weight_ih, weight_hh, bias_ih, bias_hh)
                output[t, :, :self.hidden_size] = h_t
                h_prev = h_t
            
            h_n[layer * num_directions] = h_prev
            
            # Backward direction
            h_prev = h0[layer * num_directions + 1, :, :]
            for t in range(seq_len - 1, -1, -1):
                if layer == 0:
                    x_t = x[t]
                else:
                    x_t = output[t, :, self.hidden_size:]
                
                weight_ih = self.weight_ih_l[layer * num_directions + 1]
                weight_hh = self.weight_hh_l[layer * num_directions + 1]
                bias_ih = self.bias_ih_l[layer * num_directions + 1] if self.bias else torch.zeros(3 * self.hidden_size, device=x.device)
                bias_hh = self.bias_hh_l[layer * num_directions + 1] if self.bias else torch.zeros(3 * self.hidden_size, device=x.device)
                
                if not self.bias:
                    bias_ih = bias_ih.to(x.device)
                    bias_hh = bias_hh.to(x.device)
                
                h_t = self.gru_fn.gru_cell_forward(x_t, h_prev, weight_ih, weight_hh, bias_ih, bias_hh)
                output[t, :, self.hidden_size:] = h_t
                h_prev = h_t
            
            h_n[layer * num_directions + 1] = h_prev
        
        if self.batch_first:
            output = output.transpose(0, 1)
        
        return h_n