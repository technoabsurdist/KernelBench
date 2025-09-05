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

__global__ void gru_kernel(
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
        r += weight_ih[hidden_idx * input_size + i] * shared_input[i];
    }
    for (int i = 0; i < hidden_size; i++) {
        r += weight_hh[hidden_idx * hidden_size + i] * shared_hidden[i];
    }
    r += bias_ih[hidden_idx] + bias_hh[hidden_idx];
    r = 1.0f / (1.0f + expf(-r)); // sigmoid
    
    // Update gate
    for (int i = 0; i < input_size; i++) {
        z += weight_ih[(hidden_size + hidden_idx) * input_size + i] * shared_input[i];
    }
    for (int i = 0; i < hidden_size; i++) {
        z += weight_hh[(hidden_size + hidden_idx) * hidden_size + i] * shared_hidden[i];
    }
    z += bias_ih[hidden_size + hidden_idx] + bias_hh[hidden_size + hidden_idx];
    z = 1.0f / (1.0f + expf(-z)); // sigmoid
    
    // New gate
    for (int i = 0; i < input_size; i++) {
        n += weight_ih[(2 * hidden_size + hidden_idx) * input_size + i] * shared_input[i];
    }
    for (int i = 0; i < hidden_size; i++) {
        n += weight_hh[(2 * hidden_size + hidden_idx) * hidden_size + i] * (r * shared_hidden[i]);
    }
    n += bias_ih[2 * hidden_size + hidden_idx] + bias_hh[2 * hidden_size + hidden_idx];
    n = tanhf(n); // tanh
    
    // Final output
    float h_new = (1.0f - z) * n + z * shared_hidden[hidden_idx];
    output[output_offset + hidden_idx] = h_new;
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
    
    const int threads_per_block = hidden_size;
    const int blocks_per_grid = batch_size;
    const int shared_mem_size = (input_size + hidden_size) * sizeof(float);
    
    gru_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
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
gru_ops = load_inline(
    name="gru_ops",
    cpp_sources=gru_cpp_source,
    cuda_sources=gru_source,
    functions=["gru_cell_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class CustomGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(CustomGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        
        if bias:
            self.bias_ih = nn.Parameter(torch.randn(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.randn(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        if self.bias:
            nn.init.zeros_(self.bias_ih)
            nn.init.zeros_(self.bias_hh)
            
    def forward(self, input, hidden):
        if self.bias:
            return gru_ops.gru_cell_cuda(
                input, hidden, 
                self.weight_ih, self.weight_hh,
                self.bias_ih, self.bias_hh
            )
        else:
            zero_bias = torch.zeros(3 * self.hidden_size, device=input.device, dtype=input.dtype)
            return gru_ops.gru_cell_cuda(
                input, hidden,
                self.weight_ih, self.weight_hh,
                zero_bias, zero_bias
            )

class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False):
        super(CustomGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Create layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size * self.num_directions
            self.layers.append(CustomGRUCell(layer_input_size, hidden_size, bias))
            
            if bidirectional:
                self.layers.append(CustomGRUCell(layer_input_size, hidden_size, bias))
                
    def forward(self, input, hx=None):
        is_batched = input.dim() == 3
        if not is_batched:
            input = input.unsqueeze(1)
            
        if self.batch_first:
            input = input.transpose(0, 1)
            
        seq_len, batch_size, input_size = input.size()
        
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, 
                           device=input.device, dtype=input.dtype)
            
        # Handle bidirectional layers
        output = input
        new_hx = []
        
        for i, layer in enumerate(self.layers):
            layer_output = []
            layer_hx = hx[i] if i < hx.size(0) else torch.zeros(batch_size, self.hidden_size, 
                                                               device=input.device, dtype=input.dtype)
            
            for t in range(seq_len):
                if self.batch_first:
                    x = output[t]
                else:
                    x = output[t]
                    
                layer_hx = layer(x, layer_hx)
                layer_output.append(layer_hx)
                
            output = torch.stack(layer_output)
            new_hx.append(layer_hx)
            
        new_hx = torch.stack(new_hx)
        
        if self.batch_first:
            output = output.transpose(0, 1)
            
        return output, new_hx

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()
        
        self.gru = CustomGRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=True)
        self.h0 = torch.randn((num_layers * 2, 10, hidden_size))  # batch_size=10
    
    def forward(self, x, h0):
        output, h_n = self.gru(x, h0)
        return output