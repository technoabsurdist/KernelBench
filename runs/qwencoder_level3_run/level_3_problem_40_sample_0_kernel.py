import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GRU cell computation
gru_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
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
    
    // Compute reset gate (r_t)
    float r_t = 0.0f;
    float z_t = 0.0f;
    float n_t = 0.0f;
    
    // Reset gate calculation
    for (int i = 0; i < input_size; i++) {
        r_t += shared_input[i] * weight_ih[hidden_idx * input_size + i];
    }
    for (int i = 0; i < hidden_size; i++) {
        r_t += shared_hidden[i] * weight_hh[hidden_idx * hidden_size + i];
    }
    r_t += bias_ih[hidden_idx] + bias_hh[hidden_idx];
    r_t = 1.0f / (1.0f + expf(-r_t)); // Sigmoid
    
    // Update gate calculation
    for (int i = 0; i < input_size; i++) {
        z_t += shared_input[i] * weight_ih[(hidden_idx + hidden_size) * input_size + i];
    }
    for (int i = 0; i < hidden_size; i++) {
        z_t += shared_hidden[i] * weight_hh[(hidden_idx + hidden_size) * hidden_size + i];
    }
    z_t += bias_ih[hidden_idx + hidden_size] + bias_hh[hidden_idx + hidden_size];
    z_t = 1.0f / (1.0f + expf(-z_t)); // Sigmoid
    
    // New gate calculation
    for (int i = 0; i < input_size; i++) {
        n_t += shared_input[i] * weight_ih[(hidden_idx + 2 * hidden_size) * input_size + i];
    }
    for (int i = 0; i < hidden_size; i++) {
        n_t += r_t * shared_hidden[i] * weight_hh[(hidden_idx + 2 * hidden_size) * hidden_size + i];
    }
    n_t += bias_ih[hidden_idx + 2 * hidden_size] + bias_hh[hidden_idx + 2 * hidden_size];
    n_t = tanhf(n_t); // Tanh
    
    // Final output calculation
    float h_new = (1.0f - z_t) * n_t + z_t * shared_hidden[hidden_idx];
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

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        :param input_size: The number of expected features in the input x
        :param hidden_size: The number of features in the hidden state h
        :param num_layers: Number of recurrent layers (default: 1)
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh (default: True)
        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature) (default: False)
        """
        super(ModelNew, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        
        # Create parameters for each layer
        self.weight_ih_l = nn.ParameterList()
        self.weight_hh_l = nn.ParameterList()
        self.bias_ih_l = nn.ParameterList() if bias else None
        self.bias_hh_l = nn.ParameterList() if bias else None
        
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            
            # Weight matrices for input-hidden connections (3 * hidden_size x input_size)
            self.weight_ih_l.append(nn.Parameter(torch.randn(3 * hidden_size, layer_input_size)))
            
            # Weight matrices for hidden-hidden connections (3 * hidden_size x hidden_size)
            self.weight_hh_l.append(nn.Parameter(torch.randn(3 * hidden_size, hidden_size)))
            
            if bias:
                # Bias vectors (3 * hidden_size)
                self.bias_ih_l.append(nn.Parameter(torch.randn(3 * hidden_size)))
                self.bias_hh_l.append(nn.Parameter(torch.randn(3 * hidden_size)))
        
        # Initialize parameters
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x, h0):
        """
        :param x: The input tensor, shape (seq_len, batch_size, input_size) if batch_first=False, otherwise (batch_size, seq_len, input_size)
        :param h_0: The initial hidden state for the input sequence, shape (num_layers * num_directions, batch_size, hidden_size)
        :return: h_n: The hidden state for t = seq_len, shape (num_layers * num_directions, batch_size, hidden_size)
        """
        if self.batch_first:
            x = x.transpose(0, 1)  # Convert to (seq_len, batch_size, input_size)
        
        seq_len, batch_size, _ = x.size()
        
        # Initialize hidden states
        h = h0.clone()
        
        # Process each time step
        for t in range(seq_len):
            x_t = x[t]  # Current input (batch_size, input_size)
            
            # Process through each layer
            for layer in range(self.num_layers):
                layer_input = x_t if layer == 0 else h[layer]
                
                weight_ih = self.weight_ih_l[layer]
                weight_hh = self.weight_hh_l[layer]
                
                if self.bias:
                    bias_ih = self.bias_ih_l[layer]
                    bias_hh = self.bias_hh_l[layer]
                else:
                    bias_ih = torch.zeros(3 * self.hidden_size, device=x.device)
                    bias_hh = torch.zeros(3 * self.hidden_size, device=x.device)
                
                # Apply custom GRU cell
                h[layer] = gru_ops.gru_cell_cuda(
                    layer_input.contiguous(),
                    h[layer].contiguous(),
                    weight_ih.contiguous(),
                    weight_hh.contiguous(),
                    bias_ih.contiguous(),
                    bias_hh.contiguous()
                )
                
                x_t = h[layer]  # Output of current layer becomes input to next layer
        
        return h