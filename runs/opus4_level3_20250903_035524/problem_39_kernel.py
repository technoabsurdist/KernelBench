import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for GRU
gru_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float tanh_activation(float x) {
    float exp2x = expf(2.0f * x);
    return (exp2x - 1.0f) / (exp2x + 1.0f);
}

__global__ void gru_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ hidden,
    const float* __restrict__ weight_ih,
    const float* __restrict__ weight_hh,
    const float* __restrict__ bias_ih,
    const float* __restrict__ bias_hh,
    float* __restrict__ output,
    float* __restrict__ new_hidden,
    int batch_size,
    int hidden_size,
    int input_size
) {
    const int batch_idx = blockIdx.y;
    const int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (hidden_idx >= hidden_size || batch_idx >= batch_size) return;
    
    // Compute gates: reset, update, new
    float gi_r = 0.0f, gi_z = 0.0f, gi_n = 0.0f;
    float gh_r = 0.0f, gh_z = 0.0f, gh_n = 0.0f;
    
    // Input contribution
    for (int i = 0; i < input_size; ++i) {
        float inp = input[batch_idx * input_size + i];
        gi_r += inp * weight_ih[hidden_idx * input_size + i];
        gi_z += inp * weight_ih[(hidden_size + hidden_idx) * input_size + i];
        gi_n += inp * weight_ih[(2 * hidden_size + hidden_idx) * input_size + i];
    }
    
    // Hidden contribution
    for (int i = 0; i < hidden_size; ++i) {
        float hid = hidden[batch_idx * hidden_size + i];
        gh_r += hid * weight_hh[hidden_idx * hidden_size + i];
        gh_z += hid * weight_hh[(hidden_size + hidden_idx) * hidden_size + i];
        gh_n += hid * weight_hh[(2 * hidden_size + hidden_idx) * hidden_size + i];
    }
    
    // Add biases and apply activations
    float r = sigmoid(gi_r + bias_ih[hidden_idx] + gh_r + bias_hh[hidden_idx]);
    float z = sigmoid(gi_z + bias_ih[hidden_size + hidden_idx] + gh_z + bias_hh[hidden_size + hidden_idx]);
    float n = tanh_activation(gi_n + bias_ih[2 * hidden_size + hidden_idx] + 
                              r * (gh_n + bias_hh[2 * hidden_size + hidden_idx]));
    
    // Compute new hidden state
    float h_prev = hidden[batch_idx * hidden_size + hidden_idx];
    float h_new = (1.0f - z) * n + z * h_prev;
    
    output[batch_idx * hidden_size + hidden_idx] = h_new;
    new_hidden[batch_idx * hidden_size + hidden_idx] = h_new;
}

std::vector<torch::Tensor> gru_forward_cuda(
    torch::Tensor input,
    torch::Tensor hidden,
    torch::Tensor weight_ih,
    torch::Tensor weight_hh,
    torch::Tensor bias_ih,
    torch::Tensor bias_hh,
    int seq_len,
    int batch_size,
    int hidden_size,
    int num_layers
) {
    auto output = torch::zeros({seq_len, batch_size, hidden_size}, input.options());
    auto h_n = hidden.clone();
    
    const int threads = 256;
    const int blocks_x = (hidden_size + threads - 1) / threads;
    dim3 blocks(blocks_x, batch_size);
    
    for (int layer = 0; layer < num_layers; ++layer) {
        auto layer_input = (layer == 0) ? input : output;
        auto layer_hidden = h_n[layer];
        
        int layer_input_size = (layer == 0) ? input.size(2) : hidden_size;
        
        auto wi = weight_ih.slice(0, layer * 3 * hidden_size, (layer + 1) * 3 * hidden_size);
        auto wh = weight_hh.slice(0, layer * 3 * hidden_size, (layer + 1) * 3 * hidden_size);
        auto bi = bias_ih.slice(0, layer * 3 * hidden_size, (layer + 1) * 3 * hidden_size);
        auto bh = bias_hh.slice(0, layer * 3 * hidden_size, (layer + 1) * 3 * hidden_size);
        
        for (int t = 0; t < seq_len; ++t) {
            auto step_input = layer_input[t];
            auto step_output = output[t];
            
            gru_forward_kernel<<<blocks, threads>>>(
                step_input.data_ptr<float>(),
                layer_hidden.data_ptr<float>(),
                wi.data_ptr<float>(),
                wh.data_ptr<float>(),
                bi.data_ptr<float>(),
                bh.data_ptr<float>(),
                step_output.data_ptr<float>(),
                layer_hidden.data_ptr<float>(),
                batch_size,
                hidden_size,
                layer_input_size
            );
            
            cudaDeviceSynchronize();
        }
        
        h_n[layer] = layer_hidden;
    }
    
    return {output, h_n};
}
"""

gru_cpp_source = """
std::vector<torch::Tensor> gru_forward_cuda(
    torch::Tensor input,
    torch::Tensor hidden,
    torch::Tensor weight_ih,
    torch::Tensor weight_hh,
    torch::Tensor bias_ih,
    torch::Tensor bias_hh,
    int seq_len,
    int batch_size,
    int hidden_size,
    int num_layers
);
"""

# Compile the inline CUDA code
gru_cuda = load_inline(
    name="gru_cuda",
    cpp_sources=gru_cpp_source,
    cuda_sources=gru_cuda_source,
    functions=["gru_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # Initialize weights and biases
        self.weight_ih = nn.Parameter(torch.randn(num_layers * 3 * hidden_size, max(input_size, hidden_size)))
        self.weight_hh = nn.Parameter(torch.randn(num_layers * 3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(num_layers * 3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(num_layers * 3 * hidden_size))
        
        # Initialize weights
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            nn.init.xavier_uniform_(self.weight_ih[layer*3*hidden_size:(layer+1)*3*hidden_size, :layer_input_size])
            nn.init.xavier_uniform_(self.weight_hh[layer*3*hidden_size:(layer+1)*3*hidden_size])
            nn.init.zeros_(self.bias_ih[layer*3*hidden_size:(layer+1)*3*hidden_size])
            nn.init.zeros_(self.bias_hh[layer*3*hidden_size:(layer+1)*3*hidden_size])
        
        self.gru_cuda = gru_cuda
    
    def forward(self, x, h0):
        if self.batch_first:
            x = x.transpose(0, 1)
        
        seq_len, batch_size, _ = x.shape
        
        # Ensure contiguous tensors
        x = x.contiguous()
        h0 = h0.contiguous()
        
        # Call custom CUDA kernel
        output, h_n = self.gru_cuda.gru_forward_cuda(
            x.cuda(),
            h0.cuda(),
            self.weight_ih.cuda(),
            self.weight_hh.cuda(),
            self.bias_ih.cuda(),
            self.bias_hh.cuda(),
            seq_len,
            batch_size,
            self.hidden_size,
            self.num_layers
        )
        
        if self.batch_first:
            output = output.transpose(0, 1)
        
        return output