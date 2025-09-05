import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused bidirectional GRU
gru_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float tanh_activation(float x) {
    return tanhf(x);
}

__global__ void gru_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight_ih,
    const float* __restrict__ weight_hh,
    const float* __restrict__ bias_ih,
    const float* __restrict__ bias_hh,
    const float* __restrict__ h_prev,
    float* __restrict__ h_next,
    int batch_size,
    int hidden_size,
    int input_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / hidden_size;
    int hid_idx = tid % hidden_size;
    
    if (batch_idx < batch_size && hid_idx < hidden_size) {
        // Compute gates
        float r_gate = 0.0f, z_gate = 0.0f, n_gate = 0.0f;
        
        // Input contribution
        for (int i = 0; i < input_size; i++) {
            float x_val = input[batch_idx * input_size + i];
            r_gate += x_val * weight_ih[hid_idx * input_size * 3 + i];
            z_gate += x_val * weight_ih[(hidden_size + hid_idx) * input_size * 3 + i];
            n_gate += x_val * weight_ih[(2 * hidden_size + hid_idx) * input_size * 3 + i];
        }
        
        // Add bias_ih
        r_gate += bias_ih[hid_idx];
        z_gate += bias_ih[hidden_size + hid_idx];
        n_gate += bias_ih[2 * hidden_size + hid_idx];
        
        // Hidden state contribution for r and z gates
        float h_r = 0.0f, h_z = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            float h_val = h_prev[batch_idx * hidden_size + i];
            h_r += h_val * weight_hh[hid_idx * hidden_size * 3 + i];
            h_z += h_val * weight_hh[(hidden_size + hid_idx) * hidden_size * 3 + i];
        }
        
        r_gate += h_r + bias_hh[hid_idx];
        z_gate += h_z + bias_hh[hidden_size + hid_idx];
        
        // Apply sigmoid activation
        r_gate = sigmoid(r_gate);
        z_gate = sigmoid(z_gate);
        
        // Compute new gate with reset
        float h_n = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            h_n += r_gate * h_prev[batch_idx * hidden_size + i] * weight_hh[(2 * hidden_size + hid_idx) * hidden_size * 3 + i];
        }
        n_gate += h_n + bias_hh[2 * hidden_size + hid_idx];
        n_gate = tanh_activation(n_gate);
        
        // Update hidden state
        h_next[batch_idx * hidden_size + hid_idx] = (1 - z_gate) * n_gate + z_gate * h_prev[batch_idx * hidden_size + hid_idx];
    }
}

std::vector<torch::Tensor> gru_bidirectional_cuda(
    torch::Tensor input,
    torch::Tensor h0,
    torch::Tensor weight_ih_l,
    torch::Tensor weight_hh_l,
    torch::Tensor bias_ih_l,
    torch::Tensor bias_hh_l
) {
    auto seq_len = input.size(0);
    auto batch_size = input.size(1);
    auto input_size = input.size(2);
    auto num_layers = h0.size(0) / 2;
    auto hidden_size = h0.size(2);
    
    auto output = torch::zeros({seq_len, batch_size, hidden_size * 2}, input.options());
    auto h_n = h0.clone();
    
    const int block_size = 256;
    const int num_blocks = (batch_size * hidden_size + block_size - 1) / block_size;
    
    // Process all layers
    for (int layer = 0; layer < num_layers; layer++) {
        auto layer_input = (layer == 0) ? input : output.slice(2, 0, hidden_size * 2);
        auto h_forward = torch::zeros({batch_size, hidden_size}, input.options());
        auto h_backward = torch::zeros({batch_size, hidden_size}, input.options());
        
        // Initialize hidden states
        h_forward.copy_(h0[layer * 2]);
        h_backward.copy_(h0[layer * 2 + 1]);
        
        // Forward direction
        for (int t = 0; t < seq_len; t++) {
            gru_forward_kernel<<<num_blocks, block_size>>>(
                layer_input[t].data_ptr<float>(),
                weight_ih_l[layer * 2].data_ptr<float>(),
                weight_hh_l[layer * 2].data_ptr<float>(),
                bias_ih_l[layer * 2].data_ptr<float>(),
                bias_hh_l[layer * 2].data_ptr<float>(),
                h_forward.data_ptr<float>(),
                h_forward.data_ptr<float>(),
                batch_size,
                hidden_size,
                (layer == 0) ? input_size : hidden_size * 2
            );
            output[t].slice(1, 0, hidden_size).copy_(h_forward);
        }
        
        // Backward direction
        for (int t = seq_len - 1; t >= 0; t--) {
            gru_forward_kernel<<<num_blocks, block_size>>>(
                layer_input[t].data_ptr<float>(),
                weight_ih_l[layer * 2 + 1].data_ptr<float>(),
                weight_hh_l[layer * 2 + 1].data_ptr<float>(),
                bias_ih_l[layer * 2 + 1].data_ptr<float>(),
                bias_hh_l[layer * 2 + 1].data_ptr<float>(),
                h_backward.data_ptr<float>(),
                h_backward.data_ptr<float>(),
                batch_size,
                hidden_size,
                (layer == 0) ? input_size : hidden_size * 2
            );
            output[t].slice(1, hidden_size, hidden_size * 2).copy_(h_backward);
        }
        
        h_n[layer * 2].copy_(h_forward);
        h_n[layer * 2 + 1].copy_(h_backward);
    }
    
    return {output, h_n};
}
"""

gru_cpp_source = """
std::vector<torch::Tensor> gru_bidirectional_cuda(
    torch::Tensor input,
    torch::Tensor h0,
    torch::Tensor weight_ih_l,
    torch::Tensor weight_hh_l,
    torch::Tensor bias_ih_l,
    torch::Tensor bias_hh_l
);
"""

# Compile the inline CUDA code
gru_bidirectional = load_inline(
    name="gru_bidirectional",
    cpp_sources=gru_cpp_source,
    cuda_sources=gru_cuda_source,
    functions=["gru_bidirectional_cuda"],
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
        self.bias = bias
        self.batch_first = batch_first
        
        # Initialize weights and biases for all layers
        self.weight_ih_l = nn.ParameterList()
        self.weight_hh_l = nn.ParameterList()
        self.bias_ih_l = nn.ParameterList() if bias else None
        self.bias_hh_l = nn.ParameterList() if bias else None
        
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * 2
            
            # Forward direction weights
            w_ih_fw = nn.Parameter(torch.randn(3 * hidden_size, layer_input_size))
            w_hh_fw = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
            self.weight_ih_l.append(w_ih_fw)
            self.weight_hh_l.append(w_hh_fw)
            
            # Backward direction weights
            w_ih_bw = nn.Parameter(torch.randn(3 * hidden_size, layer_input_size))
            w_hh_bw = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
            self.weight_ih_l.append(w_ih_bw)
            self.weight_hh_l.append(w_hh_bw)
            
            if bias:
                b_ih_fw = nn.Parameter(torch.randn(3 * hidden_size))
                b_hh_fw = nn.Parameter(torch.randn(3 * hidden_size))
                self.bias_ih_l.append(b_ih_fw)
                self.bias_hh_l.append(b_hh_fw)
                
                b_ih_bw = nn.Parameter(torch.randn(3 * hidden_size))
                b_hh_bw = nn.Parameter(torch.randn(3 * hidden_size))
                self.bias_ih_l.append(b_ih_bw)
                self.bias_hh_l.append(b_hh_bw)
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1.0 / (self.hidden_size) ** 0.5
        for weight in self.weight_ih_l:
            weight.data.uniform_(-std, std)
        for weight in self.weight_hh_l:
            weight.data.uniform_(-std, std)
        if self.bias:
            for bias in self.bias_ih_l:
                bias.data.uniform_(-std, std)
            for bias in self.bias_hh_l:
                bias.data.uniform_(-std, std)
    
    def forward(self, x, h0):
        if self.batch_first:
            x = x.transpose(0, 1)
        
        weight_ih = torch.stack(list(self.weight_ih_l))
        weight_hh = torch.stack(list(self.weight_hh_l))
        
        if self.bias:
            bias_ih = torch.stack(list(self.bias_ih_l))
            bias_hh = torch.stack(list(self.bias_hh_l))
        else:
            bias_ih = torch.zeros_like(weight_ih[:, :, 0])
            bias_hh = torch.zeros_like(weight_hh[:, :, 0])
        
        output, h_n = gru_bidirectional.gru_bidirectional_cuda(
            x.contiguous(), 
            h0.contiguous(), 
            weight_ih.contiguous(), 
            weight_hh.contiguous(), 
            bias_ih.contiguous(), 
            bias_hh.contiguous()
        )
        
        if self.batch_first:
            output = output.transpose(0, 1)
        
        return h_n

batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

def get_inputs():
    return [torch.rand(seq_len, batch_size, input_size).cuda(), torch.rand((num_layers*2, batch_size, hidden_size)).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, num_layers]