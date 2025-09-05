import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for GRU
gru_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float tanh_activation(float x) {
    return tanhf(x);
}

__global__ void gru_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ h_prev,
    const float* __restrict__ weight_ih,
    const float* __restrict__ weight_hh,
    const float* __restrict__ bias_ih,
    const float* __restrict__ bias_hh,
    float* __restrict__ h_next,
    int batch_size,
    int hidden_size,
    int input_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / hidden_size;
    int hid_idx = tid % hidden_size;
    
    if (batch_idx >= batch_size || hid_idx >= hidden_size) return;
    
    // Compute gates
    float r_gate = 0.0f, z_gate = 0.0f, n_gate = 0.0f;
    
    // Input contributions
    for (int i = 0; i < input_size; ++i) {
        float x_val = input[batch_idx * input_size + i];
        r_gate += x_val * weight_ih[hid_idx * input_size + i];
        z_gate += x_val * weight_ih[(hidden_size + hid_idx) * input_size + i];
        n_gate += x_val * weight_ih[(2 * hidden_size + hid_idx) * input_size + i];
    }
    
    // Hidden state contributions
    for (int i = 0; i < hidden_size; ++i) {
        float h_val = h_prev[batch_idx * hidden_size + i];
        r_gate += h_val * weight_hh[hid_idx * hidden_size + i];
        z_gate += h_val * weight_hh[(hidden_size + hid_idx) * hidden_size + i];
    }
    
    // Add biases
    r_gate += bias_ih[hid_idx] + bias_hh[hid_idx];
    z_gate += bias_ih[hidden_size + hid_idx] + bias_hh[hidden_size + hid_idx];
    
    // Apply activations
    r_gate = sigmoid(r_gate);
    z_gate = sigmoid(z_gate);
    
    // Compute new gate with reset
    for (int i = 0; i < hidden_size; ++i) {
        float h_val = h_prev[batch_idx * hidden_size + i];
        n_gate += r_gate * h_val * weight_hh[(2 * hidden_size + hid_idx) * hidden_size + i];
    }
    n_gate += bias_ih[2 * hidden_size + hid_idx] + bias_hh[2 * hidden_size + hid_idx];
    n_gate = tanh_activation(n_gate);
    
    // Compute output
    float h_prev_val = h_prev[batch_idx * hidden_size + hid_idx];
    h_next[batch_idx * hidden_size + hid_idx] = (1.0f - z_gate) * n_gate + z_gate * h_prev_val;
}

torch::Tensor gru_forward_cuda(
    torch::Tensor input,
    torch::Tensor h_prev,
    torch::Tensor weight_ih,
    torch::Tensor weight_hh,
    torch::Tensor bias_ih,
    torch::Tensor bias_hh,
    int seq_len,
    int batch_size,
    int hidden_size,
    int input_size
) {
    auto h_out = torch::zeros({seq_len + 1, batch_size, hidden_size}, input.options());
    h_out[0] = h_prev;
    
    const int threads = 256;
    const int blocks = (batch_size * hidden_size + threads - 1) / threads;
    
    for (int t = 0; t < seq_len; ++t) {
        gru_forward_kernel<<<blocks, threads>>>(
            input[t].data_ptr<float>(),
            h_out[t].data_ptr<float>(),
            weight_ih.data_ptr<float>(),
            weight_hh.data_ptr<float>(),
            bias_ih.data_ptr<float>(),
            bias_hh.data_ptr<float>(),
            h_out[t + 1].data_ptr<float>(),
            batch_size,
            hidden_size,
            input_size
        );
    }
    
    return h_out[seq_len];
}

torch::Tensor gru_multi_layer_cuda(
    torch::Tensor input,
    torch::Tensor h0,
    torch::Tensor weight_ih_all,
    torch::Tensor weight_hh_all,
    torch::Tensor bias_ih_all,
    torch::Tensor bias_hh_all,
    int num_layers
) {
    auto sizes = input.sizes();
    int seq_len = sizes[0];
    int batch_size = sizes[1];
    int input_size = sizes[2];
    int hidden_size = h0.size(2);
    
    auto layer_input = input;
    auto h_n = torch::zeros_like(h0);
    
    for (int layer = 0; layer < num_layers; ++layer) {
        auto weight_ih = weight_ih_all[layer];
        auto weight_hh = weight_hh_all[layer];
        auto bias_ih = bias_ih_all[layer];
        auto bias_hh = bias_hh_all[layer];
        auto h_prev = h0[layer];
        
        auto h_layer = gru_forward_cuda(
            layer_input, h_prev, weight_ih, weight_hh, bias_ih, bias_hh,
            seq_len, batch_size, hidden_size, (layer == 0) ? input_size : hidden_size
        );
        
        h_n[layer] = h_layer;
        
        if (layer < num_layers - 1) {
            layer_input = torch::zeros({seq_len, batch_size, hidden_size}, input.options());
            const int threads = 256;
            const int blocks = (batch_size * hidden_size + threads - 1) / threads;
            
            for (int t = 0; t < seq_len; ++t) {
                auto h_temp = h_prev.clone();
                gru_forward_kernel<<<blocks, threads>>>(
                    (layer == 0) ? input[t].data_ptr<float>() : layer_input[t-1].data_ptr<float>(),
                    h_temp.data_ptr<float>(),
                    weight_ih.data_ptr<float>(),
                    weight_hh.data_ptr<float>(),
                    bias_ih.data_ptr<float>(),
                    bias_hh.data_ptr<float>(),
                    layer_input[t].data_ptr<float>(),
                    batch_size,
                    hidden_size,
                    (layer == 0) ? input_size : hidden_size
                );
                h_prev = layer_input[t];
            }
        }
    }
    
    return h_n;
}
"""

gru_cpp_source = """
torch::Tensor gru_multi_layer_cuda(
    torch::Tensor input,
    torch::Tensor h0,
    torch::Tensor weight_ih_all,
    torch::Tensor weight_hh_all,
    torch::Tensor bias_ih_all,
    torch::Tensor bias_hh_all,
    int num_layers
);
"""

gru_cuda = load_inline(
    name="gru_cuda",
    cpp_sources=gru_cpp_source,
    cuda_sources=gru_cuda_source,
    functions=["gru_multi_layer_cuda"],
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
        self.weight_ih_l = nn.ParameterList()
        self.weight_hh_l = nn.ParameterList()
        self.bias_ih_l = nn.ParameterList() if bias else None
        self.bias_hh_l = nn.ParameterList() if bias else None
        
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            
            w_ih = nn.Parameter(torch.randn(3 * hidden_size, layer_input_size))
            w_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
            self.weight_ih_l.append(w_ih)
            self.weight_hh_l.append(w_hh)
            
            if bias:
                b_ih = nn.Parameter(torch.randn(3 * hidden_size))
                b_hh = nn.Parameter(torch.randn(3 * hidden_size))
                self.bias_ih_l.append(b_ih)
                self.bias_hh_l.append(b_hh)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1.0 / (self.hidden_size) ** 0.5
        for weight in self.weight_ih_l:
            weight.data.uniform_(-std, std)
        for weight in self.weight_hh_l:
            weight.data.uniform_(-std, std)
        if self.bias_ih_l is not None:
            for bias in self.bias_ih_l:
                bias.data.uniform_(-std, std)
            for bias in self.bias_hh_l:
                bias.data.uniform_(-std, std)
    
    def forward(self, x, h0):
        if self.batch_first:
            x = x.transpose(0, 1)
        
        weight_ih_all = torch.stack([w.t().contiguous() for w in self.weight_ih_l])
        weight_hh_all = torch.stack([w.t().contiguous() for w in self.weight_hh_l])
        
        if self.bias_ih_l is not None:
            bias_ih_all = torch.stack(list(self.bias_ih_l))
            bias_hh_all = torch.stack(list(self.bias_hh_l))
        else:
            bias_ih_all = torch.zeros(self.num_layers, 3 * self.hidden_size, device=x.device)
            bias_hh_all = torch.zeros(self.num_layers, 3 * self.hidden_size, device=x.device)
        
        h_n = gru_cuda.gru_multi_layer_cuda(
            x.contiguous(),
            h0.contiguous(),
            weight_ih_all,
            weight_hh_all,
            bias_ih_all,
            bias_hh_all,
            self.num_layers
        )
        
        return h_n