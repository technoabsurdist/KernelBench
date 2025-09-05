import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused GRU operations
gru_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float sigmoid_gpu(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float tanh_gpu(float x) {
    float exp2x = expf(2.0f * x);
    return (exp2x - 1.0f) / (exp2x + 1.0f);
}

__global__ void gru_cell_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ hidden,
    const float* __restrict__ weight_ih,
    const float* __restrict__ weight_hh,
    const float* __restrict__ bias_ih,
    const float* __restrict__ bias_hh,
    float* __restrict__ output,
    int batch_size,
    int input_size,
    int hidden_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / hidden_size;
    int hid_idx = tid % hidden_size;
    
    if (batch_idx >= batch_size || hid_idx >= hidden_size) return;
    
    // Compute reset, update, and new gates
    float r_i = 0.0f, z_i = 0.0f, n_i = 0.0f;
    float r_h = 0.0f, z_h = 0.0f, n_h = 0.0f;
    
    // Input contributions
    for (int i = 0; i < input_size; ++i) {
        float in_val = input[batch_idx * input_size + i];
        r_i += in_val * weight_ih[hid_idx * input_size + i];
        z_i += in_val * weight_ih[(hidden_size + hid_idx) * input_size + i];
        n_i += in_val * weight_ih[(2 * hidden_size + hid_idx) * input_size + i];
    }
    
    // Hidden contributions
    for (int i = 0; i < hidden_size; ++i) {
        float hid_val = hidden[batch_idx * hidden_size + i];
        r_h += hid_val * weight_hh[hid_idx * hidden_size + i];
        z_h += hid_val * weight_hh[(hidden_size + hid_idx) * hidden_size + i];
        n_h += hid_val * weight_hh[(2 * hidden_size + hid_idx) * hidden_size + i];
    }
    
    // Add biases and apply activations
    float r = sigmoid_gpu(r_i + bias_ih[hid_idx] + r_h + bias_hh[hid_idx]);
    float z = sigmoid_gpu(z_i + bias_ih[hidden_size + hid_idx] + z_h + bias_hh[hidden_size + hid_idx]);
    float n = tanh_gpu(n_i + bias_ih[2 * hidden_size + hid_idx] + r * (n_h + bias_hh[2 * hidden_size + hid_idx]));
    
    // Compute new hidden state
    float h_prev = hidden[batch_idx * hidden_size + hid_idx];
    output[batch_idx * hidden_size + hid_idx] = (1.0f - z) * n + z * h_prev;
}

torch::Tensor gru_forward_cuda(
    torch::Tensor input,
    torch::Tensor hidden,
    torch::Tensor weight_ih,
    torch::Tensor weight_hh,
    torch::Tensor bias_ih,
    torch::Tensor bias_hh,
    int batch_size,
    int input_size,
    int hidden_size
) {
    auto output = torch::zeros_like(hidden);
    
    const int threads = 256;
    const int blocks = (batch_size * hidden_size + threads - 1) / threads;
    
    gru_cell_forward_kernel<<<blocks, threads>>>(
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
torch::Tensor gru_forward_cuda(
    torch::Tensor input,
    torch::Tensor hidden,
    torch::Tensor weight_ih,
    torch::Tensor weight_hh,
    torch::Tensor bias_ih,
    torch::Tensor bias_hh,
    int batch_size,
    int input_size,
    int hidden_size
);
"""

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
        self.bidirectional = True
        
        # Initialize weights and biases for each layer and direction
        self.weight_ih_l = nn.ParameterList()
        self.weight_hh_l = nn.ParameterList()
        self.bias_ih_l = nn.ParameterList() if bias else None
        self.bias_hh_l = nn.ParameterList() if bias else None
        
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * 2
            
            # Forward direction
            w_ih = nn.Parameter(torch.randn(3 * hidden_size, layer_input_size) * 0.01)
            w_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size) * 0.01)
            self.weight_ih_l.append(w_ih)
            self.weight_hh_l.append(w_hh)
            
            # Backward direction
            w_ih_reverse = nn.Parameter(torch.randn(3 * hidden_size, layer_input_size) * 0.01)
            w_hh_reverse = nn.Parameter(torch.randn(3 * hidden_size, hidden_size) * 0.01)
            self.weight_ih_l.append(w_ih_reverse)
            self.weight_hh_l.append(w_hh_reverse)
            
            if bias:
                b_ih = nn.Parameter(torch.zeros(3 * hidden_size))
                b_hh = nn.Parameter(torch.zeros(3 * hidden_size))
                b_ih_reverse = nn.Parameter(torch.zeros(3 * hidden_size))
                b_hh_reverse = nn.Parameter(torch.zeros(3 * hidden_size))
                self.bias_ih_l.append(b_ih)
                self.bias_hh_l.append(b_hh)
                self.bias_ih_l.append(b_ih_reverse)
                self.bias_hh_l.append(b_hh_reverse)
        
        self.gru_cuda = gru_cuda
    
    def forward(self, x, h0):
        if self.batch_first:
            x = x.transpose(0, 1)
        
        seq_len, batch_size, _ = x.size()
        
        # Process through layers
        layer_input = x
        hidden_states = []
        
        for layer in range(self.num_layers):
            # Forward direction
            h_forward = h0[layer * 2].contiguous()
            forward_output = []
            
            for t in range(seq_len):
                h_forward = self.gru_cuda.gru_forward_cuda(
                    layer_input[t].contiguous(),
                    h_forward,
                    self.weight_ih_l[layer * 2].t().contiguous(),
                    self.weight_hh_l[layer * 2].t().contiguous(),
                    self.bias_ih_l[layer * 2] if self.bias_ih_l else torch.zeros(3 * self.hidden_size, device=x.device),
                    self.bias_hh_l[layer * 2] if self.bias_hh_l else torch.zeros(3 * self.hidden_size, device=x.device),
                    batch_size,
                    self.input_size if layer == 0 else self.hidden_size * 2,
                    self.hidden_size
                )
                forward_output.append(h_forward)
            
            # Backward direction
            h_backward = h0[layer * 2 + 1].contiguous()
            backward_output = []
            
            for t in range(seq_len - 1, -1, -1):
                h_backward = self.gru_cuda.gru_forward_cuda(
                    layer_input[t].contiguous(),
                    h_backward,
                    self.weight_ih_l[layer * 2 + 1].t().contiguous(),
                    self.weight_hh_l[layer * 2 + 1].t().contiguous(),
                    self.bias_ih_l[layer * 2 + 1] if self.bias_ih_l else torch.zeros(3 * self.hidden_size, device=x.device),
                    self.bias_hh_l[layer * 2 + 1] if self.bias_hh_l else torch.zeros(3 * self.hidden_size, device=x.device),
                    batch_size,
                    self.input_size if layer == 0 else self.hidden_size * 2,
                    self.hidden_size
                )
                backward_output.insert(0, h_backward)
            
            # Concatenate forward and backward outputs
            layer_output = []
            for fwd, bwd in zip(forward_output, backward_output):
                layer_output.append(torch.cat([fwd, bwd], dim=-1))
            
            layer_input = torch.stack(layer_output)
            hidden_states.append(h_forward)
            hidden_states.append(h_backward)
        
        output = layer_input
        
        if self.batch_first:
            output = output.transpose(0, 1)
        
        return output

batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

def get_inputs():
    return [torch.rand(seq_len, batch_size, input_size).cuda(), torch.rand((num_layers*2, batch_size, hidden_size)).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, num_layers]