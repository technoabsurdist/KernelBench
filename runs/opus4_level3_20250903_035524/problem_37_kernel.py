import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for LSTM cell computation
lstm_cell_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float tanhf_custom(float x) {
    float exp2x = expf(2.0f * x);
    return (exp2x - 1.0f) / (exp2x + 1.0f);
}

__global__ void lstm_cell_kernel(
    const float* __restrict__ input,
    const float* __restrict__ h_prev,
    const float* __restrict__ c_prev,
    const float* __restrict__ weight_ih,
    const float* __restrict__ weight_hh,
    const float* __restrict__ bias_ih,
    const float* __restrict__ bias_hh,
    float* __restrict__ h_out,
    float* __restrict__ c_out,
    int batch_size,
    int input_size,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / hidden_size;
    int hidden_idx = idx % hidden_size;
    
    if (batch_idx >= batch_size || hidden_idx >= hidden_size) return;
    
    // Compute gates: i, f, g, o
    float i_gate = 0, f_gate = 0, g_gate = 0, o_gate = 0;
    
    // Input contributions
    for (int j = 0; j < input_size; j++) {
        float inp = input[batch_idx * input_size + j];
        i_gate += inp * weight_ih[hidden_idx * input_size + j];
        f_gate += inp * weight_ih[(hidden_size + hidden_idx) * input_size + j];
        g_gate += inp * weight_ih[(2 * hidden_size + hidden_idx) * input_size + j];
        o_gate += inp * weight_ih[(3 * hidden_size + hidden_idx) * input_size + j];
    }
    
    // Hidden state contributions
    for (int j = 0; j < hidden_size; j++) {
        float h = h_prev[batch_idx * hidden_size + j];
        i_gate += h * weight_hh[hidden_idx * hidden_size + j];
        f_gate += h * weight_hh[(hidden_size + hidden_idx) * hidden_size + j];
        g_gate += h * weight_hh[(2 * hidden_size + hidden_idx) * hidden_size + j];
        o_gate += h * weight_hh[(3 * hidden_size + hidden_idx) * hidden_size + j];
    }
    
    // Add biases
    i_gate += bias_ih[hidden_idx] + bias_hh[hidden_idx];
    f_gate += bias_ih[hidden_size + hidden_idx] + bias_hh[hidden_size + hidden_idx];
    g_gate += bias_ih[2 * hidden_size + hidden_idx] + bias_hh[2 * hidden_size + hidden_idx];
    o_gate += bias_ih[3 * hidden_size + hidden_idx] + bias_hh[3 * hidden_size + hidden_idx];
    
    // Apply activations
    i_gate = sigmoidf(i_gate);
    f_gate = sigmoidf(f_gate);
    g_gate = tanhf_custom(g_gate);
    o_gate = sigmoidf(o_gate);
    
    // Compute new cell state and hidden state
    float c_new = f_gate * c_prev[batch_idx * hidden_size + hidden_idx] + i_gate * g_gate;
    float h_new = o_gate * tanhf_custom(c_new);
    
    c_out[batch_idx * hidden_size + hidden_idx] = c_new;
    h_out[batch_idx * hidden_size + hidden_idx] = h_new;
}

std::vector<torch::Tensor> lstm_forward_cuda(
    torch::Tensor input,
    torch::Tensor h0,
    torch::Tensor c0,
    torch::Tensor weight_ih,
    torch::Tensor weight_hh,
    torch::Tensor bias_ih,
    torch::Tensor bias_hh,
    int num_layers
) {
    auto batch_size = input.size(0);
    auto seq_length = input.size(1);
    auto input_size = input.size(2);
    auto hidden_size = h0.size(2);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto output = torch::zeros({batch_size, seq_length, hidden_size}, options);
    auto h_out = h0.clone();
    auto c_out = c0.clone();
    
    const int threads = 256;
    const int blocks = (batch_size * hidden_size + threads - 1) / threads;
    
    for (int layer = 0; layer < num_layers; layer++) {
        auto h_prev = h_out[layer].clone();
        auto c_prev = c_out[layer].clone();
        
        for (int t = 0; t < seq_length; t++) {
            auto input_t = (layer == 0) ? input.select(1, t) : output.select(1, t);
            
            lstm_cell_kernel<<<blocks, threads>>>(
                input_t.data_ptr<float>(),
                h_prev.data_ptr<float>(),
                c_prev.data_ptr<float>(),
                weight_ih[layer].data_ptr<float>(),
                weight_hh[layer].data_ptr<float>(),
                bias_ih[layer].data_ptr<float>(),
                bias_hh[layer].data_ptr<float>(),
                h_prev.data_ptr<float>(),
                c_prev.data_ptr<float>(),
                batch_size,
                (layer == 0) ? input_size : hidden_size,
                hidden_size
            );
            
            if (layer == num_layers - 1) {
                output.select(1, t).copy_(h_prev);
            }
        }
        
        h_out[layer] = h_prev;
        c_out[layer] = c_prev;
    }
    
    return {output, h_out, c_out};
}
"""

lstm_cpp_source = """
std::vector<torch::Tensor> lstm_forward_cuda(
    torch::Tensor input,
    torch::Tensor h0,
    torch::Tensor c0,
    torch::Tensor weight_ih,
    torch::Tensor weight_hh,
    torch::Tensor bias_ih,
    torch::Tensor bias_hh,
    int num_layers
);
"""

lstm_cuda = load_inline(
    name="lstm_cuda",
    cpp_sources=lstm_cpp_source,
    cuda_sources=lstm_cell_source,
    functions=["lstm_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Initialize LSTM weights
        self.weight_ih = nn.ParameterList([
            nn.Parameter(torch.randn(4 * hidden_size, input_size if i == 0 else hidden_size) * 0.01)
            for i in range(num_layers)
        ])
        self.weight_hh = nn.ParameterList([
            nn.Parameter(torch.randn(4 * hidden_size, hidden_size) * 0.01)
            for i in range(num_layers)
        ])
        self.bias_ih = nn.ParameterList([
            nn.Parameter(torch.zeros(4 * hidden_size))
            for i in range(num_layers)
        ])
        self.bias_hh = nn.ParameterList([
            nn.Parameter(torch.zeros(4 * hidden_size))
            for i in range(num_layers)
        ])
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.lstm_cuda = lstm_cuda
        
    def forward(self, x, h0, c0):
        x = x.cuda()
        h0 = h0.cuda()
        c0 = c0.cuda()
        
        weight_ih = torch.stack([w.cuda() for w in self.weight_ih])
        weight_hh = torch.stack([w.cuda() for w in self.weight_hh])
        bias_ih = torch.stack([b.cuda() for b in self.bias_ih])
        bias_hh = torch.stack([b.cuda() for b in self.bias_hh])
        
        output, h_out, c_out = self.lstm_cuda.lstm_forward_cuda(
            x.contiguous(),
            h0.contiguous(),
            c0.contiguous(),
            weight_ih.contiguous(),
            weight_hh.contiguous(),
            bias_ih.contiguous(),
            bias_hh.contiguous(),
            self.num_layers
        )
        
        out = self.fc(output[:, -1, :])
        
        return c_out