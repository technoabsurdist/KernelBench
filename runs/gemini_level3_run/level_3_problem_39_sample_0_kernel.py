import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for the element-wise part of the GRU cell
# This kernel fuses the gate activations and the final hidden state calculation
gru_cell_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Device function for sigmoid
__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// The kernel that performs the fused element-wise operations for a GRU cell
__global__ void gru_cell_kernel(
    const float* __restrict__ gates_ih,
    const float* __restrict__ gates_hh,
    const float* __restrict__ h_prev,
    float* __restrict__ h_next,
    const int batch_size,
    const int hidden_size) {

    const int total_threads = batch_size * hidden_size;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_threads) {
        const int batch_idx = idx / hidden_size;
        const int hidden_idx = idx % hidden_size;

        // Calculate linear indices for the gate tensors (which have shape [batch_size, 3 * hidden_size])
        const int gates_stride = 3 * hidden_size;
        const int r_idx = batch_idx * gates_stride + hidden_idx;
        const int z_idx = r_idx + hidden_size;
        const int n_idx = z_idx + hidden_size;

        // Extract gate values from the pre-computed matrix products
        const float r_ih = gates_ih[r_idx];
        const float z_ih = gates_ih[z_idx];
        const float n_ih = gates_ih[n_idx];

        const float r_hh = gates_hh[r_idx];
        const float z_hh = gates_hh[z_idx];
        const float n_hh = gates_hh[n_idx];

        // Compute reset and update gates
        const float r_t = sigmoidf(r_ih + r_hh);
        const float z_t = sigmoidf(z_ih + z_hh);

        // Compute new gate candidate
        const float n_t = tanhf(n_ih + r_t * n_hh);

        // Compute the next hidden state
        const float h_prev_val = h_prev[idx];
        h_next[idx] = (1.0f - z_t) * n_t + z_t * h_prev_val;
    }
}

// C++ wrapper function that orchestrates the GRU cell computation
torch::Tensor gru_cell_forward_cuda(
    torch::Tensor x_t,
    torch::Tensor h_prev,
    torch::Tensor weight_ih,
    torch::Tensor weight_hh,
    torch::Tensor bias_ih,
    torch::Tensor bias_hh) {

    // Input validation
    TORCH_CHECK(x_t.is_cuda(), "x_t must be a CUDA tensor");
    TORCH_CHECK(h_prev.is_cuda(), "h_prev must be a CUDA tensor");
    TORCH_CHECK(weight_ih.is_cuda(), "weight_ih must be a CUDA tensor");
    TORCH_CHECK(weight_hh.is_cuda(), "weight_hh must be a CUDA tensor");

    TORCH_CHECK(x_t.is_contiguous(), "x_t must be contiguous");
    TORCH_CHECK(h_prev.is_contiguous(), "h_prev must be contiguous");
    TORCH_CHECK(weight_ih.is_contiguous(), "weight_ih must be contiguous");
    TORCH_CHECK(weight_hh.is_contiguous(), "weight_hh must be contiguous");

    const auto batch_size = x_t.size(0);
    const auto hidden_size = h_prev.size(1);

    // Step 1: Perform the batched matrix multiplications (GEMMs)
    // These are compute-bound and are best handled by cuBLAS (which backs torch.addmm)
    torch::Tensor gates_ih, gates_hh;
    if (bias_ih.defined()) {
        TORCH_CHECK(bias_ih.is_cuda(), "bias_ih must be a CUDA tensor");
        TORCH_CHECK(bias_hh.is_cuda(), "bias_hh must be a CUDA tensor");
        TORCH_CHECK(bias_ih.is_contiguous(), "bias_ih must be contiguous");
        TORCH_CHECK(bias_hh.is_contiguous(), "bias_hh must be contiguous");
        gates_ih = torch::addmm(bias_ih, x_t, weight_ih.t());
        gates_hh = torch::addmm(bias_hh, h_prev, weight_hh.t());
    } else {
        gates_ih = torch::matmul(x_t, weight_ih.t());
        gates_hh = torch::matmul(h_prev, weight_hh.t());
    }

    // Step 2: Launch the custom kernel to perform the fused element-wise operations
    // This part is memory-bandwidth bound, and fusion helps by reducing memory I/O
    auto h_next = torch::empty_like(h_prev);

    const int block_size = 256;
    const int num_blocks = (batch_size * hidden_size + block_size - 1) / block_size;

    gru_cell_kernel<<<num_blocks, block_size>>>(
        gates_ih.data_ptr<float>(),
        gates_hh.data_ptr<float>(),
        h_prev.data_ptr<float>(),
        h_next.data_ptr<float>(),
        batch_size,
        hidden_size);

    return h_next;
}
"""

gru_cell_cpp_source = """
torch::Tensor gru_cell_forward_cuda(
    torch::Tensor x_t,
    torch::Tensor h_prev,
    torch::Tensor weight_ih,
    torch::Tensor weight_hh,
    torch::Tensor bias_ih,
    torch::Tensor bias_hh);
"""

# JIT compile the CUDA and C++ code
gru_cell = load_inline(
    name="gru_cell",
    cpp_sources=gru_cell_cpp_source,
    cuda_sources=gru_cell_source,
    functions=["gru_cell_forward_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        Custom GRU implementation using a fused CUDA kernel for the cell.
        :param input_size: The number of expected features in the input x
        :param hidden_size: The number of features in the hidden state h
        :param num_layers: Number of recurrent layers.
        :param bias: If False, then the layer does not use bias weights.
        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature).
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        self.weight_ih_l = nn.ParameterList()
        self.weight_hh_l = nn.ParameterList()
        
        if bias:
            self.bias_ih_l = nn.ParameterList()
            self.bias_hh_l = nn.ParameterList()
        else:
            self.register_parameter('bias_ih_l', None)
            self.register_parameter('bias_hh_l', None)

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            
            # Input-to-hidden weights
            W_ih = nn.Parameter(torch.Tensor(3 * hidden_size, layer_input_size))
            self.weight_ih_l.append(W_ih)

            # Hidden-to-hidden weights
            W_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
            self.weight_hh_l.append(W_hh)

            if bias:
                b_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
                b_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
                self.bias_ih_l.append(b_ih)
                self.bias_hh_l.append(b_hh)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, h0):
        """
        :param x: The input tensor, shape (seq_len, batch_size, input_size) or (batch_size, seq_len, input_size)
        :param h0: The initial hidden state, shape (num_layers, batch_size, hidden_size)
        :return: output, h_n
        """
        is_batched = x.dim() == 3
        if not is_batched:
             x = x.unsqueeze(1)
             if h0 is not None:
                 h0 = h0.unsqueeze(1)

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        
        hidden_states = list(torch.unbind(h0, dim=0))
        
        current_input = x
        
        for i in range(self.num_layers):
            h = hidden_states[i]
            output_seq = []
            
            w_ih = self.weight_ih_l[i]
            w_hh = self.weight_hh_l[i]
            b_ih = self.bias_ih_l[i] if self.bias else torch.Tensor().to(x.device)
            b_hh = self.bias_hh_l[i] if self.bias else torch.Tensor().to(x.device)
            
            for t in range(seq_len):
                x_t = current_input[t]
                h = gru_cell.gru_cell_forward_cuda(x_t, h, w_ih, w_hh, b_ih, b_hh)
                output_seq.append(h)
            
            current_input = torch.stack(output_seq, dim=0)
            hidden_states[i] = h
            
        output = current_input
        h_n = torch.stack(hidden_states, dim=0)
        
        if self.batch_first:
            output = output.transpose(0, 1)
        
        if not is_batched:
            output = output.squeeze(1)
            h_n = h_n.squeeze(1)
            
        return output, h_n