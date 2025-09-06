import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel and C++ wrapper for a bidirectional GRU
# The C++ part handles the loops over layers and time steps to minimize Python overhead.
# The CUDA part implements a fused cell operation to combine multiple element-wise
# operations into a single kernel launch, reducing overhead and improving memory access patterns.
cuda_sources = r"""
#include <torch/extension.hh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Device function for sigmoid
__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// The fused GRU cell kernel.
// This kernel computes the GRU update for an entire batch at a single time step.
// It fuses the gate calculations (sigmoid, tanh) and the final hidden state update.
__global__ void gru_fused_cell_kernel(
    const float* __restrict__ gates_ih,      // Precomputed: x_t @ W_ih^T + b_ih
    const float* __restrict__ gates_hh,      // Precomputed: h_{t-1} @ W_hh^T + b_hh
    const float* __restrict__ hidden_prev,   // h_{t-1}
    float* hidden_next,                      // h_t (output)
    int batch_size,
    int hidden_size
) {
    // Each thread computes one element in the batch x hidden_size matrix
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * hidden_size) {
        return;
    }

    int batch_idx = idx / hidden_size;
    int hidden_idx = idx % hidden_size;

    // Pointers to the start of the current batch item's data
    const float* gates_ih_b = gates_ih + batch_idx * 3 * hidden_size;
    const float* gates_hh_b = gates_hh + batch_idx * 3 * hidden_size;
    const float* hidden_prev_b = hidden_prev + batch_idx * hidden_size;
    float* hidden_next_b = hidden_next + batch_idx * hidden_size;

    // Calculate reset gate (r_t)
    float r_val = gates_ih_b[hidden_idx] + gates_hh_b[hidden_idx];
    float r = sigmoidf(r_val);

    // Calculate update gate (z_t)
    float z_val = gates_ih_b[hidden_size + hidden_idx] + gates_hh_b[hidden_size + hidden_idx];
    float z = sigmoidf(z_val);

    // Calculate new gate (n_t)
    float n_val = gates_ih_b[2 * hidden_size + hidden_idx] + r * gates_hh_b[2 * hidden_size + hidden_idx];
    float n = tanhf(n_val);

    // Calculate next hidden state (h_t)
    hidden_next_b[hidden_idx] = (1.0f - z) * n + z * hidden_prev_b[hidden_idx];
}

// C++ function to launch the CUDA kernel
void gru_fused_cell_cuda_launcher(
    torch::Tensor gates_ih,
    torch::Tensor gates_hh,
    torch::Tensor hidden_prev,
    torch::Tensor hidden_next
) {
    const int batch_size = hidden_prev.size(0);
    const int hidden_size = hidden_prev.size(1);

    const int block_size = 256;
    const int num_blocks = (batch_size * hidden_size + block_size - 1) / block_size;

    gru_fused_cell_kernel<<<num_blocks, block_size>>>(
        gates_ih.data_ptr<float>(),
        gates_hh.data_ptr<float>(),
        hidden_prev.data_ptr<float>(),
        hidden_next.data_ptr<float>(),
        batch_size,
        hidden_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Main C++ function callable from Python. This function orchestrates the entire GRU computation.
std::vector<torch::Tensor> gru_forward(
    torch::Tensor input,
    torch::Tensor h_0,
    std::vector<torch::Tensor> weights, // Flat list of all weights and biases
    bool batch_first
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(h_0.is_cuda(), "h_0 must be a CUDA tensor");

    if (batch_first) {
        input = input.permute({1, 0, 2});
    }

    const auto seq_len = input.size(0);
    const auto batch_size = input.size(1);
    const auto hidden_size = h_0.size(2);
    const auto num_layers = h_0.size(0) / 2; // 2 for bidirectional

    auto current_input = input;
    auto h_n = torch::empty_like(h_0);

    for (int layer = 0; layer < num_layers; ++layer) {
        // 8 tensors per layer: w_ih, w_hh, b_ih, b_hh (fwd), and same for (bwd)
        const int weight_idx = layer * 8;
        auto w_ih_fwd = weights[weight_idx + 0];
        auto w_hh_fwd = weights[weight_idx + 1];
        auto b_ih_fwd = weights[weight_idx + 2];
        auto b_hh_fwd = weights[weight_idx + 3];
        auto w_ih_bwd = weights[weight_idx + 4];
        auto w_hh_bwd = weights[weight_idx + 5];
        auto b_ih_bwd = weights[weight_idx + 6];
        auto b_hh_bwd = weights[weight_idx + 7];

        auto h_fwd = h_0.index({layer * 2});
        auto h_bwd = h_0.index({layer * 2 + 1});

        auto layer_output_fwd = torch::empty({seq_len, batch_size, hidden_size}, input.options());
        auto layer_output_bwd = torch::empty({seq_len, batch_size, hidden_size}, input.options());

        // Forward direction
        for (int t = 0; t < seq_len; ++t) {
            auto x_t = current_input.index({t});
            auto gates_ih = torch::addmm(b_ih_fwd, x_t, w_ih_fwd.t());
            auto gates_hh = torch::addmm(b_hh_fwd, h_fwd, w_hh_fwd.t());
            auto h_next = torch::empty_like(h_fwd);
            gru_fused_cell_cuda_launcher(gates_ih, gates_hh, h_fwd, h_next);
            h_fwd = h_next;
            layer_output_fwd.index_put_({t}, h_fwd);
        }
        h_n.index_put_({layer * 2}, h_fwd);

        // Backward direction
        for (int t = seq_len - 1; t >= 0; --t) {
            auto x_t = current_input.index({t});
            auto gates_ih = torch::addmm(b_ih_bwd, x_t, w_ih_bwd.t());
            auto gates_hh = torch::addmm(b_hh_bwd, h_bwd, w_hh_bwd.t());
            auto h_next = torch::empty_like(h_bwd);
            gru_fused_cell_cuda_launcher(gates_ih, gates_hh, h_bwd, h_next);
            h_bwd = h_next;
            layer_output_bwd.index_put_({t}, h_bwd);
        }
        h_n.index_put_({layer * 2 + 1}, h_bwd);

        current_input = torch::cat({layer_output_fwd, layer_output_bwd}, 2);
    }

    if (batch_first) {
        current_input = current_input.permute({1, 0, 2});
    }

    return {current_input, h_n};
}
"""

# Define the C++ function signature for PyTorch's JIT compiler
cpp_wrapper_source = """
std::vector<torch::Tensor> gru_forward(
    torch::Tensor input,
    torch::Tensor h_0,
    std::vector<torch::Tensor> weights,
    bool batch_first
);
"""

# Compile the inline CUDA code. This happens once when the Python module is imported.
custom_gru_module = load_inline(
    name="custom_gru_module",
    cpp_sources=cpp_wrapper_source,
    cuda_sources=cuda_sources,
    functions=["gru_forward"],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        A custom implementation of a bidirectional GRU layer using a fused CUDA kernel.
        This module mimics the interface of torch.nn.GRU.
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = True  # Hardcoded from original model
        num_directions = 2 if self.bidirectional else 1

        self.all_weights = nn.ParameterList()
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = self.input_size if layer == 0 else self.hidden_size * num_directions
                
                # Weights for input-hidden connections (W_ir, W_iz, W_in)
                w_ih = nn.Parameter(torch.Tensor(3 * self.hidden_size, layer_input_size))
                # Weights for hidden-hidden connections (W_hr, W_hz, W_hn)
                w_hh = nn.Parameter(torch.Tensor(3 * self.hidden_size, self.hidden_size))
                
                self.all_weights.append(w_ih)
                self.all_weights.append(w_hh)

                if self.bias:
                    b_ih = nn.Parameter(torch.Tensor(3 * self.hidden_size))
                    b_hh = nn.Parameter(torch.Tensor(3 * self.hidden_size))
                    self.all_weights.append(b_ih)
                    self.all_weights.append(b_hh)
                else:
                    # Add placeholder tensors if no bias, to keep the C++ interface simple
                    self.all_weights.append(torch.zeros(3 * self.hidden_size))
                    self.all_weights.append(torch.zeros(3 * self.hidden_size))

        self.reset_parameters()
        self.custom_gru = custom_gru_module

    def reset_parameters(self):
        """Initialize weights and biases uniformly."""
        std_v = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -std_v, std_v)

    def forward(self, x, h0):
        """
        :param x: The input tensor.
        :param h0: The initial hidden state.
        :return: output, h_n
        """
        # The C++ code expects a flat list of tensors.
        # The order is: w_ih_l0, w_hh_l0, b_ih_l0, b_hh_l0, w_ih_l0_b, w_hh_l0_b, ...
        
        # If no bias, the placeholder tensors need to be on the correct device
        flat_weights = []
        for p in self.all_weights:
            if isinstance(p, nn.Parameter):
                flat_weights.append(p)
            else:  # It's a placeholder zero tensor for the no-bias case
                flat_weights.append(p.to(x.device, x.dtype))

        output, h_n = self.custom_gru.gru_forward(x, h0, flat_weights, self.batch_first)
        # The original model only returns the output, so we match that behavior.
        return output