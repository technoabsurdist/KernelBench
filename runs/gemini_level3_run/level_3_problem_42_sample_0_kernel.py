import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for the fused GRU cell operations
gru_cell_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Device function for sigmoid, numerically stable
__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// The fused GRU cell kernel
__global__ void gru_cell_kernel(
    const float* __restrict__ gi,      // Input gate pre-activations (from matmul with x_t + bias)
    const float* __restrict__ gh,      // Hidden gate pre-activations (from matmul with h_{t-1} + bias)
    const float* __restrict__ h_prev,  // Previous hidden state h_{t-1}
    float* __restrict__ h_new,         // Output new hidden state h_t
    const int batch_size,
    const int hidden_size
) {
    // Each thread computes one hidden unit for one batch item
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * hidden_size;

    if (idx < total_threads) {
        int batch_idx = idx / hidden_size;
        int hidden_idx = idx % hidden_size;

        // Pointers to the start of the current batch item's data
        const float* gi_b = gi + batch_idx * 3 * hidden_size;
        const float* gh_b = gh + batch_idx * 3 * hidden_size;

        // Calculate reset gate (r)
        float r = sigmoidf(gi_b[hidden_idx] + gh_b[hidden_idx]);

        // Calculate update gate (z)
        float z = sigmoidf(gi_b[hidden_idx + hidden_size] + gh_b[hidden_idx + hidden_size]);

        // Calculate new gate (n)
        float n = tanhf(gi_b[hidden_idx + 2 * hidden_size] + r * gh_b[hidden_idx + 2 * hidden_size]);

        // Calculate new hidden state h_t
        h_new[idx] = (1.0f - z) * n + z * h_prev[idx];
    }
}

// C++ wrapper function to be called from PyTorch
torch::Tensor gru_cell_forward_cuda(
    torch::Tensor gi,      // Result of linear_ih(x_t)
    torch::Tensor gh,      // Result of linear_hh(h_{t-1})
    torch::Tensor h_prev   // h_{t-1}
) {
    // Input validation checks
    TORCH_CHECK(gi.is_cuda(), "gi must be a CUDA tensor");
    TORCH_CHECK(gh.is_cuda(), "gh must be a CUDA tensor");
    TORCH_CHECK(h_prev.is_cuda(), "h_prev must be a CUDA tensor");
    
    TORCH_CHECK(gi.is_contiguous(), "gi must be contiguous");
    TORCH_CHECK(gh.is_contiguous(), "gh must be contiguous");
    TORCH_CHECK(h_prev.is_contiguous(), "h_prev must be contiguous");

    const auto batch_size = h_prev.size(0);
    const auto hidden_size = h_prev.size(1);
    
    auto h_new = torch::empty_like(h_prev);
    
    const int total_threads = batch_size * hidden_size;
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;

    gru_cell_kernel<<<num_blocks, block_size>>>(
        gi.data_ptr<float>(),
        gh.data_ptr<float>(),
        h_prev.data_ptr<float>(),
        h_new.data_ptr<float>(),
        batch_size,
        hidden_size
    );
    
    // Check for any CUDA errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return h_new;
}
"""

gru_cell_cpp_source = (
    "torch::Tensor gru_cell_forward_cuda(torch::Tensor gi, torch::Tensor gh, torch::Tensor h_prev);"
)

# JIT compile the inline CUDA code
custom_gru_cell = load_inline(
    name="custom_gru_cell",
    cpp_sources=gru_cell_cpp_source,
    cuda_sources=gru_cell_source,
    functions=["gru_cell_forward_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = True # Hardcoded from original model
        self.num_directions = 2 if self.bidirectional else 1

        # Use ModuleList to hold all the linear layers for the GRU
        self.gru_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size * self.num_directions
            
            # Forward direction weights (packaged as nn.Linear for convenience)
            self.gru_layers.add_module(f'layer_{i}_fwd_ih', nn.Linear(layer_input_size, 3 * hidden_size, bias=bias))
            self.gru_layers.add_module(f'layer_{i}_fwd_hh', nn.Linear(hidden_size, 3 * hidden_size, bias=bias))
            
            # Backward direction weights
            self.gru_layers.add_module(f'layer_{i}_bwd_ih', nn.Linear(layer_input_size, 3 * hidden_size, bias=bias))
            self.gru_layers.add_module(f'layer_{i}_bwd_hh', nn.Linear(hidden_size, 3 * hidden_size, bias=bias))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights and biases uniformly."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def _gru_layer_forward(self, x, h0, layer_idx):
        """Executes a single bidirectional GRU layer."""
        # Get pre-created linear layers for this GRU layer
        fwd_ih = getattr(self.gru_layers, f'layer_{layer_idx}_fwd_ih')
        fwd_hh = getattr(self.gru_layers, f'layer_{layer_idx}_fwd_hh')
        bwd_ih = getattr(self.gru_layers, f'layer_{layer_idx}_bwd_ih')
        bwd_hh = getattr(self.gru_layers, f'layer_{layer_idx}_bwd_hh')

        # h0 shape is (num_directions, batch, hidden) for this specific layer
        h_fwd, h_bwd = h0[0], h0[1]
        
        seq_len = x.size(0)
        
        # Forward pass
        outputs_fwd = []
        for t in range(seq_len):
            gi = fwd_ih(x[t])
            gh = fwd_hh(h_fwd)
            h_fwd = custom_gru_cell.gru_cell_forward_cuda(gi, gh, h_fwd)
            outputs_fwd.append(h_fwd)
        
        # Backward pass
        outputs_bwd = []
        for t in range(seq_len - 1, -1, -1):
            gi = bwd_ih(x[t])
            gh = bwd_hh(h_bwd)
            h_bwd = custom_gru_cell.gru_cell_forward_cuda(gi, gh, h_bwd)
            outputs_bwd.append(h_bwd)
        
        outputs_fwd = torch.stack(outputs_fwd)
        outputs_bwd = torch.stack(outputs_bwd[::-1]) # Reverse the collected backward outputs
        
        layer_output = torch.cat([outputs_fwd, outputs_bwd], dim=2)
        final_hidden = torch.stack([h_fwd, h_bwd])
        
        return layer_output, final_hidden

    def forward(self, x, h0):
        if self.batch_first:
            x = x.transpose(0, 1)
        
        # h0 shape: (num_layers * num_directions, batch_size, hidden_size)
        # Reshape for easier iteration: (num_layers, num_directions, batch_size, hidden_size)
        h0 = h0.view(self.num_layers, self.num_directions, x.size(1), self.hidden_size)
        
        layer_input = x
        final_hiddens = []
        
        for i in range(self.num_layers):
            layer_output, final_hidden = self._gru_layer_forward(layer_input, h0[i], i)
            layer_input = layer_output
            final_hiddens.append(final_hidden)
            
        # Stack final hidden states from all layers to match PyTorch's output format
        h_n = torch.cat(final_hiddens, dim=0)
        
        return h_n