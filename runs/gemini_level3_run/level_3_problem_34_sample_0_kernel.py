import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for the fused RNN forward pass
rnn_fused_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused RNN forward kernel that processes the entire sequence in one launch.
// Grid is sized to the batch size, with each block handling one sequence.
// Block is sized to a fixed number of threads that cooperate on the computation.
__global__ void rnn_fused_forward_kernel(
    const float* __restrict__ x,         // Input: (seq_len, batch_size, input_size)
    const float* __restrict__ h_init,    // Initial hidden: (batch_size, hidden_size)
    const float* __restrict__ W_ih,      // i2h weight: (hidden_size, input_size + hidden_size)
    const float* __restrict__ b_ih,      // i2h bias: (hidden_size)
    const float* __restrict__ W_ho,      // h2o weight: (output_size, hidden_size)
    const float* __restrict__ b_ho,      // h2o bias: (output_size)
    float* __restrict__ outputs,         // Output: (seq_len, batch_size, output_size)
    const int seq_len,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const int output_size
) {
    // Each block processes one sequence in the batch
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Dynamically sized shared memory to store the current and next hidden states
    // for this block's batch item. This avoids repeated global memory access.
    extern __shared__ float s_mem[];
    float* h_current = s_mem;               // buffer for h_t, size: hidden_size
    float* h_next = s_mem + hidden_size;    // buffer for h_{t+1}, size: hidden_size

    // Threads in the block cooperate to load the initial hidden state into shared memory
    for (int i = tid; i < hidden_size; i += num_threads) {
        h_current[i] = h_init[batch_idx * hidden_size + i];
    }
    __syncthreads(); // Ensure h_init is fully loaded before proceeding

    const int combined_size = input_size + hidden_size;

    // The main loop over the sequence length, now inside the CUDA kernel
    for (int t = 0; t < seq_len; ++t) {
        // Pointer to the current input slice x[t] for this batch item
        const float* x_t = x + t * batch_size * input_size + batch_idx * input_size;

        // --- 1. Fused i2h layer: hidden = tanh(i2h(cat(x_t, h_prev))) ---
        // This is a matrix-vector product: (hidden_size, combined_size) x (combined_size, 1)
        // Threads cooperate to compute the output of the i2h linear layer.
        // Each thread computes one or more elements of the output vector (h_next).
        for (int i = tid; i < hidden_size; i += num_threads) {
            float sum = 0.0f;
            // Dot product for one row of W_ih with the "combined" vector (x_t and h_current)
            // Part 1: dot with x_t from global memory
            for (int k = 0; k < input_size; ++k) {
                sum += x_t[k] * W_ih[i * combined_size + k];
            }
            // Part 2: dot with h_current (previous hidden state) from shared memory
            for (int k = 0; k < hidden_size; ++k) {
                sum += h_current[k] * W_ih[i * combined_size + input_size + k];
            }
            // Add bias and apply tanh, storing in the h_next buffer in shared memory
            h_next[i] = tanhf(sum + b_ih[i]);
        }
        __syncthreads(); // Ensure all parts of h_next are computed

        // Copy h_next to h_current for the next timestep and for the h2o layer
        for (int i = tid; i < hidden_size; i += num_threads) {
            h_current[i] = h_next[i];
        }
        __syncthreads(); // Ensure h_current is fully updated

        // --- 2. Fused h2o layer: output = h2o(hidden) ---
        // This is another matrix-vector product: (output_size, hidden_size) x (hidden_size, 1)
        // Pointer to the output slice for this timestep and batch item
        float* output_t = outputs + t * batch_size * output_size + batch_idx * output_size;

        // Threads cooperate to compute the output vector
        for (int i = tid; i < output_size; i += num_threads) {
            float sum = 0.0f;
            // Dot product with the new h_current from shared memory
            for (int k = 0; k < hidden_size; ++k) {
                sum += h_current[k] * W_ho[i * hidden_size + k];
            }
            // Add bias and write the final result to global memory
            output_t[i] = sum + b_ho[i];
        }
        // Sync before the next time step to ensure all threads in the block are done
        // with the current time step before starting the next one.
        __syncthreads();
    }
}

// C++ wrapper function to launch the CUDA kernel
torch::Tensor rnn_fused_forward_cuda(
    torch::Tensor x,
    torch::Tensor h_init,
    torch::Tensor W_ih,
    torch::Tensor b_ih,
    torch::Tensor W_ho,
    torch::Tensor b_ho
) {
    // Input validation checks
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(h_init.is_cuda(), "h_init must be a CUDA tensor");
    TORCH_CHECK(W_ih.is_cuda(), "W_ih must be a CUDA tensor");
    TORCH_CHECK(b_ih.is_cuda(), "b_ih must be a CUDA tensor");
    TORCH_CHECK(W_ho.is_cuda(), "W_ho must be a CUDA tensor");
    TORCH_CHECK(b_ho.is_cuda(), "b_ho must be a CUDA tensor");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(h_init.is_contiguous(), "h_init must be contiguous");
    TORCH_CHECK(W_ih.is_contiguous(), "W_ih must be contiguous");
    TORCH_CHECK(b_ih.is_contiguous(), "b_ih must be contiguous");
    TORCH_CHECK(W_ho.is_contiguous(), "W_ho must be contiguous");
    TORCH_CHECK(b_ho.is_contiguous(), "b_ho must be contiguous");

    const auto seq_len = x.size(0);
    const auto batch_size = x.size(1);
    const auto input_size = x.size(2);
    const auto hidden_size = h_init.size(1);
    const auto output_size = W_ho.size(0);

    auto outputs = torch::zeros({seq_len, batch_size, output_size}, x.options());

    const int threads_per_block = 256; // A common and often optimal choice
    const dim3 blocks(batch_size);
    const dim3 threads(threads_per_block);

    // Shared memory size: 2 * hidden_size for h_current and h_next buffers
    const size_t shared_mem_size = 2 * hidden_size * sizeof(float);

    // Launch the kernel
    rnn_fused_forward_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        h_init.data_ptr<float>(),
        W_ih.data_ptr<float>(),
        b_ih.data_ptr<float>(),
        W_ho.data_ptr<float>(),
        b_ho.data_ptr<float>(),
        outputs.data_ptr<float>(),
        seq_len,
        batch_size,
        input_size,
        hidden_size,
        output_size
    );
    
    // Use PyTorch's macro to check for errors after kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return outputs;
}
"""

# C++ source for the function signature
rnn_fused_cpp_source = """
torch::Tensor rnn_fused_forward_cuda(
    torch::Tensor x,
    torch::Tensor h_init,
    torch::Tensor W_ih,
    torch::Tensor b_ih,
    torch::Tensor W_ho,
    torch::Tensor b_ho
);
"""

# Use torch's JIT compiler to build the custom operator
rnn_fused_op = load_inline(
    name="rnn_fused_op",
    cpp_sources=rnn_fused_cpp_source,
    cuda_sources=rnn_fused_cuda_source,
    functions=["rnn_fused_forward_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the Vanilla RNN model with a custom fused CUDA kernel.

        :param input_size: The number of input features (int).
        :param hidden_size: The size of the hidden state (int).
        :param output_size: The number of output features (int).
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Define the standard PyTorch layers. These will hold the learnable parameters (weights and biases)
        # that we will pass to our custom kernel.
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        
        # The Tanh activation is now fused inside our custom kernel and is no longer needed here.

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vanilla RNN using the custom fused CUDA kernel.

        :param x: Input tensor of shape (seq_len, batch_size, input_size)
        :param h0: Initial hidden state tensor of shape (batch_size, hidden_size)
        :return: Output tensor of shape (seq_len, batch_size, output_size)
        """
        # The custom kernel expects contiguous tensors, which is handled by the C++ wrapper's checks.
        # We pass the weight and bias tensors from our nn.Linear layers directly to the kernel.
        return rnn_fused_op.rnn_fused_forward_cuda(
            x,
            h0,
            self.i2h.weight,
            self.i2h.bias,
            self.h2o.weight,
            self.h2o.bias
        )