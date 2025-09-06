import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation
fused_linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// TILE_WIDTH determines the size of the tile processed by each thread block.
// 32 is a common choice that balances register usage and parallelism.
#define TILE_WIDTH 32

__global__ void fused_linear_kernel(
    const float* x, 
    const float* weight, 
    const float* bias, 
    float* out, 
    const float subtract_value, 
    const float multiply_value,
    const int M, 
    const int N, 
    const int K) {

    // Shared memory for tiles of x (A) and weight (B)
    __shared__ float x_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float w_tile[TILE_WIDTH][TILE_WIDTH];

    // Identify the row and column of the output matrix tile this block will compute
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;

    // Identify the thread's position within the block
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // Calculate the global row and column for this thread's output element
    int global_row = tile_row * TILE_WIDTH + thread_row;
    int global_col = tile_col * TILE_WIDTH + thread_col;

    // Accumulator for the dot product, stored in a register for each thread
    float acc = 0.0f;

    // Loop over tiles along the K dimension (the shared dimension of x and weight)
    for (int k_tile = 0; k_tile < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++k_tile) {
        // --- Load tiles into shared memory ---
        
        // Load a tile of x
        int x_k = k_tile * TILE_WIDTH + thread_col;
        if (global_row < M && x_k < K) {
            x_tile[thread_row][thread_col] = x[global_row * K + x_k];
        } else {
            x_tile[thread_row][thread_col] = 0.0f; // Padding for out-of-bounds access
        }

        // Load a tile of weight. We are computing x @ W.T, so we need W[global_col, k].
        // We load it transposed into shared memory to facilitate coalesced access and a simpler multiply loop.
        int w_k = k_tile * TILE_WIDTH + thread_row;
        if (global_col < N && w_k < K) {
            w_tile[thread_col][thread_row] = weight[global_col * K + w_k];
        } else {
            w_tile[thread_col][thread_row] = 0.0f; // Padding
        }

        __syncthreads(); // Synchronize to ensure both tiles are fully loaded

        // --- Compute dot product for the tiles ---
        // Each thread computes a partial sum for its output element.
        for (int k = 0; k < TILE_WIDTH; ++k) {
            acc += x_tile[thread_row][k] * w_tile[thread_col][k];
        }

        __syncthreads(); // Synchronize before loading the next tile
    }

    // --- Apply fused operations and write to output ---
    // Check bounds to ensure we only write within the output matrix dimensions
    if (global_row < M && global_col < N) {
        // Add bias
        acc += bias[global_col];
        // Subtract value
        acc -= subtract_value;
        // Multiply value
        acc *= multiply_value;
        // ReLU activation
        acc = fmaxf(0.0f, acc);
        
        out[global_row * N + global_col] = acc;
    }
}

// C++ wrapper function to be called from Python
torch::Tensor fused_linear_cuda(
    torch::Tensor x, 
    torch::Tensor weight, 
    torch::Tensor bias,
    double subtract_value,
    double multiply_value) {
    
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Input weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Input bias must be a CUDA tensor");
    
    TORCH_CHECK(x.dim() == 2, "Input x must be 2D");
    TORCH_CHECK(weight.dim() == 2, "Input weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "Input bias must be 1D");

    // Ensure tensors are contiguous in memory for predictable access patterns
    x = x.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    const int M = x.size(0); // batch_size
    const int K = x.size(1); // in_features
    const int N = weight.size(0); // out_features

    TORCH_CHECK(K == weight.size(1), "Matrix dimensions are not compatible for multiplication");
    TORCH_CHECK(N == bias.size(0), "Bias dimension does not match weight output dimension");

    // Create the output tensor
    auto out = torch::empty({M, N}, x.options());

    // Configure kernel launch parameters
    const dim3 threads(TILE_WIDTH, TILE_WIDTH);
    const dim3 blocks(
        (N + TILE_WIDTH - 1) / TILE_WIDTH, 
        (M + TILE_WIDTH - 1) / TILE_WIDTH
    );

    // Launch the kernel
    fused_linear_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<float>(subtract_value),
        static_cast<float>(multiply_value),
        M, N, K
    );
    
    // Check for any CUDA errors during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
"""

fused_linear_cpp_source = """
torch::Tensor fused_linear_cuda(
    torch::Tensor x, 
    torch::Tensor weight, 
    torch::Tensor bias,
    double subtract_value,
    double multiply_value);
"""

# Compile the inline CUDA code using torch.utils.cpp_extension
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_linear_cpp_source,
    cuda_sources=fused_linear_source,
    functions=["fused_linear_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, subtraction, multiplication, and ReLU activation
    using a single fused CUDA kernel. This avoids writing intermediate tensors to global memory,
    improving performance.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        # We still use nn.Linear to conveniently manage the weight and bias parameters.
        # These parameters will be passed to our custom kernel.
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self.fused_op = fused_op

    def forward(self, x):
        # Call the custom fused CUDA kernel, passing the input tensor,
        # the weight and bias from our nn.Linear layer, and the scalar values.
        return self.fused_op.fused_linear_cuda(
            x, 
            self.linear.weight, 
            self.linear.bias, 
            self.subtract_value, 
            self.multiply_value
        )