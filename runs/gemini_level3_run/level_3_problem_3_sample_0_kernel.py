import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA/C++ source code for the fused Linear + ReLU operation
fused_linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

// The dimension of the square tile processed by each thread block.
// A value of 32 is a good default for modern GPUs.
#define TILE_DIM 32

// Custom CUDA kernel for a Fused Linear + ReLU operation.
// This kernel computes Z = ReLU(X @ W.T + B) using a tiled matrix multiplication approach.
//
// Template parameters:
// - X: Input tensor with shape (M, K)
// - W: Weight tensor with shape (N, K)
// - B: Bias tensor with shape (N)
// - Z: Output tensor with shape (M, N)
__global__ void fused_linear_relu_kernel(
    const float* X, const float* W, const float* B, float* Z,
    int M, int N, int K)
{
    // Shared memory to store tiles of X and W, allowing for faster access during computation.
    __shared__ float X_tile[TILE_DIM][TILE_DIM];
    __shared__ float W_tile[TILE_DIM][TILE_DIM];

    // Get the thread and block indices.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Calculate the row and column of the output matrix Z that this thread will compute.
    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    // Accumulator for the dot product, stored in a register for each thread.
    float acc = 0.0f;

    // Loop over the tiles along the K dimension (the dimension being summed over).
    for (int k_tile = 0; k_tile < (K + TILE_DIM - 1) / TILE_DIM; ++k_tile) {
        // --- Step 1: Load tiles from global memory to shared memory ---
        
        // Load a tile of the input matrix X.
        // Each thread loads one element of the tile.
        int x_idx = row * K + k_tile * TILE_DIM + tx;
        if (row < M && (k_tile * TILE_DIM + tx) < K) {
            X_tile[ty][tx] = X[x_idx];
        } else {
            X_tile[ty][tx] = 0.0f; // Pad with zero if out of bounds
        }

        // Load a tile of the weight matrix W.
        // This access pattern is designed for coalesced memory reads.
        int w_idx = (bx * TILE_DIM + ty) * K + (k_tile * TILE_DIM + tx);
        if ((bx * TILE_DIM + ty) < N && (k_tile * TILE_DIM + tx) < K) {
            W_tile[ty][tx] = W[w_idx];
        } else {
            W_tile[ty][tx] = 0.0f; // Pad with zero if out of bounds
        }
        
        // Synchronize to ensure all threads in the block have finished loading their data
        // into shared memory before proceeding to the computation.
        __syncthreads();

        // --- Step 2: Compute the dot product for the current tiles ---
        // Each thread computes a partial sum for its output element.
        for (int k = 0; k < TILE_DIM; ++k) {
            // Accessing W_tile[tx][k] effectively transposes the W tile on-the-fly,
            // which is what's needed for the X @ W.T operation. This also helps
            // avoid shared memory bank conflicts.
            acc += X_tile[ty][k] * W_tile[tx][k];
        }

        // Synchronize again to ensure all computations for the current tile are finished
        // before the next iteration loads new data into shared memory.
        __syncthreads();
    }

    // --- Step 3: Add bias, apply ReLU, and write the final result to global memory ---
    if (row < M && col < N) {
        // Add the bias term.
        acc += B[col];
        // Apply the ReLU activation function.
        Z[row * N + col] = fmaxf(0.0f, acc);
    }
}

// C++ wrapper function that will be exposed to Python via PyTorch's C++ extension API.
torch::Tensor fused_linear_relu_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    // Input validation checks.
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "Input w must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "Input b must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input x must be 2D");
    TORCH_CHECK(w.dim() == 2, "Input w must be 2D");
    TORCH_CHECK(b.dim() == 1, "Input b must be 1D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input x must be a float32 tensor");
    TORCH_CHECK(w.scalar_type() == torch::kFloat32, "Input w must be a float32 tensor");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32, "Input b must be a float32 tensor");
    TORCH_CHECK(x.size(1) == w.size(1), "Input x and weight w have incompatible shapes for matmul");
    TORCH_CHECK(w.size(0) == b.size(0), "Weight w and bias b have incompatible shapes");

    // Get tensor dimensions for kernel launch configuration.
    const int M = x.size(0); // batch_size
    const int K = x.size(1); // in_features
    const int N = w.size(0); // out_features

    // Create the output tensor on the same device as the input.
    auto out = torch::zeros({M, N}, x.options());

    // Define grid and block dimensions for the CUDA kernel launch.
    const dim3 threads(TILE_DIM, TILE_DIM);
    const dim3 blocks(
        (N + TILE_DIM - 1) / TILE_DIM,
        (M + TILE_DIM - 1) / TILE_DIM
    );

    // Launch the custom CUDA kernel.
    fused_linear_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N, K
    );
    
    // Check for any errors during kernel execution.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
"""

# C++ source for the function signature, required by load_inline.
fused_linear_relu_cpp_source = "torch::Tensor fused_linear_relu_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);"

# Use torch.utils.cpp_extension.load_inline to JIT compile the CUDA code.
# This is done once when the Python module is imported.
fused_op_module = load_inline(
    name="fused_op_module",
    cpp_sources=fused_linear_relu_cpp_source,
    cuda_sources=fused_linear_relu_source,
    functions=["fused_linear_relu_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()
        
        # Store the compiled custom operator for use in the forward pass.
        self.fused_linear_relu = fused_op_module.fused_linear_relu_cuda
        
        # Create the layers. We still use nn.Linear to manage weights and biases,
        # but we will call our custom kernel in the forward pass.
        self.hidden_layers = nn.ModuleList()
        current_input_size = input_size
        
        for hidden_size in hidden_layer_sizes:
            self.hidden_layers.append(nn.Linear(current_input_size, hidden_size))
            current_input_size = hidden_size
        
        # The final output layer does not have a ReLU, so we define it separately.
        self.output_layer = nn.Linear(current_input_size, output_size)
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Apply the custom fused Linear + ReLU operation for all hidden layers.
        for layer in self.hidden_layers:
            x = self.fused_linear_relu(x, layer.weight, layer.bias)
            
        # Apply the final linear layer using the standard, highly-optimized PyTorch operator.
        x = F.linear(x, self.output_layer.weight, self.output_layer.bias)
        
        return x