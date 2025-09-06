import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernels for Fused Linear+ReLU and Linear
# We use a tiled matrix multiplication for performance.
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_DIM 32

// CUDA kernel for Fused Linear (MatMul + Bias) + ReLU
__global__ void fused_linear_relu_kernel(const float* A, const float* B, const float* bias, float* C, int M, int N, int K) {
    // A: (M, K) -> input
    // B: (N, K) -> weight
    // C: (M, N) -> output
    // bias: (N)
    // Operation: C = relu(A @ B.T + bias)

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    int C_row = block_row * TILE_DIM + thread_row;
    int C_col = block_col * TILE_DIM + thread_col;

    float C_value = 0.0f;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Load a tile of A into shared memory
        int A_row_load = block_row * TILE_DIM + thread_row;
        int A_col_load = t * TILE_DIM + thread_col;
        if (A_row_load < M && A_col_load < K) {
            As[thread_row][thread_col] = A[A_row_load * K + A_col_load];
        } else {
            As[thread_row][thread_col] = 0.0f;
        }

        // Load a tile of B into shared memory
        int B_row_load = block_col * TILE_DIM + thread_row;
        int B_col_load = t * TILE_DIM + thread_col;
        if (B_row_load < N && B_col_load < K) {
            Bs[thread_row][thread_col] = B[B_row_load * K + B_col_load];
        } else {
            Bs[thread_row][thread_col] = 0.0f;
        }

        __syncthreads();

        // Multiply the two tiles
        for (int k = 0; k < TILE_DIM; ++k) {
            C_value += As[thread_row][k] * Bs[thread_col][k];
        }

        __syncthreads();
    }

    if (C_row < M && C_col < N) {
        C_value += bias[C_col];
        C[C_row * N + C_col] = fmaxf(0.0f, C_value); // ReLU fusion
    }
}

// CUDA kernel for Linear (MatMul + Bias)
__global__ void linear_kernel(const float* A, const float* B, const float* bias, float* C, int M, int N, int K) {
    // A: (M, K) -> input
    // B: (N, K) -> weight
    // C: (M, N) -> output
    // bias: (N)
    // Operation: C = A @ B.T + bias

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    int C_row = block_row * TILE_DIM + thread_row;
    int C_col = block_col * TILE_DIM + thread_col;

    float C_value = 0.0f;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        int A_row_load = block_row * TILE_DIM + thread_row;
        int A_col_load = t * TILE_DIM + thread_col;
        if (A_row_load < M && A_col_load < K) {
            As[thread_row][thread_col] = A[A_row_load * K + A_col_load];
        } else {
            As[thread_row][thread_col] = 0.0f;
        }

        int B_row_load = block_col * TILE_DIM + thread_row;
        int B_col_load = t * TILE_DIM + thread_col;
        if (B_row_load < N && B_col_load < K) {
            Bs[thread_row][thread_col] = B[B_row_load * K + B_col_load];
        } else {
            Bs[thread_row][thread_col] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            C_value += As[thread_row][k] * Bs[thread_col][k];
        }

        __syncthreads();
    }

    if (C_row < M && C_col < N) {
        C_value += bias[C_col];
        C[C_row * N + C_col] = C_value; // No ReLU
    }
}
"""

cpp_source = """
#include <torch/extension.h>

#define TILE_DIM 32

// Forward declarations of the CUDA kernels
void fused_linear_relu_kernel(const float* A, const float* B, const float* bias, float* C, int M, int N, int K);
void linear_kernel(const float* A, const float* B, const float* bias, float* C, int M, int N, int K);

// C++ interface for FusedLinearReLU
torch::Tensor fused_linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
    TORCH_CHECK(input.size(1) == weight.size(1), "Input and weight feature dimensions must match: ", input.size(1), " vs ", weight.size(1));
    TORCH_CHECK(weight.size(0) == bias.size(0), "Weight and bias output dimensions must match: ", weight.size(0), " vs ", bias.size(0));

    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weight.size(0);

    auto output = torch::zeros({M, N}, input.options());

    const dim3 threads(TILE_DIM, TILE_DIM);
    const dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    fused_linear_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );
    return output;
}

// C++ interface for CustomLinear
torch::Tensor linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
    TORCH_CHECK(input.size(1) == weight.size(1), "Input and weight feature dimensions must match: ", input.size(1), " vs ", weight.size(1));
    TORCH_CHECK(weight.size(0) == bias.size(0), "Weight and bias output dimensions must match: ", weight.size(0), " vs ", bias.size(0));

    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weight.size(0);

    auto output = torch::zeros({M, N}, input.options());

    const dim3 threads(TILE_DIM, TILE_DIM);
    const dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    linear_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );
    return output;
}
"""

# Compile the inline CUDA code
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_linear_relu_cuda", "linear_cuda"],
    verbose=False,
)

class FusedLinearReLU(nn.Module):
    """
    Custom nn.Module for a Linear layer followed by a ReLU activation,
    fused into a single high-performance CUDA kernel.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device='cuda'))
        self.bias = nn.Parameter(torch.empty(out_features, device='cuda'))
        self.reset_parameters()

    def reset_parameters(self):
        # Mimic nn.Linear's default initialization for fair comparison
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return custom_ops.fused_linear_relu_cuda(x.contiguous(), self.weight, self.bias)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'

class CustomLinear(nn.Module):
    """
    Custom nn.Module for a Linear layer, implemented with a custom CUDA kernel.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device='cuda'))
        self.bias = nn.Parameter(torch.empty(out_features, device='cuda'))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return custom_ops.linear_cuda(x.contiguous(), self.weight, self.bias)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()
        
        layers = []
        current_input_size = input_size
        
        # Create hidden layers with our custom FusedLinearReLU module
        for hidden_size in hidden_layer_sizes:
            layers.append(FusedLinearReLU(current_input_size, hidden_size))
            current_input_size = hidden_size
        
        # The final layer has no ReLU, so we use our CustomLinear module
        layers.append(CustomLinear(current_input_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        return self.network(x)