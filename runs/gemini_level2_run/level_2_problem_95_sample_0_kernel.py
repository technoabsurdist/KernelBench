import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation
# This kernel performs:
# 1. Matrix Multiplication (GEMM)
# 2. Fused Epilogue:
#    - Bias addition (from nn.Linear)
#    - Element-wise addition (from add_value parameter)
#    - Swish activation
#    - Tanh activation
#    - GELU activation
#    - Hardtanh activation
# This fusion reduces memory bandwidth and kernel launch overhead.
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// --- Tiling configuration for GEMM ---
// Using a 16x16 tile size is a safe and common choice that balances
// register usage and shared memory capacity on most GPUs.
#define TILE_DIM 16
#define PI 3.14159265358979323846f

// --- Device-side activation functions ---
// These inline functions will be compiled directly into the kernel code.

__device__ inline float sigmoid_dev(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ inline float swish_dev(float x) {
    return x * sigmoid_dev(x);
}

__device__ inline float tanh_dev(float x) {
    return tanhf(x);
}

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ inline float gelu_dev(float x) {
    float c = sqrtf(2.0f / PI);
    float inner = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ inline float hardtanh_dev(float x, float min_val, float max_val) {
    return fmaxf(min_val, fminf(x, max_val));
}


// --- Fused Kernel ---
// A tiled matrix multiplication kernel with a fused epilogue.
// Each thread block computes one tile of the output matrix C.
// Threads within a block cooperate to load tiles of A and B into shared memory.
__global__ void fused_gemm_activations_kernel(
    const float* A, const float* B, const float* bias1, const float* bias2, float* C,
    int M, int N, int K) {

    // Shared memory for tiles of A and B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // Thread and block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identify the row and column of the C sub-matrix to work on
    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    float Cvalue = 0.0f;

    // Loop over the tiles of A and B required to compute the C sub-matrix
    for (int m = 0; m < (K + TILE_DIM - 1) / TILE_DIM; ++m) {
        // Load tile of A from global memory to shared memory
        int a_row = by * TILE_DIM + ty;
        int a_col = m * TILE_DIM + tx;
        if (a_row < M && a_col < K) {
            As[ty][tx] = A[a_row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B from global memory to shared memory
        int b_row = m * TILE_DIM + ty;
        int b_col = bx * TILE_DIM + tx;
        if (b_row < K && b_col < N) {
            Bs[ty][tx] = B[b_row * N + b_col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();

        // Multiply the tiles from shared memory
        for (int k = 0; k < TILE_DIM; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    // Fused epilogue: apply biases and activations
    if (row < M && col < N) {
        // Add biases from nn.Linear and the custom add_value parameter
        Cvalue += bias1[col] + bias2[col];

        // Apply activations in sequence
        Cvalue = swish_dev(Cvalue);
        Cvalue = tanh_dev(Cvalue);
        Cvalue = gelu_dev(Cvalue);
        Cvalue = hardtanh_dev(Cvalue, -1.0f, 1.0f);

        // Write the final result to global memory
        C[row * N + col] = Cvalue;
    }
}


// --- Host-side C++ wrapper ---
// This function is the interface between PyTorch and the CUDA kernel.
torch::Tensor fused_linear_activations_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor add_value) {

    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias tensor must be a CUDA tensor");
    TORCH_CHECK(add_value.is_cuda(), "add_value tensor must be a CUDA tensor");

    TORCH_CHECK(x.dim() == 2, "Input tensor x must be 2D");
    TORCH_CHECK(weight.dim() == 2, "Weight tensor must be 2D");

    // Get dimensions
    // x: (M, K), weight: (N, K)
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);

    TORCH_CHECK(weight.size(1) == K, "Dimension mismatch: x and weight");
    TORCH_CHECK(bias.size(0) == N, "Dimension mismatch: weight and bias");
    TORCH_CHECK(add_value.size(0) == N, "Dimension mismatch: weight and add_value");

    // Allocate output tensor
    auto out = torch::empty({M, N}, x.options());

    // PyTorch's nn.Linear computes x @ weight.T.
    // Our GEMM computes A @ B. So, A=x, B=weight.T.
    // We need to transpose the weight matrix before passing it to the kernel.
    auto weight_t = weight.t().contiguous();

    // Setup grid and block dimensions for the kernel launch
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // Launch kernel
    fused_gemm_activations_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight_t.data_ptr<float>(),
        bias.data_ptr<float>(),
        add_value.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N, K
    );
    
    // Check for any CUDA errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# C++ source for function signature binding
fused_op_cpp_source = """
torch::Tensor fused_linear_activations_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor add_value);
"""

# Compile the inline CUDA/C++ code using PyTorch's JIT compiler
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_linear_activations_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that replaces the entire operation sequence with a single
    custom fused CUDA kernel.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        # These layers are kept to store the learnable parameters (weight, bias)
        # in a standard PyTorch way. They are not used for computation in the forward pass.
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))

    def forward(self, x):
        # The forward pass now consists of a single call to our custom fused kernel.
        # This passes the input tensor and the learnable parameters directly to the GPU
        # for a single, efficient computation.
        return fused_op.fused_linear_activations_cuda(
            x,
            self.matmul.weight,
            self.matmul.bias,
            self.add_value
        )

batch_size = 1024
in_features = 8192
out_features = 8192
add_value_shape = (out_features,)

def get_inputs():
    # Return a CUDA tensor as required by the custom kernel
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, add_value_shape]