import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels and C++ wrappers for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel 1: Fused GEMM + Bias + Sigmoid
// Computes C = sigmoid(A @ B.T + bias)
// A: (M, K), B: (N, K), bias: (N), C: (M, N)
__global__ void gemm_bias_sigmoid_kernel(
    const float* A, const float* B, const float* bias, float* C,
    int M, int N, int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int k = 0; k < K; ++k) {
            // B is weight (N, K), so B.T is (K, N)
            // A @ B.T is equivalent to sum over k of A[row, k] * B[col, k]
            value += A[row * K + k] * B[col * K + k];
        }
        value += bias[col];
        C[row * N + col] = 1.0f / (1.0f + expf(-value));
    }
}

// Kernel 2: Fused GEMM + Bias + LogSumExp
// Computes C_i = logsumexp(A_i @ B.T + bias) for each row i
// A: (B_dim, K), B: (N, K), bias: (N), C: (B_dim)
__global__ void gemm_bias_logsumexp_kernel(
    const float* A, const float* B, const float* bias, float* C,
    int B_dim, int N, int K) {

    // Each block processes one row of the input batch A
    int row = blockIdx.x;
    if (row >= B_dim) return;

    extern __shared__ float sdata[];
    // sdata will be used for two purposes:
    // 1. To store the intermediate GEMM result for the current row (size N)
    // 2. To perform parallel reductions (reusing the first blockDim.x elements)

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // --- Step 1: Compute GEMM row and store in shared memory ---
    for (int col = tid; col < N; col += block_size) {
        float value = 0.0f;
        const float* A_row = A + row * K;
        const float* B_col = B + col * K;
        for (int k = 0; k < K; ++k) {
            value += A_row[k] * B_col[k];
        }
        sdata[col] = value + bias[col];
    }
    __syncthreads();

    // --- Step 2: Find max value in parallel ---
    float thread_max = -1.0f/0.0f; // -INFINITY
    for (int i = tid; i < N; i += block_size) {
        if (sdata[i] > thread_max) {
            thread_max = sdata[i];
        }
    }

    // Reduce max values across the block
    // We reuse the start of sdata for the reduction array
    sdata[tid] = thread_max;
    __syncthreads();

    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }
    float block_max = sdata[0];
    __syncthreads();

    // --- Step 3: Compute sum of exp(val - max) in parallel ---
    // We need the original GEMM results again. They are still in sdata[0...N-1].
    float* s_gemm_results = sdata;
    float* s_reduce = sdata;

    float thread_sum = 0.0f;
    for (int i = tid; i < N; i += block_size) {
        thread_sum += expf(s_gemm_results[i] - block_max);
    }

    // Reduce sum values across the block
    s_reduce[tid] = thread_sum;
    __syncthreads();

    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_reduce[tid] += s_reduce[tid + s];
        }
        __syncthreads();
    }
    float block_sum = s_reduce[0];

    // --- Step 4: Final calculation and write output ---
    if (tid == 0) {
        C[row] = block_max + logf(block_sum);
    }
}

// C++ Wrapper for gemm_bias_sigmoid
torch::Tensor gemm_bias_sigmoid_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor bias) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be a float32 tensor");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be a float32 tensor");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be a float32 tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(A.size(1) == B.size(1), "Inner dimensions of A and B must match");
    TORCH_CHECK(B.size(0) == bias.size(0), "Dimensions of B and bias must match");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    const dim3 block_dim(16, 16);
    const dim3 grid_dim(
        (N + block_dim.x - 1) / block_dim.x,
        (M + block_dim.y - 1) / block_dim.y
    );

    gemm_bias_sigmoid_kernel<<<grid_dim, block_dim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        bias.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

// C++ Wrapper for gemm_bias_logsumexp
torch::Tensor gemm_bias_logsumexp_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor bias) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be a float32 tensor");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be a float32 tensor");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be a float32 tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(A.size(1) == B.size(1), "Inner dimensions of A and B must match");
    TORCH_CHECK(B.size(0) == bias.size(0), "Dimensions of B and bias must match");

    const int B_dim = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({B_dim}, A.options());

    const int block_size = 256; // A common choice, can be tuned
    const dim3 block_dim(block_size);
    const dim3 grid_dim(B_dim);
    const size_t shared_mem_size = N * sizeof(float);

    gemm_bias_logsumexp_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        bias.data_ptr<float>(),
        C.data_ptr<float>(),
        B_dim, N, K
    );

    return C;
}
"""

fused_ops_cpp_source = """
torch::Tensor gemm_bias_sigmoid_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor bias);
torch::Tensor gemm_bias_logsumexp_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["gemm_bias_sigmoid_cuda", "gemm_bias_logsumexp_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses operations into custom CUDA kernels.
    - Fuses (linear1 + sigmoid) into a single kernel.
    - Fuses (linear2 + logsumexp) into a single kernel.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        # These layers are still needed to store and manage the model parameters
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.fused_ops = fused_ops

    def forward(self, x):
        # Ensure input is contiguous for custom kernels
        x = x.contiguous()
        
        # Call the first fused kernel: GEMM + Bias + Sigmoid
        x = self.fused_ops.gemm_bias_sigmoid_cuda(
            x, self.linear1.weight, self.linear1.bias
        )
        
        # Call the second fused kernel: GEMM + Bias + LogSumExp
        # The output of the first kernel must be contiguous for the next kernel
        x = self.fused_ops.gemm_bias_logsumexp_cuda(
            x.contiguous(), self.linear2.weight, self.linear2.bias
        )
        return x