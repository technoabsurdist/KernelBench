import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA source code for the fused operations
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// --- Device Functions for Activations ---
__device__ __forceinline__ float hardtanh_func(float x) {
    return fmaxf(-1.0f, fminf(1.0f, x));
}

__device__ __forceinline__ float mish_func(float x) {
    // Use a numerically stable implementation of softplus
    float sp;
    if (x > 20.0f) {
        sp = x;
    } else if (x < -20.0f) {
        sp = expf(x);
    } else {
        sp = logf(1.0f + expf(x));
    }
    return x * tanhf(sp);
}

// --- Fused GEMM + Bias + Activations Kernel ---
// Tiled GEMM for C = A * B, where A is input, B is weight.T
// A: (M, K), B: (K, N), C: (M, N)
#define TILE_DIM 32

__global__ void fused_gemm_bias_hardtanh_mish_kernel(
    const float* A, const float* B, const float* bias, float* C,
    int M, int N, int K) {

    // Shared memory for tiles of A and B
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global row and column for this thread's output element
    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    // Accumulator for the output element
    float acc = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Load tile of A into shared memory
        int a_row = row;
        int a_col = t * TILE_DIM + tx;
        if (a_row < M && a_col < K) {
            sA[ty][tx] = A[a_row * K + a_col];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        int b_row = t * TILE_DIM + ty;
        int b_col = col;
        if (b_row < K && b_col < N) {
            sB[ty][tx] = B[b_row * N + b_col];
        } else {
            sB[ty][tx] = 0.0f;
        }
        __syncthreads();

        // Multiply tiles from shared memory
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }

    // Apply epilogue (BiasAdd -> Hardtanh -> Mish) and write to global memory
    if (row < M && col < N) {
        acc += bias[col];
        acc = hardtanh_func(acc);
        acc = mish_func(acc);
        C[row * N + col] = acc;
    }
}

torch::Tensor fused_gemm_op_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor combined_bias) {

    const auto M = x.size(0);
    const auto K = x.size(1);
    const auto N = weight.size(0);

    auto y = torch::empty({M, N}, x.options());

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // We need W.T, which is (K, N). The weight tensor is (N, K).
    auto W_T = weight.transpose(0, 1).contiguous();

    fused_gemm_bias_hardtanh_mish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        W_T.data_ptr<float>(),
        combined_bias.data_ptr<float>(),
        y.data_ptr<float>(),
        M, N, K
    );
    return y;
}

// --- Custom GroupNorm Kernels (Two-Pass) ---

// Kernel 1: Calculate mean and inverse standard deviation for each group
__global__ void group_norm_stats_kernel(
    const float* x, float* mean, float* inv_std,
    int N, int C, int G, float eps) {

    int group_idx = blockIdx.x; // Each block processes one group for one batch item
    int n = group_idx / G;
    int g = group_idx % G;

    int group_size = C / G;
    int start_channel = g * group_size;

    __shared__ float s_sum;
    __shared__ float s_sum_sq;

    if (threadIdx.x == 0) {
        s_sum = 0.0f;
        s_sum_sq = 0.0f;
    }
    __syncthreads();

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // Each thread sums up a portion of the elements in the group
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c = start_channel + i;
        float val = x[n * C + c];
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Reduce within the block using atomic operations
    atomicAdd(&s_sum, local_sum);
    atomicAdd(&s_sum_sq, local_sum_sq);
    __syncthreads();

    // Thread 0 computes final stats and writes to global memory
    if (threadIdx.x == 0) {
        float mu = s_sum / group_size;
        float var = s_sum_sq / group_size - mu * mu;
        mean[group_idx] = mu;
        inv_std[group_idx] = rsqrtf(var + eps);
    }
}

// Kernel 2: Apply the normalization using the calculated stats
__global__ void group_norm_forward_kernel(
    const float* x, const float* mean, const float* inv_std,
    const float* gamma, const float* beta, float* y,
    int N, int C, int G) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) return;

    int n = idx / C;
    int c = idx % C;

    int group_size = C / G;
    int g = c / group_size;
    int group_idx = n * G + g;

    float mu = mean[group_idx];
    float rsigma = inv_std[group_idx];
    float scale = gamma[c];
    float bias = beta[c];

    y[idx] = (x[idx] - mu) * rsigma * scale + bias;
}

torch::Tensor group_norm_op_cuda(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int num_groups, double eps) {

    const int N = x.size(0);
    const int C = x.size(1);
    const int G = num_groups;

    auto mean = torch::empty({N, G}, x.options());
    auto inv_std = torch::empty({N, G}, x.options());
    auto y = torch::empty_like(x);

    // Launch Kernel 1: Calculate stats
    const int group_size = C / G;
    // Choose a reasonable block size for the reduction
    const int block_size_stats = (group_size > 512) ? 512 : (group_size > 256 ? 256 : 128);
    dim3 blocks_stats(N * G);
    dim3 threads_stats(block_size_stats);

    group_norm_stats_kernel<<<blocks_stats, threads_stats>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        N, C, G, static_cast<float>(eps)
    );

    // Launch Kernel 2: Apply normalization
    const int block_size_fwd = 1024;
    const int num_blocks_fwd = (N * C + block_size_fwd - 1) / block_size_fwd;
    dim3 blocks_fwd(num_blocks_fwd);
    dim3 threads_fwd(block_size_fwd);

    group_norm_forward_kernel<<<blocks_fwd, threads_fwd>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, G
    );

    return y;
}
"""

# Define the C++ source for function signatures that will be exposed to Python
fused_op_cpp_source = """
torch::Tensor fused_gemm_op_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor combined_bias);
torch::Tensor group_norm_op_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int num_groups, double eps);
"""

# Compile the inline CUDA code using JIT compilation
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_gemm_op_cuda", "group_norm_op_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    An optimized model that fuses GEMM, BiasAdd, Hardtanh, and Mish operations
    into a single CUDA kernel, and uses a custom two-pass CUDA kernel for GroupNorm.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        # Keep original PyTorch modules to hold the learnable parameters (weights and biases).
        # This makes the model compatible with PyTorch's optimizers and state_dict.
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)

        # Store the compiled custom functions
        self.fused_gemm_op = fused_ops.fused_gemm_op_cuda
        self.group_norm_op = fused_ops.group_norm_op_cuda

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Ensure input is on the correct device
        x = x.cuda()

        # Combine the two biases before calling the fused kernel.
        # This is done on the CPU/GPU by PyTorch and is a small operation.
        if self.gemm.bias is not None:
            combined_bias = self.gemm.bias + self.bias
        else:
            combined_bias = self.bias
        
        # Call the first fused kernel for: GEMM -> BiasAdd -> Hardtanh -> Mish
        x = self.fused_gemm_op(x, self.gemm.weight, combined_bias)
        
        # Call the custom GroupNorm kernel
        x = self.group_norm_op(
            x, self.groupnorm.weight, self.groupnorm.bias,
            self.groupnorm.num_groups, self.groupnorm.eps
        )
        return x