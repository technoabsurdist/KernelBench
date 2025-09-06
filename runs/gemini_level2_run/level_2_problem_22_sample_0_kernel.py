import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- CUDA Kernel 1: Fused Linear + Scale + Add + Clamp ---
# This kernel fuses the following operations:
# 1. Matrix multiplication (nn.Linear)
# 2. Scaling (x * scale_factor)
# 3. Residual addition (x + x, which is x * 2)
# 4. Clamping (torch.clamp)
# The combined operation is: output = clamp(((input @ W.T + bias) * (scale_factor * 2)), min, max)
# A tiled matrix multiplication approach with shared memory is used for performance.

fused_linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_W 16

__global__ void fused_linear_kernel(
    const float* A, const float* B, const float* bias, float* C,
    int M, int N, int K,
    float scale, float min_val, float max_val) {

    // A: (M, K) -> input
    // B: (N, K) -> weight
    // C: (M, N) -> output
    // Computes C = clamp(((A @ B.T + bias) * scale), min_val, max_val)

    __shared__ float sA[TILE_W][TILE_W];
    __shared__ float sB[TILE_W][TILE_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_W + ty;
    int col = blockIdx.x * TILE_W + tx;

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_W - 1) / TILE_W; ++t) {
        // Load tile of A into shared memory (coalesced access)
        int a_idx = row * K + (t * TILE_W + tx);
        if (row < M && (t * TILE_W + tx) < K) {
            sA[ty][tx] = A[a_idx];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory (coalesced access)
        // This effectively loads a tile of B.T into sB to facilitate the matmul calculation.
        int b_row = blockIdx.x * TILE_W + ty; // Corresponds to col in C
        int b_col = t * TILE_W + tx;          // Corresponds to k
        if (b_row < N && b_col < K) {
            sB[tx][ty] = B[b_row * K + b_col];
        } else {
            sB[tx][ty] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product from tiles in shared memory
        for (int k = 0; k < TILE_W; ++k) {
            acc += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    // Apply the fused epilogue (bias, scale, clamp) and write to output
    if (row < M && col < N) {
        acc += bias[col];
        acc *= scale;
        acc = fmaxf(min_val, fminf(acc, max_val));
        C[row * N + col] = acc;
    }
}

torch::Tensor fused_linear_scale_add_clamp_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scale, float min_val, float max_val) {

    const int M = input.size(0); // batch_size
    const int K = input.size(1); // input_size
    const int N = weight.size(0); // hidden_size

    auto output = torch::empty({M, N}, input.options());

    dim3 blockDim(TILE_W, TILE_W);
    dim3 gridDim((N + TILE_W - 1) / TILE_W, (M + TILE_W - 1) / TILE_W);

    fused_linear_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        M, N, K, scale, min_val, max_val
    );
    return output;
}
"""

fused_linear_cpp_source = "torch::Tensor fused_linear_scale_add_clamp_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scale, float min_val, float max_val);"

fused_linear_op = load_inline(
    name="fused_linear_op",
    cpp_sources=fused_linear_cpp_source,
    cuda_sources=fused_linear_source,
    functions=["fused_linear_scale_add_clamp_cuda"],
    verbose=False,
)

# --- CUDA Kernel 2: Fused LogSumExp + Mish ---
# This kernel fuses the following operations:
# 1. LogSumExp reduction along dim=1
# 2. Final activation: x * mish(x)
# A numerically stable LogSumExp is implemented using a two-pass reduction within each CUDA block.
# The Mish activation and the final multiplication are fused into the epilogue of the reduction.

fused_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

// Helper for block-wide reduction to find the maximum value
__device__ void block_reduce_max(float& val, float* sdata) {
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // Warp-level reduction using shuffles
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }

    // Store warp maxes in shared memory
    if (lane == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();

    // Reduce warp maxes to find block max (first warp only)
    val = (threadIdx.x < blockDim.x / warpSize) ? sdata[threadIdx.x] : -std::numeric_limits<float>::infinity();
    if (warp_id == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
    }
}

// Helper for block-wide reduction to sum values
__device__ void block_reduce_sum(float& val, float* sdata) {
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // Store warp sums in shared memory
    if (lane == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();

    // Reduce warp sums to find block sum (first warp only)
    val = (threadIdx.x < blockDim.x / warpSize) ? sdata[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
}


__global__ void logsumexp_mish_kernel(const float* input, float* output, int batch_size, int hidden_size) {
    // One block is responsible for processing one row of the input tensor
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_input = input + row * hidden_size;
    
    extern __shared__ float sdata[];

    // --- Pass 1: Find max value for numerical stability ---
    float thread_max = -std::numeric_limits<float>::infinity();
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }
    block_reduce_max(thread_max, sdata);
    
    // Broadcast block max to all threads via shared memory
    if (threadIdx.x == 0) {
        sdata[0] = thread_max;
    }
    __syncthreads();
    const float block_max = sdata[0];

    // --- Pass 2: Sum of exps ---
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        thread_sum += expf(row_input[i] - block_max);
    }
    block_reduce_sum(thread_sum, sdata);

    // --- Final calculation by thread 0 ---
    if (threadIdx.x == 0) {
        float block_sum = thread_sum;
        float lse = block_max + logf(block_sum);
        
        // Fused epilogue: x * mish(x)
        // Mish activation: x * tanh(softplus(x))
        // softplus(x) = log(1 + exp(x))
        float softplus_lse = logf(1.0f + expf(lse));
        float mish_lse = lse * tanhf(softplus_lse);
        
        // Final op: lse * mish(lse)
        output[row] = lse * mish_lse;
    }
}

torch::Tensor custom_logsumexp_mish_cuda(torch::Tensor input) {
    const int batch_size = input.size(0);
    const int hidden_size = input.size(1);

    auto output = torch::empty({batch_size, 1}, input.options());

    const int threads_per_block = 256;
    const int num_blocks = batch_size;
    
    int num_warps = threads_per_block / 32;
    size_t shared_mem_size = num_warps * sizeof(float);

    logsumexp_mish_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), batch_size, hidden_size
    );

    return output;
}
"""

fused_reduction_cpp_source = "torch::Tensor custom_logsumexp_mish_cuda(torch::Tensor input);"

fused_reduction_op = load_inline(
    name="fused_reduction_op",
    cpp_sources=fused_reduction_cpp_source,
    cuda_sources=fused_reduction_source,
    functions=["custom_logsumexp_mish_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that replaces the original PyTorch operator sequence with two custom,
    fused CUDA kernels for improved performance.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        # We still use nn.Linear to manage the weight and bias parameters
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        # The original `x + x` is equivalent to `x * 2`. We fuse this into the scale factor.
        combined_scale = self.scale_factor * 2.0

        # Call the first fused kernel for linear, scale, add, and clamp
        x = fused_linear_op.fused_linear_scale_add_clamp_cuda(
            x, self.matmul.weight, self.matmul.bias,
            combined_scale, self.clamp_min, self.clamp_max
        )

        # Call the second fused kernel for LogSumExp and the final Mish activation
        x = fused_reduction_op.custom_logsumexp_mish_cuda(x)
        
        return x