import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: GEMM + Scaling + BatchNorm
fused_gemm_scale_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

// CUDA kernel to perform a tiled matrix multiplication and fuse it with scaling and batch normalization.
// The operation is equivalent to:
// y = (x @ W.T + gemm_bias) * scale
// z = batch_norm(y)
// This kernel is optimized for inference mode.
// The mathematical simplification is:
// z = (x @ W.T) * final_scale + final_bias
// where final_scale and final_bias are pre-computed from the original parameters.
__global__ void fused_gemm_scale_bn_kernel(
    const float* A,          // Input tensor x (M x K)
    const float* B,          // Transposed weight tensor W.T (K x N)
    float* C,                // Output tensor (M x N)
    const float* final_scale,// Fused scale vector (N)
    const float* final_bias, // Fused bias vector (N)
    int M, int N, int K) {

    // Shared memory for tiles of A and B
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global row and column for this thread's output element
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // Accumulator for the dot product
    float acc = 0.0f;

    // Loop over the tiles of A and B required to compute the C tile
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Load a tile of A into shared memory
        int a_col = t * TILE_WIDTH + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load a tile of B into shared memory
        int b_row = t * TILE_WIDTH + ty;
        if (col < N && b_row < K) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Multiply the tiles from shared memory
        for (int k = 0; k < TILE_WIDTH; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Apply the fused post-processing operations (scaling and bias) and write to global memory
    if (row < M && col < N) {
        C[row * N + col] = acc * final_scale[col] + final_bias[col];
    }
}

// C++ wrapper function that PyTorch will call
torch::Tensor fused_gemm_scale_bn_cuda(
    torch::Tensor x,
    torch::Tensor gemm_weight,
    torch::Tensor gemm_bias,
    torch::Tensor scale,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    double bn_eps) {

    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");

    const auto batch_size = x.size(0);
    const auto in_features = x.size(1);
    const auto out_features = gemm_weight.size(0);

    // Pre-compute the fused scale and bias parameters on the GPU
    // inv_std = 1.0 / sqrt(var + eps)
    const auto inv_std = (bn_running_var + bn_eps).rsqrt();

    // final_scale = (scale * inv_std * bn.weight)
    const auto final_scale = scale * inv_std * bn_weight;

    // final_bias = (gemm_bias * scale - bn_running_mean) * inv_std * bn_weight + bn_bias
    const auto final_bias = (gemm_bias * scale - bn_running_mean) * inv_std * bn_weight + bn_bias;

    // Allocate the output tensor
    auto out = torch::empty({batch_size, out_features}, x.options());

    // The kernel expects the weight matrix to be transposed (K x N)
    // nn.Linear stores weight as (out_features, in_features), so we transpose it.
    // .contiguous() is important for performance as it ensures coalesced memory access.
    auto gemm_weight_t = gemm_weight.transpose(0, 1).contiguous();

    // Setup grid and block dimensions for the CUDA kernel
    const dim3 threads(TILE_WIDTH, TILE_WIDTH);
    const dim3 blocks(
        (out_features + TILE_WIDTH - 1) / TILE_WIDTH,
        (batch_size + TILE_WIDTH - 1) / TILE_WIDTH
    );

    // Launch the kernel
    fused_gemm_scale_bn_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        gemm_weight_t.data_ptr<float>(),
        out.data_ptr<float>(),
        final_scale.data_ptr<float>(),
        final_bias.data_ptr<float>(),
        batch_size,
        out_features,
        in_features
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
"""

fused_gemm_scale_bn_cpp_source = """
torch::Tensor fused_gemm_scale_bn_cuda(
    torch::Tensor x,
    torch::Tensor gemm_weight,
    torch::Tensor gemm_bias,
    torch::Tensor scale,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    double bn_eps);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_gemm_scale_bn",
    cpp_sources=fused_gemm_scale_bn_cpp_source,
    cuda_sources=fused_gemm_scale_bn_source,
    functions=["fused_gemm_scale_bn_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses GEMM, scaling, and batch normalization into a single CUDA kernel for inference.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # We keep the original layers to hold the parameters (weights, biases, etc.)
        # and to provide a fallback path for training mode.
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        # During training, we use the standard PyTorch layers. This ensures that
        # gradients are computed correctly and the batch norm's running statistics
        # are updated.
        if self.training:
            x = self.gemm(x)
            x = x * self.scale
            x = self.bn(x)
            return x
        else:
            # During inference (model.eval()), we use our highly optimized fused CUDA kernel.
            # This avoids launching multiple kernels and writing intermediate results to global memory,
            # leading to significant speedups.
            return fused_op.fused_gemm_scale_bn_cuda(
                x,
                self.gemm.weight,
                self.gemm.bias,
                self.scale,
                self.bn.weight,
                self.bn.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps
            )