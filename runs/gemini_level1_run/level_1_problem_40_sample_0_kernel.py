import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for a fused Layer Normalization
layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// CUDA kernel for forward pass of Layer Normalization
// Fuses the calculation of mean/variance with the normalization step
__global__ void layer_norm_fwd_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    float eps,
    int M) {

    // Shared memory for reduction
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sum_sq = &sdata[blockDim.x];

    // Each block processes one row (one item in the batch)
    int row_idx = blockIdx.x;
    const float* x_row = x + row_idx * M;
    float* y_row = y + row_idx * M;

    // Step 1: Parallel reduction to find sum and sum of squares
    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        float val = x_row[i];
        thread_sum += val;
        thread_sum_sq += val * val;
    }
    s_sum[threadIdx.x] = thread_sum;
    s_sum_sq[threadIdx.x] = thread_sum_sq;
    __syncthreads();

    // Reduce sums in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Step 2: Thread 0 computes mean and inverse standard deviation
    if (threadIdx.x == 0) {
        float mean = s_sum[0] / M;
        float var = (s_sum_sq[0] / M) - (mean * mean);
        // Store mean and rstd in shared memory to broadcast to all threads
        s_sum[0] = mean;
        s_sum_sq[0] = rsqrtf(var + eps);
    }
    __syncthreads();

    // Load mean and rstd from shared memory
    float mean = s_sum[0];
    float rstd = s_sum_sq[0];

    // Step 3: Apply normalization and affine transformation
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        float val = x_row[i];
        // gamma and beta are indexed by the feature dimension, not the full row
        y_row[i] = (val - mean) * rstd * gamma[i] + beta[i];
    }
}

torch::Tensor layer_norm_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps) {

    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "Input gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "Input beta must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "Input gamma must be contiguous");
    TORCH_CHECK(beta.is_contiguous(), "Input beta must be contiguous");

    const auto x_sizes = x.sizes();
    const int normalized_ndim = gamma.dim();
    const int total_ndim = x.dim();

    int N = 1;
    for (int i = 0; i < total_ndim - normalized_ndim; ++i) {
        N *= x_sizes[i];
    }
    int M = 1;
    for (int i = 0; i < total_ndim - normalized_ndim; ++i) {
        M *= x_sizes[i + total_ndim - normalized_ndim];
    }

    auto y = torch::empty_like(x);

    // Heuristic for block size
    const int block_size = 1024;
    const int num_blocks = N;
    const size_t smem_size = 2 * block_size * sizeof(float);

    layer_norm_fwd_kernel<<<num_blocks, block_size, smem_size>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<float>(eps),
        M
    );

    return y;
}
"""

layernorm_cpp_source = "torch::Tensor layer_norm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps);"

# Compile the inline CUDA code for Layer Normalization
custom_layernorm = load_inline(
    name="custom_layernorm",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_source,
    functions=["layer_norm_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Model that performs Layer Normalization using a custom fused CUDA kernel.
    """
    def __init__(self, normalized_shape: tuple, eps: float = 1e-5):
        """
        Initializes the custom LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
            eps (float): A small value added to the denominator for numerical stability.
        """
        super(ModelNew, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies custom Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return custom_layernorm.layer_norm_cuda(x, self.weight, self.bias, self.eps)