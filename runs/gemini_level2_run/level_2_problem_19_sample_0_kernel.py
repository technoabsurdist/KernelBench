import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GELU + GroupNorm
fused_gelu_groupnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// GELU device function using error function (matches PyTorch)
__device__ __forceinline__ float gelu_erf_impl(float x) {
    return x * 0.5f * (1.0f + erff(x * 0.7071067811865475f)); // 1/sqrt(2)
}

// Kernel to compute mean and inv_std after applying GELU
// Each block processes one group for one batch item.
// Grid: (N, num_groups), Block: (threads_per_block)
__global__ void gelu_group_norm_stats_kernel(
    const float* __restrict__ x,
    float* __restrict__ mean,
    float* __restrict__ inv_std,
    const int N, const int C, const int H, const int W,
    const int num_groups, const float eps) {

    const int batch_idx = blockIdx.x;
    const int group_idx = blockIdx.y;

    const int num_channels_per_group = C / num_groups;
    const int group_size = num_channels_per_group * H * W;

    // Shared memory for reduction
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sum_sq = &sdata[blockDim.x];

    // Initialize shared memory
    if (threadIdx.x < blockDim.x) {
        s_sum[threadIdx.x] = 0.0f;
        s_sum_sq[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;

    const int C_offset = (group_idx * num_channels_per_group) * H * W;
    const int N_offset = batch_idx * C * H * W;
    const float* x_group_ptr = x + N_offset + C_offset;

    // Loop over the elements in the group, applying GELU and accumulating sums
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float val = x_group_ptr[i];
        float gelu_val = gelu_erf_impl(val);

        thread_sum += gelu_val;
        thread_sum_sq += gelu_val * gelu_val;
    }

    s_sum[threadIdx.x] = thread_sum;
    s_sum_sq[threadIdx.x] = thread_sum_sq;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 computes and writes the final mean and inv_std
    if (threadIdx.x == 0) {
        float group_sum = s_sum[0];
        float group_sum_sq = s_sum_sq[0];

        float mu = group_sum / group_size;
        float var = group_sum_sq / group_size - mu * mu;
        float rstd = rsqrtf(var + eps);

        int stats_idx = batch_idx * num_groups + group_idx;
        mean[stats_idx] = mu;
        inv_std[stats_idx] = rstd;
    }
}

// Kernel to apply the normalization using the pre-computed stats
// Each thread processes one element of the output tensor.
// Grid: (total_elements), Block: (threads_per_block)
__global__ void gelu_group_norm_apply_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    const int N, const int C, const int H, const int W,
    const int num_groups) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = N * C * H * W;
    if (idx >= total_elements) return;

    // Decompose linear index to N,C,H,W
    const int w = idx % W;
    const int h = (idx / W) % H;
    const int c = (idx / (W * H)) % C;
    const int n = idx / (C * W * H);

    const int num_channels_per_group = C / num_groups;
    const int group_idx = c / num_channels_per_group;

    const int stats_idx = n * num_groups + group_idx;
    const float mu = mean[stats_idx];
    const float rstd = inv_std[stats_idx];

    // Re-compute GELU to avoid storing intermediate tensor
    const float val = x[idx];
    const float gelu_val = gelu_erf_impl(val);

    const float gamma_val = gamma[c];
    const float beta_val = beta[c];

    y[idx] = (gelu_val - mu) * rstd * gamma_val + beta_val;
}

// C++ wrapper function to be called from Python
torch::Tensor gelu_group_norm_cuda(
    torch::Tensor x,
    int64_t num_groups,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps) {

    TORCH_CHECK(x.is_cuda(), "Input tensor 'x' must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor 'x' must be contiguous");
    TORCH_CHECK(gamma.is_cuda(), "Gamma tensor must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "Beta tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor 'x' must be of type float32");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32, "Gamma tensor must be of type float32");
    TORCH_CHECK(beta.scalar_type() == torch::kFloat32, "Beta tensor must be of type float32");

    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);

    TORCH_CHECK(C % num_groups == 0, "Number of channels must be divisible by num_groups");

    auto y = torch::empty_like(x);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto mean = torch::empty({N, num_groups}, options);
    auto inv_std = torch::empty({N, num_groups}, options);

    // --- First Pass: Calculate Stats ---
    const int threads_per_block_stats = 512;
    const dim3 blocks_stats(N, num_groups);
    // Shared memory size: (sum + sum_sq) * threads * sizeof(float)
    const int shared_mem_size = 2 * threads_per_block_stats * sizeof(float);

    gelu_group_norm_stats_kernel<<<blocks_stats, threads_per_block_stats, shared_mem_size>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        N, C, H, W,
        num_groups,
        static_cast<float>(eps)
    );

    // --- Second Pass: Apply Normalization ---
    const int total_elements = N * C * H * W;
    const int threads_per_block_apply = 256;
    const int blocks_apply = (total_elements + threads_per_block_apply - 1) / threads_per_block_apply;

    gelu_group_norm_apply_kernel<<<blocks_apply, threads_per_block_apply>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        num_groups
    );
    
    // Check for any CUDA errors that might have occurred during kernel launches
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return y;
}
"""

fused_gelu_groupnorm_cpp_source = """
torch::Tensor gelu_group_norm_cuda(
    torch::Tensor x,
    int64_t num_groups,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps);
"""

# JIT compile the inline CUDA code
fused_gelu_groupnorm_op = load_inline(
    name="fused_gelu_groupnorm",
    cpp_sources=fused_gelu_groupnorm_cpp_source,
    cuda_sources=fused_gelu_groupnorm_source,
    functions=["gelu_group_norm_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that replaces GELU and GroupNorm with a single fused CUDA kernel.
    The ConvTranspose2d layer remains as the standard PyTorch implementation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        # Use the standard, highly optimized ConvTranspose2d from PyTorch
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        
        # Store parameters for the custom GroupNorm part of the fused kernel
        self.num_groups = num_groups
        self.eps = 1e-5  # Standard epsilon for nn.GroupNorm

        # Replicate the learnable parameters from nn.GroupNorm
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # 1. Perform transposed convolution
        x = self.conv_transpose(x)
        
        # 2. Apply the custom fused GELU + GroupNorm operation
        x = fused_gelu_groupnorm_op.gelu_group_norm_cuda(
            x, self.num_groups, self.weight, self.bias, self.eps
        )
        return x