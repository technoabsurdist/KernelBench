import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: Swish -> GroupNorm -> HardSwish
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Device function for sigmoid, used in Swish
__device__ inline float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Device function for HardSwish
__device__ inline float hardswishf(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

// Fused kernel that performs Swish, Group Normalization, and HardSwish in one pass.
__global__ void fused_swish_groupnorm_hardswish_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const int N, const int C, const int D, const int H, const int W,
    const int G,
    const float eps) {

    // Use dynamically allocated shared memory for reduction
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sum_sq = &sdata[blockDim.x];

    // Each block processes one group for one batch item
    const int group_idx = blockIdx.x;
    const int batch_idx = group_idx / G;
    const int group_in_batch_idx = group_idx % G;

    const int channels_per_group = C / G;
    const int plane_size = D * H * W;
    const int group_size = channels_per_group * plane_size;

    // --- Step 1: Parallel reduction to find mean and variance of the Swish-activated input ---
    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;

    // Grid-stride loop for reduction
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        const int plane_idx = i % plane_size;
        const int channel_in_group_idx = i / plane_size;
        const int channel_idx = group_in_batch_idx * channels_per_group + channel_in_group_idx;
        const int data_idx = batch_idx * C * plane_size + channel_idx * plane_size + plane_idx;

        // Apply Swish activation before summing
        float val = input[data_idx];
        float swish_val = val * sigmoidf(val);

        thread_sum += swish_val;
        thread_sum_sq += swish_val * swish_val;
    }

    s_sum[threadIdx.x] = thread_sum;
    s_sum_sq[threadIdx.x] = thread_sum_sq;
    __syncthreads();

    // In-block reduction using shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 computes mean and inverse standard deviation for the group
    float mean, inv_stddev;
    if (threadIdx.x == 0) {
        mean = s_sum[0] / group_size;
        float var = s_sum_sq[0] / group_size - mean * mean;
        inv_stddev = rsqrtf(var + eps);
        
        // Store back to shared memory for other threads in the block to access
        s_sum[0] = mean;
        s_sum_sq[0] = inv_stddev;
    }
    __syncthreads();

    mean = s_sum[0];
    inv_stddev = s_sum_sq[0];

    // --- Step 2: Apply normalization, affine transform (gamma/beta), and HardSwish ---
    // Grid-stride loop for applying the transformation
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        const int plane_idx = i % plane_size;
        const int channel_in_group_idx = i / plane_size;
        const int channel_idx = group_in_batch_idx * channels_per_group + channel_in_group_idx;
        const int data_idx = batch_idx * C * plane_size + channel_idx * plane_size + plane_idx;

        // Re-compute Swish. This is a trade-off to avoid a separate global memory write/read.
        float val = input[data_idx];
        float swish_val = val * sigmoidf(val);

        // Apply GroupNorm
        float norm_val = (swish_val - mean) * inv_stddev;
        float scaled_val = norm_val * gamma[channel_idx] + beta[channel_idx];

        // Apply HardSwish and write to output
        output[data_idx] = hardswishf(scaled_val);
    }
}
"""

fused_op_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int groups,
    double eps);
"""

# JIT compile the CUDA and C++ code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

# C++ wrapper function that launches the CUDA kernel
# This C++ code is defined as a string and passed to the JIT compiler
# It's separate from the CUDA source for clarity
fused_op_launcher_cpp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Forward declaration of the CUDA kernel
void fused_swish_groupnorm_hardswish_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const int N, const int C, const int D, const int H, const int W,
    const int G,
    const float eps);

torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int groups,
    double eps) {

    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "Gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "Beta must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 5, "Input must be a 5D tensor (N, C, D, H, W)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32, "Gamma must be a float32 tensor");
    TORCH_CHECK(beta.scalar_type() == torch::kFloat32, "Beta must be a float32 tensor");

    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto D = input.size(2);
    const auto H = input.size(3);
    const auto W = input.size(4);

    TORCH_CHECK(C % groups == 0, "Number of channels must be divisible by groups");
    TORCH_CHECK(gamma.numel() == C, "Gamma must have C elements");
    TORCH_CHECK(beta.numel() == C, "Beta must have C elements");

    auto output = torch::empty_like(input);

    const int block_size = 512; // A common, effective block size
    const int grid_size = N * groups; // Launch one block per group in the batch

    // Shared memory size: 2 * block_size * sizeof(float) for sum and sum_sq
    const int shared_mem_size = 2 * block_size * sizeof(float);

    fused_swish_groupnorm_hardswish_kernel<<<grid_size, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        N, C, D, H, W,
        groups,
        static_cast<float>(eps)
    );

    // Check for any CUDA errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

# JIT compile the complete C++/CUDA source
fused_op = load_inline(
    name="fused_op_v2", # Use a different name to avoid conflicts if run in the same session
    cpp_sources=fused_op_launcher_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses Swish, GroupNorm, and HardSwish into a single custom CUDA kernel.
    The ConvTranspose3d operation remains as the standard PyTorch implementation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        # Use the highly optimized standard ConvTranspose3d implementation
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        
        # Manually define the learnable parameters (weight and bias) for the GroupNorm part of the fused op
        self.group_norm_weight = nn.Parameter(torch.ones(out_channels))
        self.group_norm_bias = nn.Parameter(torch.zeros(out_channels))
        
        # Store non-tensor parameters
        self.groups = groups
        self.eps = eps
        
        # Store the compiled custom CUDA function
        self.fused_op = fused_op

    def forward(self, x):
        # Step 1: Perform the 3D transposed convolution
        x = self.conv_transpose(x)
        
        # Step 2: Call the single fused CUDA kernel for the remaining operations
        # The kernel handles: Swish -> GroupNorm -> HardSwish
        return self.fused_op.fused_op_cuda(
            x, 
            self.group_norm_weight, 
            self.group_norm_bias, 
            self.groups, 
            self.eps
        )