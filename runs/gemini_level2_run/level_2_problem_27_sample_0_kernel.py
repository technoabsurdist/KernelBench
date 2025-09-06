import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel source for the fused operation
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// --- Device-side utility functions ---

// Hardswish activation function
__device__ __forceinline__ float hardswish(float x) {
    return x * fmaxf(0.0f, fminf(6.0f, x + 3.0f)) / 6.0f;
}

// Block-wide sum reduction using shared memory
template <typename T>
__device__ __forceinline__ T block_reduce_sum(T val, T* shared_mem) {
    int tid = threadIdx.x;
    shared_mem[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    return shared_mem[0];
}


// --- Kernel 1: Calculate Mean and Inverse Standard Deviation per group ---
// This kernel computes statistics over each group after applying HardSwish.
// Grid: (BatchSize, NumGroups), Block: (threads_per_block)
__global__ void calculate_group_stats_kernel(
    const float* __restrict__ x,
    float* __restrict__ mean_out,
    float* __restrict__ inv_std_out,
    const int B, const int C, const int D, const int H, const int W,
    const int G, const float eps) {

    // Each block processes one group for one batch item
    const int batch_idx = blockIdx.x;
    const int group_idx = blockIdx.y;

    const int C_per_G = C / G;
    const int spatial_size = D * H * W;
    const long N = (long)C_per_G * spatial_size; // Total elements per group

    // Pointer to the start of the current group's data
    const int start_channel = group_idx * C_per_G;
    const float* group_x_ptr = x + (long)batch_idx * C * spatial_size + (long)start_channel * spatial_size;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // Parallel reduction over all elements in the group
    for (long i = threadIdx.x; i < N; i += blockDim.x) {
        // Unravel index i to (channel_in_group, spatial_offset)
        int c_local = i / spatial_size;
        int spatial_offset = i % spatial_size;
        
        // Read value, apply hardswish, and accumulate for stats
        float val = group_x_ptr[(long)c_local * spatial_size + spatial_offset];
        float val_hs = hardswish(val);
        local_sum += val_hs;
        local_sum_sq += val_hs * val_hs;
    }

    // Reduce sum and sum_sq across the block using shared memory
    extern __shared__ float shared_mem[];
    float* shared_sum = shared_mem;
    float* shared_sum_sq = &shared_mem[blockDim.x];

    float total_sum = block_reduce_sum(local_sum, shared_sum);
    float total_sum_sq = block_reduce_sum(local_sum_sq, shared_sum_sq);

    // Thread 0 calculates final stats and writes to global memory
    if (threadIdx.x == 0) {
        float mean = total_sum / N;
        float var = total_sum_sq / N - mean * mean;
        float inv_std = rsqrtf(var + eps);

        int out_idx = batch_idx * G + group_idx;
        mean_out[out_idx] = mean;
        inv_std_out[out_idx] = inv_std;
    }
}


// --- Kernel 2: Apply GroupNorm and reduce mean over spatial dimensions ---
// This kernel applies the normalization and computes the final channel-wise mean.
// Grid: (BatchSize, NumChannels), Block: (threads_per_block)
__global__ void apply_gn_and_reduce_mean_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    const int B, const int C, const int D, const int H, const int W,
    const int G) {

    // Each block processes one channel for one batch item
    const int batch_idx = blockIdx.x;
    const int channel_idx = blockIdx.y;

    const int C_per_G = C / G;
    const int group_idx = channel_idx / C_per_G;
    const int spatial_size = D * H * W;

    // Load stats for the group and scale/shift parameters for the channel
    const float group_mean = mean[batch_idx * G + group_idx];
    const float group_inv_std = inv_std[batch_idx * G + group_idx];
    const float channel_gamma = gamma[channel_idx];
    const float channel_beta = beta[channel_idx];

    // Pointer to the start of the current channel's data
    const float* channel_x_ptr = x + (long)batch_idx * C * spatial_size + (long)channel_idx * spatial_size;

    float local_sum = 0.0f;

    // Parallel reduction over all spatial elements in the channel
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float val = channel_x_ptr[i];
        float val_hs = hardswish(val);
        float val_gn = channel_gamma * (val_hs - group_mean) * group_inv_std + channel_beta;
        local_sum += val_gn;
    }

    // Reduce sum across the block
    extern __shared__ float shared_mem[];
    float total_sum = block_reduce_sum(local_sum, shared_mem);

    // Thread 0 calculates final mean and writes to output
    if (threadIdx.x == 0) {
        out[batch_idx * C + channel_idx] = total_sum / spatial_size;
    }
}


// --- C++ Wrapper Function ---
// This function is called from Python and orchestrates the kernel launches.
torch::Tensor fused_hardswish_groupnorm_mean_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    double eps) {

    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor x must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "Input tensor gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "Input tensor beta must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 5, "Input tensor x must be 5D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor x must be float32");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32, "Input tensor gamma must be float32");
    TORCH_CHECK(beta.scalar_type() == torch::kFloat32, "Input tensor beta must be float32");

    const int B = x.size(0);
    const int C = x.size(1);
    const int D = x.size(2);
    const int H = x.size(3);
    const int W = x.size(4);
    const int G = num_groups;

    TORCH_CHECK(C % G == 0, "Number of channels must be divisible by number of groups");

    // Create output tensor
    auto out = torch::zeros({B, C}, x.options());

    // Create intermediate tensors for mean and inv_std
    auto stats_options = x.options().dtype(torch::kFloat32);
    auto mean = torch::empty({B, G}, stats_options);
    auto inv_std = torch::empty({B, G}, stats_options);

    // --- Launch Kernel 1: Calculate Stats ---
    const dim3 grid1(B, G);
    const int block_size1 = 512; // Can be tuned for performance
    const int shared_mem_size1 = 2 * block_size1 * sizeof(float); // For sum and sum_sq

    calculate_group_stats_kernel<<<grid1, block_size1, shared_mem_size1>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        B, C, D, H, W, G, static_cast<float>(eps)
    );

    // --- Launch Kernel 2: Apply Norm and Reduce ---
    const dim3 grid2(B, C);
    const int block_size2 = 512; // Can be tuned for performance
    const int shared_mem_size2 = block_size2 * sizeof(float); // For sum

    apply_gn_and_reduce_mean_kernel<<<grid2, block_size2, shared_mem_size2>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, D, H, W, G
    );
    
    // Use PyTorch's macro for checking CUDA errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

fused_ops_cpp_source = """
#include <torch/extension.h>

torch::Tensor fused_hardswish_groupnorm_mean_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    double eps);
"""

# JIT compile the inline CUDA code. This will be done only once.
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_hardswish_groupnorm_mean_cuda"],
    verbose=False, # Set to True for compilation details
)

class ModelNew(nn.Module):
    """
    Model that performs:
    1. Conv3D
    2. Fused (HardSwish + GroupNorm + Mean pooling) using a custom CUDA kernel.
    
    The custom kernel is used during evaluation (inference) for speed.
    The standard PyTorch operators are used during training to leverage
    the built-in autograd functionality.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        
        # We still need the GroupNorm layer to hold the learnable parameters (weight, bias)
        # and its configuration (num_groups, eps).
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)
        
        # For training, we use the standard PyTorch ops to get gradients automatically.
        # For inference (eval mode), we use our highly optimized fused CUDA kernel.
        if self.training:
            x = F.hardswish(x)
            x = self.group_norm(x)
            x = torch.mean(x, dim=[2, 3, 4])
        else:
            # Ensure input is contiguous for the custom kernel
            x = x.contiguous()
            x = fused_ops.fused_hardswish_groupnorm_mean_cuda(
                x, 
                self.group_norm.weight, 
                self.group_norm.bias, 
                self.group_norm.num_groups, 
                self.group_norm.eps
            )
        return x