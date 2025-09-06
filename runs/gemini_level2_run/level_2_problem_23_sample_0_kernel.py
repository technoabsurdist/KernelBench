import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GroupNorm + Mean
fused_gn_mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel 1: Calculate mean and inv_stddev for each group
__global__ void group_stats_kernel(
    const float* __restrict__ x,
    float* __restrict__ means,
    float* __restrict__ inv_stddevs,
    const int B, const int C, const int D, const int H, const int W,
    const int num_groups, const float eps) {

    const int g = blockIdx.y; // group index
    const int b = blockIdx.x; // batch index

    const int channels_per_group = C / num_groups;
    const long elements_per_group = (long)channels_per_group * D * H * W;

    // Pointers to the start of the data for this batch item
    const float* x_batch = x + (long)b * C * D * H * W;

    // Reduction for sum and sum_sq
    double local_sum = 0.0;
    double local_sum_sq = 0.0;

    for (long i = threadIdx.x; i < elements_per_group; i += blockDim.x) {
        // Decompose i into c_group, d, h, w
        long spatial_idx = i % ((long)D * H * W);
        int c_group = i / ((long)D * H * W);
        int c_global = g * channels_per_group + c_group;

        long global_idx = (long)c_global * D * H * W + spatial_idx;
        float val = x_batch[global_idx];
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Block-level reduction
    extern __shared__ double sdata[];
    double* s_sum = sdata;
    double* s_sum_sq = (double*)&sdata[blockDim.x];

    s_sum[threadIdx.x] = local_sum;
    s_sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final result
    if (threadIdx.x == 0) {
        double group_sum = s_sum[0];
        double group_sum_sq = s_sum_sq[0];

        float mean = group_sum / elements_per_group;
        float var = group_sum_sq / elements_per_group - mean * mean;
        float inv_stddev = rsqrtf(var + eps);

        int group_output_idx = b * num_groups + g;
        means[group_output_idx] = mean;
        inv_stddevs[group_output_idx] = inv_stddev;
    }
}

// Kernel 2: Normalize, apply affine transform, and reduce to final mean
__global__ void normalize_and_reduce_mean_kernel(
    const float* __restrict__ x,
    const float* __restrict__ means,
    const float* __restrict__ inv_stddevs,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    const int B, const int C, const int D, const int H, const int W,
    const int num_groups) {

    const int b = blockIdx.x; // batch index

    const int channels_per_group = C / num_groups;
    const long elements_per_batch = (long)C * D * H * W;

    // Pointers to the start of the data for this batch item
    const float* x_batch = x + b * elements_per_batch;
    const float* means_batch = means + b * num_groups;
    const float* inv_stddevs_batch = inv_stddevs + b * num_groups;

    double local_sum = 0.0;

    for (long i = threadIdx.x; i < elements_per_batch; i += blockDim.x) {
        // Decompose i into c
        int c = i / ((long)D * H * W);
        int g = c / channels_per_group;

        float val = x_batch[i];
        float mean = means_batch[g];
        float inv_stddev = inv_stddevs_batch[g];

        // Normalize
        float normalized_val = (val - mean) * inv_stddev;

        // Affine transform
        float transformed_val = normalized_val * gamma[c] + beta[c];

        local_sum += transformed_val;
    }

    // Block-level reduction
    extern __shared__ double s_sum[];
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final result
    if (threadIdx.x == 0) {
        double total_sum = s_sum[0];
        out[b] = total_sum / elements_per_batch;
    }
}

// C++ wrapper
torch::Tensor fused_groupnorm_mean_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t num_groups,
    double eps) {

    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "Input gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "Input beta must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input x must be a float32 tensor");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32, "Input gamma must be a float32 tensor");
    TORCH_CHECK(beta.scalar_type() == torch::kFloat32, "Input beta must be a float32 tensor");
    TORCH_CHECK(x.dim() == 5, "Input x must be a 5D tensor");

    const auto B = x.size(0);
    const auto C = x.size(1);
    const auto D = x.size(2);
    const auto H = x.size(3);
    const auto W = x.size(4);

    TORCH_CHECK(C > 0 && C % num_groups == 0, "Number of channels must be divisible by num_groups");
    TORCH_CHECK(gamma.numel() == C, "gamma must have C elements");
    TORCH_CHECK(beta.numel() == C, "beta must have C elements");

    // Allocate intermediate and output tensors
    auto options = x.options();
    auto stats_options = options.sizes({B, num_groups});
    auto means = torch::empty({B, num_groups}, stats_options);
    auto inv_stddevs = torch::empty({B, num_groups}, stats_options);
    auto out = torch::empty({B}, options);

    // Kernel launch configuration
    const int threads_per_block = 512;

    // Kernel 1 launch
    const dim3 blocks1(B, num_groups);
    const dim3 threads1(threads_per_block);
    // Shared memory for reduction: one double for sum, one for sum_sq per thread
    size_t smem_size1 = threads_per_block * 2 * sizeof(double);
    group_stats_kernel<<<blocks1, threads1, smem_size1>>>(
        x.data_ptr<float>(),
        means.data_ptr<float>(),
        inv_stddevs.data_ptr<float>(),
        B, C, D, H, W,
        num_groups,
        static_cast<float>(eps)
    );

    // Kernel 2 launch
    const dim3 blocks2(B);
    const dim3 threads2(threads_per_block);
    // Shared memory for reduction: one double for sum per thread
    size_t smem_size2 = threads_per_block * sizeof(double);
    normalize_and_reduce_mean_kernel<<<blocks2, threads2, smem_size2>>>(
        x.data_ptr<float>(),
        means.data_ptr<float>(),
        inv_stddevs.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, D, H, W,
        num_groups
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

fused_gn_mean_cpp_source = """
torch::Tensor fused_groupnorm_mean_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t num_groups,
    double eps);
"""

# Compile the inline CUDA code
fused_gn_mean = load_inline(
    name="fused_gn_mean",
    cpp_sources=fused_gn_mean_cpp_source,
    cuda_sources=fused_gn_mean_source,
    functions=["fused_groupnorm_mean_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, then a fused GroupNorm and Mean reduction.
    This version replaces the standard GroupNorm and mean operations with a single
    fused custom CUDA kernel to improve performance by reducing memory I/O and
    kernel launch overhead.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # We still need the GroupNorm layer to hold the learnable parameters
        # (weight, bias) and provide its configuration (num_groups, eps).
        # The forward pass of this layer itself will not be called.
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        # The custom fused operator
        self.fused_op = fused_gn_mean.fused_groupnorm_mean_cuda

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        x = self.conv(x)
        # Call the custom fused kernel instead of the separate group_norm and mean calls
        x = self.fused_op(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.group_norm.eps
        )
        return x