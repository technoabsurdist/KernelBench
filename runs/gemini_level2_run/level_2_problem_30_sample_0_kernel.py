import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GroupNorm and HardTanh
fused_groupnorm_hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel to fuse Group Normalization and HardTanh activation
__global__ void groupnorm_hardtanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int num_groups,
    const int channels_per_group,
    const float min_val,
    const float max_val,
    const float eps
) {
    // One block per batch item
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Dynamically sized shared memory for group-wise statistics
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sum_sq = &sdata[num_groups];

    // Initialize shared memory for sums
    if (tid < num_groups) {
        s_sum[tid] = 0.0f;
        s_sum_sq[tid] = 0.0f;
    }
    __syncthreads();

    // --- Step 1: Calculate sum and sum of squares for each group in parallel ---
    // Each thread loops over the channels it's responsible for
    const float* input_row = input + batch_idx * channels;
    for (int j = tid; j < channels; j += block_size) {
        const float val = input_row[j];
        const int group_idx = j / channels_per_group;
        atomicAdd(&s_sum[group_idx], val);
        atomicAdd(&s_sum_sq[group_idx], val * val);
    }
    __syncthreads();

    // --- Step 2: Calculate mean and inverse standard deviation ---
    // Reuse shared memory for mean and inv_std
    float* s_mean = s_sum;
    float* s_inv_std = s_sum_sq;

    if (tid < num_groups) {
        const float sum = s_sum[tid];
        const float sum_sq = s_sum_sq[tid];
        const float mean = sum / channels_per_group;
        const float var = sum_sq / channels_per_group - mean * mean;
        s_mean[tid] = mean;
        s_inv_std[tid] = rsqrtf(var + eps);
    }
    __syncthreads();

    // --- Step 3: Apply normalization, scale/shift, and HardTanh activation ---
    float* output_row = output + batch_idx * channels;
    for (int j = tid; j < channels; j += block_size) {
        const int group_idx = j / channels_per_group;
        const float mean = s_mean[group_idx];
        const float inv_std = s_inv_std[group_idx];

        const float normalized_val = (input_row[j] - mean) * inv_std;
        const float scaled_val = normalized_val * gamma[j] + beta[j];
        
        // Apply HardTanh
        output_row[j] = fminf(fmaxf(scaled_val, min_val), max_val);
    }
}

// C++ wrapper function
torch::Tensor groupnorm_hardtanh_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float hardtanh_min,
    float hardtanh_max,
    float eps
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(gamma.is_cuda() && gamma.is_contiguous() && gamma.scalar_type() == torch::kFloat32, "Gamma must be a contiguous CUDA float32 tensor");
    TORCH_CHECK(beta.is_cuda() && beta.is_contiguous() && beta.scalar_type() == torch::kFloat32, "Beta must be a contiguous CUDA float32 tensor");

    const auto batch_size = input.size(0);
    const auto channels = input.size(1);

    TORCH_CHECK(channels % num_groups == 0, "Number of channels must be divisible by num_groups");
    TORCH_CHECK(gamma.numel() == channels, "Gamma must have size equal to number of channels");
    TORCH_CHECK(beta.numel() == channels, "Beta must have size equal to number of channels");

    auto output = torch::empty_like(input);

    const int channels_per_group = channels / num_groups;
    const int threads_per_block = 256;
    const int grid_dim = batch_size;
    
    // Shared memory size: 2 * num_groups * sizeof(float) for sum and sum_sq
    const int shared_mem_size = 2 * num_groups * sizeof(float);

    // Launch the kernel
    groupnorm_hardtanh_kernel<<<grid_dim, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)batch_size,
        (int)channels,
        num_groups,
        channels_per_group,
        hardtanh_min,
        hardtanh_max,
        eps
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

fused_groupnorm_hardtanh_cpp_source = """
torch::Tensor groupnorm_hardtanh_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float hardtanh_min,
    float hardtanh_max,
    float eps
);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_groupnorm_hardtanh",
    cpp_sources=fused_groupnorm_hardtanh_cpp_source,
    cuda_sources=fused_groupnorm_hardtanh_source,
    functions=["groupnorm_hardtanh_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a GEMM, then a fused Group Normalization and HardTanh.
    The GroupNorm and HardTanh operations are replaced by a single custom CUDA kernel.
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        # Use the standard, highly optimized PyTorch Linear layer for GEMM
        self.gemm = nn.Linear(in_features, out_features)
        
        # Parameters for GroupNorm (gamma and beta)
        self.weight = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Store configuration
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.eps = 1e-5  # Standard epsilon for GroupNorm

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # 1. Perform GEMM using the efficient torch.nn.Linear
        x = self.gemm(x)
        
        # 2. Apply the fused GroupNorm + HardTanh custom kernel
        x = fused_op.groupnorm_hardtanh_cuda(
            x, self.weight, self.bias, self.num_groups,
            self.hardtanh_min, self.hardtanh_max, self.eps
        )
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 16
hardtanh_min = -2.0
hardtanh_max = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, num_groups, hardtanh_min, hardtanh_max]