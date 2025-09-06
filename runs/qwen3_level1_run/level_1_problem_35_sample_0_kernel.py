import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void group_norm_kernel(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    const float* mean,
    const float* rstd,
    int64_t batch_size,
    int64_t channels,
    int64_t spatial_size,
    int64_t group_size,
    int64_t channels_per_group
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * spatial_size;
    
    if (idx < total_elements) {
        int batch_idx = idx / (channels * spatial_size);
        int channel_idx = (idx / spatial_size) % channels;
        int spatial_idx = idx % spatial_size;
        
        int group_idx = channel_idx / channels_per_group;
        
        float normalized = (input[idx] - mean[batch_idx * group_size + group_idx]) * 
                          rstd[batch_idx * group_size + group_idx];
        
        output[idx] = normalized * weight[channel_idx] + bias[channel_idx];
    }
}

__global__ void compute_stats_kernel(
    const float* input,
    float* mean,
    float* var,
    int64_t batch_size,
    int64_t channels,
    int64_t spatial_size,
    int64_t group_size,
    int64_t channels_per_group
) {
    int batch_idx = blockIdx.x;
    int group_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || group_idx >= group_size) return;
    
    int start_channel = group_idx * channels_per_group;
    int end_channel = start_channel + channels_per_group;
    
    float sum = 0.0f;
    int count = 0;
    
    for (int c = start_channel; c < end_channel; c++) {
        for (int s = 0; s < spatial_size; s++) {
            int idx = batch_idx * (channels * spatial_size) + c * spatial_size + s;
            sum += input[idx];
            count++;
        }
    }
    
    mean[batch_idx * group_size + group_idx] = sum / count;
    
    float sum_sq_diff = 0.0f;
    for (int c = start_channel; c < end_channel; c++) {
        for (int s = 0; s < spatial_size; s++) {
            int idx = batch_idx * (channels * spatial_size) + c * spatial_size + s;
            float diff = input[idx] - mean[batch_idx * group_size + group_idx];
            sum_sq_diff += diff * diff;
        }
    }
    
    var[batch_idx * group_size + group_idx] = sum_sq_diff / count;
}

torch::Tensor group_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto spatial_size = 1;
    for (int i = 2; i < input.dim(); i++) {
        spatial_size *= input.size(i);
    }
    
    auto channels_per_group = channels / num_groups;
    
    auto output = torch::zeros_like(input);
    
    auto mean = torch::zeros({batch_size, num_groups}, input.options());
    auto var = torch::zeros({batch_size, num_groups}, input.options());
    auto rstd = torch::zeros({batch_size, num_groups}, input.options());
    
    const int threads_per_block = 256;
    const int num_blocks = (batch_size * channels * spatial_size + threads_per_block - 1) / threads_per_block;
    
    // Compute mean and variance
    dim3 stats_grid(batch_size, 1, 1);
    dim3 stats_block(num_groups, 1, 1);
    
    compute_stats_kernel<<<stats_grid, stats_block>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        batch_size,
        channels,
        spatial_size,
        num_groups,
        channels_per_group
    );
    
    // Compute reciprocal of std
    auto eps_tensor = torch::full_like(var, eps);
    rstd = torch::rsqrt(var + eps_tensor);
    
    // Apply normalization
    group_norm_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        batch_size,
        channels,
        spatial_size,
        num_groups,
        channels_per_group
    );
    
    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps
);
"""

# Compile the inline CUDA code for Group Normalization
group_norm = load_inline(
    name="group_norm",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization with custom CUDA kernels.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return group_norm.group_norm_cuda(x, self.weight, self.bias, self.num_groups, self.eps)

batch_size = 112  # scaled up
features = 64
num_groups = 8
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [features, num_groups]  # num_features