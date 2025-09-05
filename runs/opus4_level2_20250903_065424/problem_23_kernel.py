import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GroupNorm + Mean
fused_groupnorm_mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void fused_groupnorm_mean_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int num_channels,
    int num_groups,
    int spatial_size,
    float eps
) {
    extern __shared__ float shared_data[];
    
    int batch_idx = blockIdx.y;
    int group_idx = blockIdx.x;
    
    int channels_per_group = num_channels / num_groups;
    int group_size = channels_per_group * spatial_size;
    
    float* s_mean = shared_data;
    float* s_var = &shared_data[blockDim.x];
    float* s_sum = &shared_data[2 * blockDim.x];
    
    int tid = threadIdx.x;
    
    // Compute mean for this group
    float sum = 0.0f;
    for (int i = tid; i < group_size; i += blockDim.x) {
        int c = i / spatial_size;
        int s = i % spatial_size;
        int global_c = group_idx * channels_per_group + c;
        int idx = batch_idx * num_channels * spatial_size + global_c * spatial_size + s;
        sum += input[idx];
    }
    s_mean[tid] = sum;
    __syncthreads();
    
    // Reduce mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mean[tid] += s_mean[tid + stride];
        }
        __syncthreads();
    }
    float mean = s_mean[0] / group_size;
    
    // Compute variance for this group
    float var_sum = 0.0f;
    for (int i = tid; i < group_size; i += blockDim.x) {
        int c = i / spatial_size;
        int s = i % spatial_size;
        int global_c = group_idx * channels_per_group + c;
        int idx = batch_idx * num_channels * spatial_size + global_c * spatial_size + s;
        float diff = input[idx] - mean;
        var_sum += diff * diff;
    }
    s_var[tid] = var_sum;
    __syncthreads();
    
    // Reduce variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_var[tid] += s_var[tid + stride];
        }
        __syncthreads();
    }
    float var = s_var[0] / group_size;
    float std = sqrtf(var + eps);
    
    // Apply normalization and accumulate for mean
    float local_sum = 0.0f;
    for (int i = tid; i < group_size; i += blockDim.x) {
        int c = i / spatial_size;
        int s = i % spatial_size;
        int global_c = group_idx * channels_per_group + c;
        int idx = batch_idx * num_channels * spatial_size + global_c * spatial_size + s;
        float normalized = (input[idx] - mean) / std;
        float scaled = normalized * weight[global_c] + bias[global_c];
        local_sum += scaled;
    }
    s_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduce for final mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(&output[batch_idx], s_sum[0] / (num_channels * spatial_size));
    }
}

torch::Tensor fused_groupnorm_mean_cuda(
    torch::Tensor input,
    torch::Tensor weight, 
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    auto spatial_size = D * H * W;
    
    auto output = torch::zeros({batch_size, 1}, input.options());
    
    const int threads = 256;
    const int shared_mem = 3 * threads * sizeof(float);
    dim3 blocks(num_groups, batch_size);
    
    fused_groupnorm_mean_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_channels,
        num_groups,
        spatial_size,
        eps
    );
    
    return output;
}
"""

fused_groupnorm_mean_cpp_source = """
torch::Tensor fused_groupnorm_mean_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
);
"""

# Compile the inline CUDA code
fused_groupnorm_mean = load_inline(
    name="fused_groupnorm_mean",
    cpp_sources=fused_groupnorm_mean_cpp_source,
    cuda_sources=fused_groupnorm_mean_source,
    functions=["fused_groupnorm_mean_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused GroupNorm + Mean kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.num_groups = num_groups
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.fused_op = fused_groupnorm_mean

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.conv(x)
        x = self.fused_op.fused_groupnorm_mean_cuda(
            x, self.weight, self.bias, self.num_groups, self.eps
        )
        return x.view(-1)

batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
num_groups = 8

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups]