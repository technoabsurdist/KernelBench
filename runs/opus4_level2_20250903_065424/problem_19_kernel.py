import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GELU + GroupNorm
fused_gelu_groupnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

__device__ float gelu_func(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void compute_group_stats(
    const float* input,
    float* group_mean,
    float* group_var,
    int batch_size,
    int num_channels,
    int num_groups,
    int spatial_size,
    int channels_per_group
) {
    int group_id = blockIdx.x;
    int batch_id = blockIdx.y;
    
    if (group_id >= num_groups || batch_id >= batch_size) return;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int elements_per_group = channels_per_group * spatial_size;
    
    for (int i = tid; i < elements_per_group; i += num_threads) {
        int channel_offset = i / spatial_size;
        int spatial_offset = i % spatial_size;
        int channel_id = group_id * channels_per_group + channel_offset;
        
        if (channel_id < num_channels) {
            int idx = batch_id * num_channels * spatial_size + 
                     channel_id * spatial_size + spatial_offset;
            float val = gelu_func(input[idx]);
            sum += val;
            sum_sq += val * val;
        }
    }
    
    __shared__ float shared_sum[256];
    __shared__ float shared_sum_sq[256];
    
    shared_sum[tid] = sum;
    shared_sum_sq[tid] = sum_sq;
    __syncthreads();
    
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
            shared_sum_sq[tid] += shared_sum_sq[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        int idx = batch_id * num_groups + group_id;
        group_mean[idx] = shared_sum[0] / elements_per_group;
        float mean = group_mean[idx];
        group_var[idx] = shared_sum_sq[0] / elements_per_group - mean * mean;
    }
}

__global__ void apply_groupnorm_with_gelu(
    const float* input,
    float* output,
    const float* group_mean,
    const float* group_var,
    const float* weight,
    const float* bias,
    int batch_size,
    int num_channels,
    int num_groups,
    int spatial_size,
    int channels_per_group,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_channels * spatial_size;
    
    if (idx >= total_elements) return;
    
    int batch_id = idx / (num_channels * spatial_size);
    int channel_id = (idx / spatial_size) % num_channels;
    int group_id = channel_id / channels_per_group;
    
    float val = gelu_func(input[idx]);
    
    int stats_idx = batch_id * num_groups + group_id;
    float mean = group_mean[stats_idx];
    float var = group_var[stats_idx];
    float std = sqrtf(var + eps);
    
    float normalized = (val - mean) / std;
    
    if (weight != nullptr) {
        normalized = normalized * weight[channel_id];
    }
    if (bias != nullptr) {
        normalized = normalized + bias[channel_id];
    }
    
    output[idx] = normalized;
}

torch::Tensor fused_gelu_groupnorm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto spatial_size = height * width;
    auto channels_per_group = num_channels / num_groups;
    
    auto output = torch::zeros_like(input);
    auto group_mean = torch::zeros({batch_size, num_groups}, input.options());
    auto group_var = torch::zeros({batch_size, num_groups}, input.options());
    
    dim3 stats_grid(num_groups, batch_size);
    dim3 stats_block(256);
    
    compute_group_stats<<<stats_grid, stats_block>>>(
        input.data_ptr<float>(),
        group_mean.data_ptr<float>(),
        group_var.data_ptr<float>(),
        batch_size,
        num_channels,
        num_groups,
        spatial_size,
        channels_per_group
    );
    
    int total_elements = batch_size * num_channels * spatial_size;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    apply_groupnorm_with_gelu<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        group_mean.data_ptr<float>(),
        group_var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size,
        num_channels,
        num_groups,
        spatial_size,
        channels_per_group,
        eps
    );
    
    return output;
}
"""

fused_gelu_groupnorm_cpp_source = """
torch::Tensor fused_gelu_groupnorm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
);
"""

fused_gelu_groupnorm = load_inline(
    name="fused_gelu_groupnorm",
    cpp_sources=fused_gelu_groupnorm_cpp_source,
    cuda_sources=fused_gelu_groupnorm_source,
    functions=["fused_gelu_groupnorm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused GELU + GroupNorm kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.eps = 1e-5
        self.fused_gelu_groupnorm = fused_gelu_groupnorm

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_gelu_groupnorm.fused_gelu_groupnorm_cuda(
            x, self.weight, self.bias, self.num_groups, self.eps
        )
        return x

batch_size   = 128  
in_channels  = 64  
out_channels = 64  
height = width = 256  
kernel_size  = 3
stride       = 1
groups = 8
num_groups = 8

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]