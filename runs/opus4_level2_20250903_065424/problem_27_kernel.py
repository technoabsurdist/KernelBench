import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused HardSwish + GroupNorm + Mean pooling
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ float hardswish(float x) {
    if (x <= -3.0f) return 0.0f;
    if (x >= 3.0f) return x;
    return x * (x + 3.0f) / 6.0f;
}

__global__ void fused_hardswish_groupnorm_mean_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int channels,
    int spatial_size,
    int num_groups,
    int channels_per_group,
    float eps)
{
    extern __shared__ float shared_mem[];
    
    int batch_idx = blockIdx.y;
    int group_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || group_idx >= num_groups) return;
    
    float* group_mean = shared_mem;
    float* group_var = shared_mem + blockDim.x;
    float* group_sum = shared_mem + 2 * blockDim.x;
    
    int group_start_channel = group_idx * channels_per_group;
    int group_end_channel = min(group_start_channel + channels_per_group, channels);
    int group_elements = (group_end_channel - group_start_channel) * spatial_size;
    
    // Phase 1: Compute mean with HardSwish applied
    float local_sum = 0.0f;
    int elements_per_thread = (group_elements + blockDim.x - 1) / blockDim.x;
    
    for (int i = 0; i < elements_per_thread; i++) {
        int elem_idx = tid + i * blockDim.x;
        if (elem_idx < group_elements) {
            int channel_offset = elem_idx / spatial_size;
            int spatial_offset = elem_idx % spatial_size;
            int channel_idx = group_start_channel + channel_offset;
            int global_idx = batch_idx * channels * spatial_size + 
                           channel_idx * spatial_size + spatial_offset;
            float val = hardswish(input[global_idx]);
            local_sum += val;
        }
    }
    
    group_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduce to get group mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            group_sum[tid] += group_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = group_sum[0] / group_elements;
    if (tid == 0) group_mean[0] = mean;
    __syncthreads();
    mean = group_mean[0];
    
    // Phase 2: Compute variance
    float local_var = 0.0f;
    for (int i = 0; i < elements_per_thread; i++) {
        int elem_idx = tid + i * blockDim.x;
        if (elem_idx < group_elements) {
            int channel_offset = elem_idx / spatial_size;
            int spatial_offset = elem_idx % spatial_size;
            int channel_idx = group_start_channel + channel_offset;
            int global_idx = batch_idx * channels * spatial_size + 
                           channel_idx * spatial_size + spatial_offset;
            float val = hardswish(input[global_idx]);
            float diff = val - mean;
            local_var += diff * diff;
        }
    }
    
    group_sum[tid] = local_var;
    __syncthreads();
    
    // Reduce to get variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            group_sum[tid] += group_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float var = group_sum[0] / group_elements;
    float std = sqrtf(var + eps);
    if (tid == 0) group_var[0] = std;
    __syncthreads();
    std = group_var[0];
    
    // Phase 3: Normalize and compute spatial mean per channel
    for (int c = group_start_channel; c < group_end_channel; c++) {
        float channel_sum = 0.0f;
        
        for (int s = tid; s < spatial_size; s += blockDim.x) {
            int global_idx = batch_idx * channels * spatial_size + c * spatial_size + s;
            float val = hardswish(input[global_idx]);
            float normalized = (val - mean) / std;
            if (weight != nullptr && bias != nullptr) {
                normalized = normalized * weight[c] + bias[c];
            }
            channel_sum += normalized;
        }
        
        // Reduce channel sum
        group_sum[tid] = channel_sum;
        __syncthreads();
        
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                group_sum[tid] += group_sum[tid + stride];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            output[batch_idx * channels + c] = group_sum[0] / spatial_size;
        }
        __syncthreads();
    }
}

torch::Tensor fused_hardswish_groupnorm_mean_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps)
{
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto spatial_size = depth * height * width;
    
    auto output = torch::zeros({batch_size, channels}, input.options());
    
    int channels_per_group = (channels + num_groups - 1) / num_groups;
    
    const int threads = 256;
    dim3 blocks(num_groups, batch_size);
    size_t shared_mem_size = 3 * threads * sizeof(float);
    
    fused_hardswish_groupnorm_mean_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.defined() ? weight.data_ptr<float>() : nullptr,
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        channels,
        spatial_size,
        num_groups,
        channels_per_group,
        eps
    );
    
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_hardswish_groupnorm_mean_cuda(
    torch::Tensor input,
    torch::Tensor weight, 
    torch::Tensor bias,
    int num_groups,
    float eps);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_hardswish_groupnorm_mean_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.num_groups = num_groups
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_hardswish_groupnorm_mean_cuda(
            x, self.weight, self.bias, self.num_groups, self.eps
        )
        return x

# === Test config ===
batch_size = 1024
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 4

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]