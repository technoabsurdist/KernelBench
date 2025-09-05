import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Swish activation
swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sig;
    }
}

torch::Tensor swish_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    swish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size
    );
    
    return output;
}
"""

swish_cpp_source = "torch::Tensor swish_cuda(torch::Tensor input);"

# Custom CUDA kernel for GroupNorm + HardSwish fusion
groupnorm_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void groupnorm_hardswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const int spatial_size,
    const float eps) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    
    const int channels_per_group = num_channels / num_groups;
    const int group_size = channels_per_group * spatial_size;
    
    for (int idx = tid; idx < batch_size * num_channels * spatial_size; idx += total_threads) {
        const int b = idx / (num_channels * spatial_size);
        const int c = (idx / spatial_size) % num_channels;
        const int s = idx % spatial_size;
        
        const int g = c / channels_per_group;
        const int group_start = b * num_groups * group_size + g * group_size;
        
        // Compute mean and variance for the group
        float mean = 0.0f;
        float m2 = 0.0f;
        
        for (int i = 0; i < group_size; ++i) {
            int offset = b * num_channels * spatial_size + (g * channels_per_group + i / spatial_size) * spatial_size + (i % spatial_size);
            float val = input[offset];
            mean += val;
            m2 += val * val;
        }
        
        mean /= group_size;
        float variance = m2 / group_size - mean * mean;
        float std = sqrtf(variance + eps);
        
        // Apply group normalization
        float normalized = (input[idx] - mean) / std;
        float scaled = normalized * weight[c] + bias[c];
        
        // Apply HardSwish: x * (x + 3).clamp(0, 6) / 6
        float hs_input = scaled + 3.0f;
        hs_input = fmaxf(0.0f, fminf(6.0f, hs_input));
        output[idx] = scaled * hs_input / 6.0f;
    }
}

torch::Tensor groupnorm_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps) {
    
    const int batch_size = input.size(0);
    const int num_channels = input.size(1);
    const int depth = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);
    const int spatial_size = depth * height * width;
    
    auto output = torch::empty_like(input);
    
    const int block_size = 128;
    const int num_blocks = (batch_size * num_channels * spatial_size + block_size - 1) / block_size;
    
    groupnorm_hardswish_kernel<<<num_blocks, block_size>>>(
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

groupnorm_hardswish_cpp_source = """
torch::Tensor groupnorm_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps);
"""

# Compile the inline CUDA codes
swish_module = load_inline(
    name="swish",
    cpp_sources=swish_cpp_source,
    cuda_sources=swish_source,
    functions=["swish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

groupnorm_hardswish_module = load_inline(
    name="groupnorm_hardswish",
    cpp_sources=groupnorm_hardswish_cpp_source,
    cuda_sources=groupnorm_hardswish_source,
    functions=["groupnorm_hardswish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.groups = groups
        self.eps = eps
        self.swish_module = swish_module
        self.groupnorm_hardswish_module = groupnorm_hardswish_module

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.swish_module.swish_cuda(x.contiguous())
        x = self.groupnorm_hardswish_module.groupnorm_hardswish_cuda(
            x.contiguous(), 
            self.group_norm.weight, 
            self.group_norm.bias,
            self.groups,
            self.eps
        )
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
groups = 4
eps = 1e-5

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, eps]