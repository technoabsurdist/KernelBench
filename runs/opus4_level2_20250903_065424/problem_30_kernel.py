import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GroupNorm + HardTanh
fused_groupnorm_hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void fused_groupnorm_hardtanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const float eps,
    const float min_val,
    const float max_val)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = batch_size * num_channels;
    
    if (tid >= total_threads) return;
    
    const int b = tid / num_channels;
    const int c = tid % num_channels;
    
    const int channels_per_group = num_channels / num_groups;
    const int g = c / channels_per_group;
    
    // Calculate mean and variance for this group
    float mean = 0.0f;
    float var = 0.0f;
    
    const int group_start = g * channels_per_group;
    const int group_end = group_start + channels_per_group;
    
    // First pass: calculate mean
    for (int ch = group_start; ch < group_end; ++ch) {
        mean += input[b * num_channels + ch];
    }
    mean /= channels_per_group;
    
    // Second pass: calculate variance
    for (int ch = group_start; ch < group_end; ++ch) {
        float diff = input[b * num_channels + ch] - mean;
        var += diff * diff;
    }
    var /= channels_per_group;
    
    // Apply GroupNorm
    float std_inv = rsqrtf(var + eps);
    float normalized = (input[tid] - mean) * std_inv;
    float scaled = normalized * weight[c] + bias[c];
    
    // Apply HardTanh
    output[tid] = fminf(fmaxf(scaled, min_val), max_val);
}

torch::Tensor fused_groupnorm_hardtanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps,
    float min_val,
    float max_val)
{
    const int batch_size = input.size(0);
    const int num_channels = input.size(1);
    
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (batch_size * num_channels + block_size - 1) / block_size;
    
    fused_groupnorm_hardtanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_channels,
        num_groups,
        eps,
        min_val,
        max_val);
    
    return output;
}
"""

fused_groupnorm_hardtanh_cpp_source = """
torch::Tensor fused_groupnorm_hardtanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps,
    float min_val,
    float max_val);
"""

# Compile the inline CUDA code
fused_groupnorm_hardtanh = load_inline(
    name="fused_groupnorm_hardtanh",
    cpp_sources=fused_groupnorm_hardtanh_cpp_source,
    cuda_sources=fused_groupnorm_hardtanh_source,
    functions=["fused_groupnorm_hardtanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused GroupNorm + HardTanh kernel
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        
        # Initialize GroupNorm parameters
        self.num_groups = num_groups
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        
        self.fused_op = fused_groupnorm_hardtanh

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.fused_op.fused_groupnorm_hardtanh_cuda(
            x, self.weight, self.bias, self.num_groups, 
            self.eps, self.hardtanh_min, self.hardtanh_max
        )
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 16
hardtanh_min = -2.0
hardtanh_max = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, hardtanh_min, hardtanh_max]