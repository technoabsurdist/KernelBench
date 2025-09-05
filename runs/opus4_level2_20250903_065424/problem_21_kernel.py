import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused kernel for bias add + scale + sigmoid
fused_bias_scale_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bias_scale_sigmoid_kernel(
    float* data, 
    const float* bias, 
    const float* scale,
    int batch_size,
    int channels,
    int spatial_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * spatial_size;
    
    if (idx < total_size) {
        int c = (idx / spatial_size) % channels;
        float val = data[idx];
        val = val + bias[c];
        val = val * scale[c];
        val = 1.0f / (1.0f + expf(-val));
        data[idx] = val;
    }
}

torch::Tensor fused_bias_scale_sigmoid_cuda(
    torch::Tensor input, 
    torch::Tensor bias, 
    torch::Tensor scale) {
    
    auto output = input.clone();
    int batch_size = output.size(0);
    int channels = output.size(1);
    int spatial_size = output.size(2) * output.size(3);
    int total_size = batch_size * channels * spatial_size;
    
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    fused_bias_scale_sigmoid_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        batch_size,
        channels,
        spatial_size
    );
    
    return output;
}
"""

fused_bias_scale_sigmoid_cpp_source = (
    "torch::Tensor fused_bias_scale_sigmoid_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor scale);"
)

# Custom group normalization kernel
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void group_norm_kernel(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    float* mean,
    float* var,
    int batch_size,
    int num_groups,
    int channels_per_group,
    int spatial_size,
    float eps) {
    
    int group_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    
    if (group_idx >= num_groups || batch_idx >= batch_size) return;
    
    int channels = num_groups * channels_per_group;
    int group_size = channels_per_group * spatial_size;
    
    // Calculate mean
    float sum = 0.0f;
    for (int c = 0; c < channels_per_group; c++) {
        int channel_idx = group_idx * channels_per_group + c;
        for (int s = 0; s < spatial_size; s++) {
            int idx = batch_idx * channels * spatial_size + channel_idx * spatial_size + s;
            sum += input[idx];
        }
    }
    float group_mean = sum / group_size;
    
    // Calculate variance
    float var_sum = 0.0f;
    for (int c = 0; c < channels_per_group; c++) {
        int channel_idx = group_idx * channels_per_group + c;
        for (int s = 0; s < spatial_size; s++) {
            int idx = batch_idx * channels * spatial_size + channel_idx * spatial_size + s;
            float diff = input[idx] - group_mean;
            var_sum += diff * diff;
        }
    }
    float group_var = var_sum / group_size;
    
    // Normalize and apply affine transform
    float inv_std = rsqrtf(group_var + eps);
    for (int c = 0; c < channels_per_group; c++) {
        int channel_idx = group_idx * channels_per_group + c;
        for (int s = 0; s < spatial_size; s++) {
            int idx = batch_idx * channels * spatial_size + channel_idx * spatial_size + s;
            float normalized = (input[idx] - group_mean) * inv_std;
            output[idx] = normalized * gamma[channel_idx] + beta[channel_idx];
        }
    }
}

torch::Tensor group_norm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps) {
    
    auto output = torch::zeros_like(input);
    int batch_size = input.size(0);
    int channels = input.size(1);
    int spatial_size = input.size(2) * input.size(3);
    int channels_per_group = channels / num_groups;
    
    auto mean = torch::zeros({batch_size, num_groups}, input.options());
    auto var = torch::zeros({batch_size, num_groups}, input.options());
    
    dim3 blocks(num_groups, batch_size);
    dim3 threads(1);
    
    group_norm_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        batch_size,
        num_groups,
        channels_per_group,
        spatial_size,
        eps
    );
    
    return output;
}
"""

group_norm_cpp_source = (
    "torch::Tensor group_norm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, int num_groups, float eps);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_bias_scale_sigmoid_cpp_source,
    cuda_sources=fused_bias_scale_sigmoid_source,
    functions=["fused_bias_scale_sigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

group_norm_op = load_inline(
    name="group_norm_op",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape).squeeze(-1).squeeze(-1))
        self.scale = nn.Parameter(torch.randn(scale_shape).squeeze(-1).squeeze(-1))
        self.num_groups = num_groups
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        self.fused_ops = fused_ops
        self.group_norm_op = group_norm_op

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_bias_scale_sigmoid_cuda(x, self.bias, self.scale)
        x = self.group_norm_op.group_norm_cuda(x, self.gamma, self.beta, self.num_groups, 1e-5)
        return x

batch_size = 128
in_channels = 8
out_channels = 32
height = width = 256
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]