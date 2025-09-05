import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + swish + bias
fused_matmul_swish_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_matmul_swish_bias_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx < total_elements) {
        int b = idx / out_features;
        int o = idx % out_features;
        
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += input[b * in_features + i] * weight[o * in_features + i];
        }
        
        float sigmoid_val = 1.0f / (1.0f + expf(-sum));
        float swish_val = sigmoid_val * sum;
        output[idx] = swish_val + bias[o];
    }
}

torch::Tensor fused_matmul_swish_bias_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    int total_elements = batch_size * out_features;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_matmul_swish_bias_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

fused_matmul_swish_bias_cpp_source = (
    "torch::Tensor fused_matmul_swish_bias_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Define the custom CUDA kernel for GroupNorm
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void group_norm_kernel(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int batch_size,
    int num_channels,
    int num_groups,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_channels;
    
    if (idx < total_elements) {
        int b = idx / num_channels;
        int c = idx % num_channels;
        int group_id = c / (num_channels / num_groups);
        int channels_per_group = num_channels / num_groups;
        
        // Calculate mean and variance for the group
        float mean = 0.0f;
        float variance = 0.0f;
        int group_start = group_id * channels_per_group;
        int group_end = group_start + channels_per_group;
        
        for (int ch = group_start; ch < group_end; ch++) {
            mean += input[b * num_channels + ch];
        }
        mean /= channels_per_group;
        
        for (int ch = group_start; ch < group_end; ch++) {
            float diff = input[b * num_channels + ch] - mean;
            variance += diff * diff;
        }
        variance /= channels_per_group;
        
        // Normalize
        float std_inv = rsqrtf(variance + eps);
        float normalized = (input[idx] - mean) * std_inv;
        
        // Apply affine transformation
        output[idx] = normalized * gamma[c] + beta[c];
    }
}

torch::Tensor group_norm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps
) {
    int batch_size = input.size(0);
    int num_channels = input.size(1);
    
    auto output = torch::zeros_like(input);
    
    int total_elements = batch_size * num_channels;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    group_norm_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        batch_size,
        num_channels,
        num_groups,
        eps
    );
    
    return output;
}
"""

group_norm_cpp_source = (
    "torch::Tensor group_norm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, int num_groups, float eps);"
)

# Compile the inline CUDA code
fused_matmul_swish_bias = load_inline(
    name="fused_matmul_swish_bias",
    cpp_sources=fused_matmul_swish_bias_cpp_source,
    cuda_sources=fused_matmul_swish_bias_source,
    functions=["fused_matmul_swish_bias_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

group_norm = load_inline(
    name="group_norm",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.num_groups = num_groups
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        self.eps = 1e-5
        self.fused_matmul_swish_bias = fused_matmul_swish_bias
        self.group_norm = group_norm

    def forward(self, x):
        x = self.fused_matmul_swish_bias.fused_matmul_swish_bias_cuda(
            x.contiguous(), self.weight.contiguous(), self.bias.contiguous()
        )
        x = self.group_norm.group_norm_cuda(
            x.contiguous(), self.gamma.contiguous(), self.beta.contiguous(), 
            self.num_groups, self.eps
        )
        return x


batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]