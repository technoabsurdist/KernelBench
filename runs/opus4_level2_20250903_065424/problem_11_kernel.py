import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused BatchNorm + Tanh
fused_bn_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float tanh_impl(float x) {
    float exp2x = expf(2.0f * x);
    return (exp2x - 1.0f) / (exp2x + 1.0f);
}

__global__ void fused_bn_tanh_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int channels,
    int spatial_size,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * spatial_size;
    
    if (idx < total_elements) {
        int c = (idx / spatial_size) % channels;
        
        float mean = running_mean[c];
        float var = running_var[c];
        float scale = gamma[c];
        float bias = beta[c];
        
        float x = input[idx];
        float x_normalized = (x - mean) / sqrtf(var + eps);
        float x_scaled = x_normalized * scale + bias;
        output[idx] = tanh_impl(x_scaled);
    }
}

torch::Tensor fused_bn_tanh_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto spatial_size = height * width;
    
    auto output = torch::zeros_like(input);
    
    int total_elements = batch_size * channels * spatial_size;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_bn_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        spatial_size,
        eps
    );
    
    return output;
}
"""

fused_bn_tanh_cpp_source = "torch::Tensor fused_bn_tanh_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, torch::Tensor running_mean, torch::Tensor running_var, float eps);"

# Compile the inline CUDA code
fused_bn_tanh = load_inline(
    name="fused_bn_tanh",
    cpp_sources=fused_bn_tanh_cpp_source,
    cuda_sources=fused_bn_tanh_source,
    functions=["fused_bn_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with fused BatchNorm + Tanh kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.fused_bn_tanh = fused_bn_tanh
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv_transpose(x)
        
        # Use fused BatchNorm + Tanh kernel
        with torch.no_grad():
            x = self.fused_bn_tanh.fused_bn_tanh_cuda(
                x,
                self.batch_norm.weight,
                self.batch_norm.bias,
                self.batch_norm.running_mean,
                self.batch_norm.running_var,
                self.eps
            )
        
        x = self.max_pool(x)
        x = self.group_norm(x)
        return x


batch_size = 512
in_channels  = 64  
out_channels = 128  
height = width = 2048  
kernel_size  = 5
stride       = 1  
padding      = 1
groups       = 8
num_groups   = 8
height, width = 32, 32

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, num_groups]