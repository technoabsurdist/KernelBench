import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused instance normalization with division
fused_instance_norm_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void fused_instance_norm_div_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    int batch_size,
    int num_channels,
    int spatial_size,
    float eps,
    float divide_by) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_channels * spatial_size;
    
    if (idx >= total_elements) return;
    
    int b = idx / (num_channels * spatial_size);
    int c = (idx / spatial_size) % num_channels;
    int s = idx % spatial_size;
    
    // Compute mean and variance for this channel in this sample
    scalar_t mean = 0;
    scalar_t var = 0;
    
    // First pass: compute mean
    for (int i = 0; i < spatial_size; ++i) {
        mean += input[b * num_channels * spatial_size + c * spatial_size + i];
    }
    mean /= spatial_size;
    
    // Second pass: compute variance
    for (int i = 0; i < spatial_size; ++i) {
        scalar_t diff = input[b * num_channels * spatial_size + c * spatial_size + i] - mean;
        var += diff * diff;
    }
    var /= spatial_size;
    
    // Normalize and apply affine transformation
    scalar_t std_inv = rsqrtf(var + eps);
    scalar_t normalized = (input[idx] - mean) * std_inv;
    
    // Apply affine parameters if provided, otherwise use defaults (gamma=1, beta=0)
    scalar_t gamma_val = (gamma != nullptr) ? gamma[c] : 1.0;
    scalar_t beta_val = (beta != nullptr) ? beta[c] : 0.0;
    
    // Apply affine transformation and division
    output[idx] = (normalized * gamma_val + beta_val) / divide_by;
}

torch::Tensor fused_instance_norm_div_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps,
    float divide_by) {
    
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto spatial_size = height * width;
    
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (batch_size * num_channels * spatial_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_instance_norm_div_cuda", ([&] {
        fused_instance_norm_div_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            gamma.defined() ? gamma.data_ptr<scalar_t>() : nullptr,
            beta.defined() ? beta.data_ptr<scalar_t>() : nullptr,
            batch_size,
            num_channels,
            spatial_size,
            eps,
            divide_by
        );
    }));
    
    return output;
}
"""

fused_instance_norm_div_cpp_source = """
torch::Tensor fused_instance_norm_div_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps,
    float divide_by);
"""

# Compile the inline CUDA code
fused_instance_norm_div = load_inline(
    name="fused_instance_norm_div",
    cpp_sources=fused_instance_norm_div_cpp_source,
    cuda_sources=fused_instance_norm_div_source,
    functions=["fused_instance_norm_div_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused instance normalization and division kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.out_channels = out_channels
        self.divide_by = divide_by
        self.eps = 1e-5
        
        # Instance norm parameters
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        self.fused_instance_norm_div = fused_instance_norm_div

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_instance_norm_div.fused_instance_norm_div_cuda(
            x, self.weight, self.bias, self.eps, self.divide_by
        )
        return x

batch_size = 128
in_channels  = 64  
out_channels = 128  
height = width = 128  
kernel_size = 3
divide_by = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]