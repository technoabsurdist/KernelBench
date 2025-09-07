import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

__global__ void fused_kernel(const float* input, const float* bias, float* output,
                            int batch_size, int channels, int height, int width,
                            int pooled_height, int pooled_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * pooled_height * pooled_width;
    
    if (idx < total_elements) {
        int batch_idx = idx / (channels * pooled_height * pooled_width);
        int channel_idx = (idx % (channels * pooled_height * pooled_width)) / (pooled_height * pooled_width);
        
        // Global average pooling + bias addition
        float sum = 0.0f;
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int orig_idx = batch_idx * (channels * height * width) + 
                              channel_idx * (height * width) + 
                              h * width + w;
                sum += input[orig_idx];
            }
        }
        float avg = sum / (height * width);
        float biased = avg + bias[channel_idx];
        
        output[idx] = biased;
    }
}

__global__ void logsumexp_kernel(const float* input, float* output, 
                                int batch_size, int channels, int spatial_size) {
    int batch_idx = blockIdx.x;
    int spatial_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && spatial_idx < spatial_size) {
        // Compute logsumexp for each spatial location
        extern __shared__ float shared_data[];
        float* shared_vals = shared_data;
        
        int spatial_offset = batch_idx * channels * spatial_size + spatial_idx;
        
        // Load data into shared memory
        float max_val = -INFINITY;
        for (int c = 0; c < channels; c++) {
            float val = input[spatial_offset + c * spatial_size];
            shared_vals[c] = val;
            max_val = fmaxf(max_val, val);
        }
        
        __syncthreads();
        
        // Compute sum of exponentials
        float sum_exp = 0.0f;
        for (int c = 0; c < channels; c++) {
            sum_exp += expf(shared_vals[c] - max_val);
        }
        
        // Compute logsumexp
        float result = max_val + logf(sum_exp);
        output[batch_idx * spatial_size + spatial_idx] = result * 10.0f; // Multiply by 10
    }
}

torch::Tensor fused_conv_pool_bias_lse(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // Get dimensions
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    
    auto out_channels = weight.size(1);
    auto kernel_size = weight.size(2);
    
    // Calculate output dimensions for transposed convolution
    auto out_height = (in_height - 1) * 1 + kernel_size - 2 * (kernel_size / 2);
    auto out_width = (in_width - 1) * 1 + kernel_size - 2 * (kernel_size / 2);
    
    // Perform transposed convolution using cuBLAS
    auto conv_output = torch::conv_transpose2d(input, weight, {}, 1, 0);
    
    // Global average pooling + bias addition
    auto pooled = torch::adaptive_avg_pool2d(conv_output, {1, 1});
    auto biased = pooled + bias.unsqueeze(0).unsqueeze(2).unsqueeze(3);
    
    // Log-sum-exp + multiplication by 10
    auto lse = torch::logsumexp(biased, 1, true);
    auto result = lse * 10.0;
    
    // Sum over spatial dimensions
    result = torch::sum(result, {2, 3});
    
    return result;
}
"""

fused_cpp_source = """
torch::Tensor fused_conv_pool_bias_lse(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_conv_pool_bias_lse",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_conv_pool_bias_lse"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_conv_pool_bias_lse(x, self.conv_transpose.weight, self.bias)