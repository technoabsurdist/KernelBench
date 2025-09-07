import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv2d + gelu + adaptive_avg_pool2d
fused_conv_gelu_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void gelu_kernel(const float* input, float* output, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    }
}

__global__ void adaptive_avg_pool_kernel(const float* input, float* output, 
                                        int batch_size, int channels, int input_height, int input_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels;
    
    if (idx < total_outputs) {
        int batch_idx = idx / channels;
        int channel_idx = idx % channels;
        
        const float* input_ptr = input + (batch_idx * channels + channel_idx) * input_height * input_width;
        float sum = 0.0f;
        
        for (int i = 0; i < input_height * input_width; ++i) {
            sum += input_ptr[i];
        }
        
        output[idx] = sum / (input_height * input_width);
    }
}

torch::Tensor fused_conv_gelu_pool2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // Conv2d
    auto conv_output = torch::conv2d(input, weight, bias, 1, 0, 1, 1);
    
    // GELU activation
    auto gelu_output = torch::zeros_like(conv_output);
    int64_t total_elements = conv_output.numel();
    
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    gelu_kernel<<<num_blocks, block_size>>>(
        conv_output.data_ptr<float>(), 
        gelu_output.data_ptr<float>(), 
        total_elements
    );
    
    // Adaptive average pooling to (1, 1)
    auto batch_size = gelu_output.size(0);
    auto channels = gelu_output.size(1);
    auto height = gelu_output.size(2);
    auto width = gelu_output.size(3);
    
    auto pool_output = torch::zeros({batch_size, channels}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int pool_block_size = 256;
    const int pool_num_blocks = (batch_size * channels + pool_block_size - 1) / pool_block_size;
    
    adaptive_avg_pool_kernel<<<pool_num_blocks, pool_block_size>>>(
        gelu_output.data_ptr<float>(),
        pool_output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width
    );
    
    return pool_output;
}
"""

fused_conv_gelu_pool_cpp_source = """
torch::Tensor fused_conv_gelu_pool2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code for fused operations
fused_conv_gelu_pool = load_inline(
    name="fused_conv_gelu_pool",
    cpp_sources=fused_conv_gelu_pool_cpp_source,
    cuda_sources=fused_conv_gelu_pool_source,
    functions=["fused_conv_gelu_pool2d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused conv2d + gelu + adaptive_avg_pool2d operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_conv_gelu_pool = fused_conv_gelu_pool

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels)
        """
        return self.fused_conv_gelu_pool.fused_conv_gelu_pool2d_cuda(x, self.conv.weight, self.conv.bias)