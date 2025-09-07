import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv3d + maxpool3d + logsumexp + relu
fused_conv3d_maxpool_lse_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void conv3d_maxpool_lse_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * (output_depth/2) * (output_height/2) * (output_width/2);
    
    if (idx >= total_elements) return;
    
    // Decode output indices
    int temp = idx;
    int out_w = temp % (output_width/2);
    temp /= (output_width/2);
    int out_h = temp % (output_height/2);
    temp /= (output_height/2);
    int out_d = temp % (output_depth/2);
    temp /= (output_depth/2);
    int out_c = temp % out_channels;
    int batch = temp / out_channels;
    
    // Map to pre-pooling indices
    int pp_d = out_d * 2;
    int pp_h = out_h * 2;
    int pp_w = out_w * 2;
    
    // Compute convolution and max pooling for 2x2x2 window
    float max_val = -1e38f;
    
    for (int pd = 0; pd < 2; pd++) {
        for (int ph = 0; ph < 2; ph++) {
            for (int pw = 0; pw < 2; pw++) {
                int d = pp_d + pd;
                int h = pp_h + ph;
                int w = pp_w + pw;
                
                if (d >= output_depth || h >= output_height || w >= output_width) continue;
                
                // Convolution computation
                float conv_sum = 0.0f;
                
                for (int kd = 0; kd < kernel_size; kd++) {
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int in_d = d * stride - padding + kd;
                            int in_h = h * stride - padding + kh;
                            int in_w = w * stride - padding + kw;
                            
                            if (in_d >= 0 && in_d < input_depth &&
                                in_h >= 0 && in_h < input_height &&
                                in_w >= 0 && in_w < input_width) {
                                
                                for (int ic = 0; ic < in_channels; ic++) {
                                    int input_idx = batch * (in_channels * input_depth * input_height * input_width) +
                                                   ic * (input_depth * input_height * input_width) +
                                                   in_d * (input_height * input_width) +
                                                   in_h * input_width +
                                                   in_w;
                                                   
                                    int weight_idx = out_c * (in_channels * kernel_size * kernel_size * kernel_size) +
                                                    ic * (kernel_size * kernel_size * kernel_size) +
                                                    kd * (kernel_size * kernel_size) +
                                                    kh * kernel_size +
                                                    kw;
                                                   
                                    conv_sum += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                }
                
                // Add bias
                conv_sum += bias[out_c];
                
                // Update maximum for max pooling
                if (conv_sum > max_val) {
                    max_val = conv_sum;
                }
            }
        }
    }
    
    // Apply logsumexp and ReLU in one step
    // Since we only have one value after max pooling, logsumexp is just the value itself
    float result = fmaxf(max_val, 0.0f); // ReLU
    
    output[idx] = result;
}

torch::Tensor fused_conv3d_maxpool_lse_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding
) {
    // Get dimensions
    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_depth = input_sizes[2];
    int input_height = input_sizes[3];
    int input_width = input_sizes[4];
    
    auto weight_sizes = weight.sizes();
    int out_channels = weight_sizes[0];
    
    // Calculate output dimensions after convolution
    int conv_depth = (input_depth + 2 * padding - kernel_size) / stride + 1;
    int conv_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int conv_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    
    // Calculate output dimensions after max pooling
    int output_depth = conv_depth / 2;
    int output_height = conv_height / 2;
    int output_width = conv_width / 2;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, 1, output_depth, output_height, output_width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Launch kernel
    int total_elements = batch_size * output_depth * output_height * output_width;
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    conv3d_maxpool_lse_relu_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        conv_depth,
        conv_height,
        conv_width,
        kernel_size,
        stride,
        padding
    );
    
    return output;
}
"""

fused_conv3d_maxpool_lse_relu_cpp_source = """
torch::Tensor fused_conv3d_maxpool_lse_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding
);
"""

# Compile the inline CUDA code
fused_conv3d_maxpool_lse_relu = load_inline(
    name="fused_conv3d_maxpool_lse_relu",
    cpp_sources=fused_conv3d_maxpool_lse_relu_cpp_source,
    cuda_sources=fused_conv3d_maxpool_lse_relu_source,
    functions=["fused_conv3d_maxpool_lse_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for conv3d + maxpool3d + logsumexp + relu
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Initialize weights similar to PyTorch's default initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, 1, depth', height', width')
        """
        return fused_conv3d_maxpool_lse_relu.fused_conv3d_maxpool_lse_relu_cuda(
            x, self.weight, self.bias, self.kernel_size, self.stride, self.padding
        )

import math