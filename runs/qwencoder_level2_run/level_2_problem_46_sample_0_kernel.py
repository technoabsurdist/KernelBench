import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv-sub-tanh-sub-avgpool operation
fused_conv_activation_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_conv_sub_tanh_sub_avgpool_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_size,
    int pool_size,
    float subtract1_value,
    float subtract2_value
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    // Decode output position
    int temp = idx;
    int w_out = temp % out_width;
    temp /= out_width;
    int h_out = temp % out_height;
    temp /= out_height;
    int c_out = temp % out_channels;
    int n = temp / out_channels;
    
    // Calculate input region for convolution
    int kernel_radius = kernel_size / 2;
    int h_in_start = h_out * pool_size - kernel_radius;
    int w_in_start = w_out * pool_size - kernel_radius;
    
    float sum = 0.0f;
    int pool_count = 0;
    
    // Perform pooling and convolution together
    for (int ph = 0; ph < pool_size && h_out * pool_size + ph < in_height + 2 * kernel_radius; ph++) {
        for (int pw = 0; pw < pool_size && w_out * pool_size + pw < in_width + 2 * kernel_radius; pw++) {
            int h_in = h_out * pool_size + ph - kernel_radius;
            int w_in = w_out * pool_size + pw - kernel_radius;
            
            if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                float conv_sum = bias[c_out];
                
                // Perform convolution
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int ih = h_in + kernel_radius - kh;
                        int iw = w_in + kernel_radius - kw;
                        
                        if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                            for (int c_in = 0; c_in < in_channels; c_in++) {
                                int input_idx = n * in_channels * in_height * in_width + 
                                               c_in * in_height * in_width + 
                                               ih * in_width + iw;
                                int weight_idx = c_out * in_channels * kernel_size * kernel_size + 
                                                c_in * kernel_size * kernel_size + 
                                                kh * kernel_size + kw;
                                conv_sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
                
                // Apply sub-tanh-sub operations
                conv_sum -= subtract1_value;
                conv_sum = tanhf(conv_sum);
                conv_sum -= subtract2_value;
                
                sum += conv_sum;
                pool_count++;
            }
        }
    }
    
    // Average pooling
    if (pool_count > 0) {
        output[idx] = sum / pool_count;
    } else {
        output[idx] = 0.0f;
    }
}

torch::Tensor fused_conv_sub_tanh_sub_avgpool_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int pool_size,
    float subtract1_value,
    float subtract2_value
) {
    // Get tensor dimensions
    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int in_height = input_sizes[2];
    int in_width = input_sizes[3];
    
    int out_channels = weight.sizes()[0];
    int out_height = (in_height - pool_size) / pool_size + 1;
    int out_width = (in_width - pool_size) / pool_size + 1;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Calculate grid and block dimensions
    int total_elements = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    // Launch kernel
    fused_conv_sub_tanh_sub_avgpool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel_size,
        pool_size,
        subtract1_value,
        subtract2_value
    );
    
    return output;
}
"""

fused_conv_activation_pool_cpp_source = """
torch::Tensor fused_conv_sub_tanh_sub_avgpool_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int pool_size,
    float subtract1_value,
    float subtract2_value
);
"""

# Compile the inline CUDA code for fused operation
fused_conv_activation_pool = load_inline(
    name="fused_conv_activation_pool",
    cpp_sources=fused_conv_activation_pool_cpp_source,
    cuda_sources=fused_conv_activation_pool_source,
    functions=["fused_conv_sub_tanh_sub_avgpool_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for convolution, subtraction, tanh activation, subtraction and average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool
        
        # Convolution parameters
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.randn(out_channels))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.conv_weight)
        nn.init.zeros_(self.conv_bias)

    def forward(self, x):
        return fused_conv_activation_pool.fused_conv_sub_tanh_sub_avgpool_cuda(
            x,
            self.conv_weight,
            self.conv_bias,
            self.kernel_size,
            self.kernel_size_pool,
            self.subtract1_value,
            self.subtract2_value
        )