import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv+tanh+scale+bias+pool operation
fused_conv_tanh_scale_bias_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

__global__ void fused_conv_tanh_scale_bias_pool_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    float scaling_factor,
    int pool_kernel_size,
    int pooled_height,
    int pooled_width
) {
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y;
    int pooled_h = threadIdx.y + blockIdx.z * blockDim.y;
    int pooled_w = threadIdx.x + blockIdx.z * blockDim.x;
    
    if (batch_idx >= batch_size || out_ch >= out_channels || 
        pooled_h >= pooled_height || pooled_w >= pooled_width) return;
        
    int pool_start_h = pooled_h * pool_kernel_size;
    int pool_start_w = pooled_w * pool_kernel_size;
    
    float max_val = -1e30f;
    
    for (int ph = 0; ph < pool_kernel_size && (pool_start_h + ph) < output_height; ++ph) {
        for (int pw = 0; pw < pool_kernel_size && (pool_start_w + pw) < output_width; ++pw) {
            int out_h = pool_start_h + ph;
            int out_w = pool_start_w + pw;
            
            float sum = 0.0f;
            for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int in_h = out_h + kh - (kernel_size/2);
                        int in_w = out_w + kw - (kernel_size/2);
                        
                        if (in_h >= 0 && in_h < input_height && in_w >= 0 && in_w < input_width) {
                            int input_idx = batch_idx * (in_channels * input_height * input_width) +
                                          in_ch * (input_height * input_width) +
                                          in_h * input_width + in_w;
                                          
                            int weight_idx = out_ch * (in_channels * kernel_size * kernel_size) +
                                           in_ch * (kernel_size * kernel_size) +
                                           kh * kernel_size + kw;
                                           
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            
            float activated = tanhf(sum) * scaling_factor + bias[out_ch];
            max_val = fmaxf(max_val, activated);
        }
    }
    
    int output_idx = batch_idx * (out_channels * pooled_height * pooled_width) +
                    out_ch * (pooled_height * pooled_width) +
                    pooled_h * pooled_width + pooled_w;
    output[output_idx] = max_val;
}

torch::Tensor fused_conv_tanh_scale_bias_pool_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    float scaling_factor,
    int pool_kernel_size
) {
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    auto bias_sizes = bias.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_height = input_sizes[2];
    int input_width = input_sizes[3];
    
    int out_channels = weight_sizes[0];
    
    int output_height = input_height;
    int output_width = input_width;
    
    int pooled_height = (output_height + pool_kernel_size - 1) / pool_kernel_size;
    int pooled_width = (output_width + pool_kernel_size - 1) / pool_kernel_size;
    
    auto output = torch::zeros({batch_size, out_channels, pooled_height, pooled_width}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    dim3 block_size(16, 16);
    dim3 grid_size(batch_size, out_channels, 
                   (pooled_height * pooled_width + block_size.x * block_size.y - 1) / (block_size.x * block_size.y));
    
    fused_conv_tanh_scale_bias_pool_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        scaling_factor,
        pool_kernel_size,
        pooled_height,
        pooled_width
    );
    
    return output;
}
"""

fused_conv_tanh_scale_bias_pool_cpp_source = """
torch::Tensor fused_conv_tanh_scale_bias_pool_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    float scaling_factor,
    int pool_kernel_size
);
"""

# Compile the inline CUDA code for fused operation
fused_conv_tanh_scale_bias_pool = load_inline(
    name="fused_conv_tanh_scale_bias_pool",
    cpp_sources=fused_conv_tanh_scale_bias_pool_cpp_source,
    cuda_sources=fused_conv_tanh_scale_bias_pool_source,
    functions=["fused_conv_tanh_scale_bias_pool_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for conv+tanh+scale+bias+pool operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scaling_factor = scaling_factor
        self.pool_kernel_size = pool_kernel_size
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Store the fused operation
        self.fused_op = fused_conv_tanh_scale_bias_pool

    def forward(self, x):
        return self.fused_op.fused_conv_tanh_scale_bias_pool_cuda(
            x, 
            self.weight, 
            self.bias, 
            self.kernel_size, 
            self.scaling_factor, 
            self.pool_kernel_size
        )

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 4

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]