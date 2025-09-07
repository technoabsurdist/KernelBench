import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv-subtract-hardswish-maxpool-mish
fused_conv_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// HardSwish function: x * relu6(x + 3) / 6
__device__ float hardswish(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

// Mish function: x * tanh(softplus(x))
__device__ float mish(float x) {
    return x * tanhf(log1pf(expf(x)));
}

// Convolution kernel (simplified for 3x3 kernel case)
__global__ void conv3x3_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int out_height,
    int out_width,
    float subtract_value
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx < total_elements) {
        int w_out = idx % out_width;
        int h_out = (idx / out_width) % out_height;
        int c_out = (idx / (out_width * out_height)) % out_channels;
        int n = idx / (out_width * out_height * out_channels);
        
        float sum = 0.0f;
        
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int h_in = h_out + kh - 1;
                    int w_in = w_out + kw - 1;
                    
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        int input_idx = ((n * in_channels + c_in) * height + h_in) * width + w_in;
                        int weight_idx = ((c_out * in_channels + c_in) * 3 + kh) * 3 + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        sum += bias[c_out];
        sum -= subtract_value;
        sum = hardswish(sum);
        output[idx] = sum;
    }
}

// MaxPool2D kernel
__global__ void maxpool2d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int pool_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_height * out_width;
    
    if (idx < total_elements) {
        int w_out = idx % out_width;
        int h_out = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % channels;
        int n = idx / (out_width * out_height * channels);
        
        float max_val = -1e38f;
        
        for (int ph = 0; ph < pool_size; ph++) {
            for (int pw = 0; pw < pool_size; pw++) {
                int h_in = h_out * pool_size + ph;
                int w_in = w_out * pool_size + pw;
                
                if (h_in < in_height && w_in < in_width) {
                    int input_idx = ((n * channels + c) * in_height + h_in) * in_width + w_in;
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }
        
        output[idx] = max_val;
    }
}

// Mish activation kernel
__global__ void mish_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = mish(data[idx]);
    }
}

torch::Tensor fused_conv_activation_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value,
    int pool_kernel_size
) {
    // Get dimensions
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int height = input_sizes[2];
    int width = input_sizes[3];
    
    int out_channels = weight_sizes[0];
    
    // Conv output size (assuming padding=1, stride=1 for 3x3 conv)
    int out_height = height;
    int out_width = width;
    
    // After maxpool
    int pooled_height = out_height / pool_kernel_size;
    int pooled_width = out_width / pool_kernel_size;
    
    // Allocate output tensor
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto conv_output = torch::zeros({batch_size, out_channels, out_height, out_width}, options);
    auto pool_output = torch::zeros({batch_size, out_channels, pooled_height, pooled_width}, options);
    
    // Launch conv kernel
    const int conv_block_size = 256;
    const int conv_num_blocks = (batch_size * out_channels * out_height * out_width + conv_block_size - 1) / conv_block_size;
    
    conv3x3_kernel<<<conv_num_blocks, conv_block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        conv_output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        out_height,
        out_width,
        subtract_value
    );
    
    // Launch maxpool kernel
    const int pool_block_size = 256;
    const int pool_num_blocks = (batch_size * out_channels * pooled_height * pooled_width + pool_block_size - 1) / pool_block_size;
    
    maxpool2d_kernel<<<pool_num_blocks, pool_block_size>>>(
        conv_output.data_ptr<float>(),
        pool_output.data_ptr<float>(),
        batch_size,
        out_channels,
        out_height,
        out_width,
        pooled_height,
        pooled_width,
        pool_kernel_size
    );
    
    // Launch mish kernel
    const int mish_block_size = 256;
    const int mish_num_blocks = (batch_size * out_channels * pooled_height * pooled_width + mish_block_size - 1) / mish_block_size;
    
    mish_kernel<<<mish_num_blocks, mish_block_size>>>(
        pool_output.data_ptr<float>(),
        batch_size * out_channels * pooled_height * pooled_width
    );
    
    return pool_output;
}
"""

fused_conv_activation_cpp_source = """
torch::Tensor fused_conv_activation_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value,
    int pool_kernel_size
);
"""

# Compile the inline CUDA code
fused_conv_activation = load_inline(
    name="fused_conv_activation",
    cpp_sources=fused_conv_activation_cpp_source,
    cuda_sources=fused_conv_activation_source,
    functions=["fused_conv_activation_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size
        
        # Create convolution parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Initialize weights similar to PyTorch's default
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        nn.init.zeros_(self.bias)
        
        self.fused_conv_activation = fused_conv_activation

    def forward(self, x):
        return self.fused_conv_activation.fused_conv_activation_cuda(
            x, self.weight, self.bias, self.subtract_value, self.pool_kernel_size
        )