import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Fused 1x1 Conv2d + ReLU kernel
conv1x1_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<typename scalar_t>
__global__ void conv1x1_relu_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_channels * height * width;
    
    if (idx < total_threads) {
        int w = idx % width;
        int h = (idx / width) % height;
        int o = (idx / (width * height)) % out_channels;
        int b = idx / (width * height * out_channels);
        
        scalar_t sum = bias ? bias[o] : 0.0f;
        
        for (int c = 0; c < in_channels; c++) {
            int input_idx = ((b * in_channels + c) * height + h) * width + w;
            int weight_idx = o * in_channels + c;
            sum += input[input_idx] * weight[weight_idx];
        }
        
        // Fused ReLU
        output[idx] = sum > 0 ? sum : 0;
    }
}

torch::Tensor conv1x1_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());
    
    int total_threads = batch_size * out_channels * height * width;
    int threads_per_block = 256;
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv1x1_relu_cuda", ([&] {
        conv1x1_relu_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels, height, width
        );
    }));
    
    return output;
}
"""

conv1x1_relu_cpp_source = "torch::Tensor conv1x1_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Fused 3x3 Conv2d + ReLU kernel
conv3x3_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void conv3x3_relu_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int pad) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_channels * height * width;
    
    if (idx < total_threads) {
        int w = idx % width;
        int h = (idx / width) % height;
        int o = (idx / (width * height)) % out_channels;
        int b = idx / (width * height * out_channels);
        
        scalar_t sum = bias ? bias[o] : 0.0f;
        
        for (int c = 0; c < in_channels; c++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int h_in = h - pad + kh;
                    int w_in = w - pad + kw;
                    
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        int input_idx = ((b * in_channels + c) * height + h_in) * width + w_in;
                        int weight_idx = ((o * in_channels + c) * 3 + kh) * 3 + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Fused ReLU
        output[idx] = sum > 0 ? sum : 0;
    }
}

torch::Tensor conv3x3_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());
    
    int total_threads = batch_size * out_channels * height * width;
    int threads_per_block = 256;
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3x3_relu_cuda", ([&] {
        conv3x3_relu_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels, height, width, 1
        );
    }));
    
    return output;
}
"""

conv3x3_relu_cpp_source = "torch::Tensor conv3x3_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Load custom CUDA kernels
conv1x1_relu_module = load_inline(
    name="conv1x1_relu",
    cpp_sources=conv1x1_relu_cpp_source,
    cuda_sources=conv1x1_relu_source,
    functions=["conv1x1_relu_cuda"],
    verbose=False,
)

conv3x3_relu_module = load_inline(
    name="conv3x3_relu",
    cpp_sources=conv3x3_relu_cpp_source,
    cuda_sources=conv3x3_relu_source,
    functions=["conv3x3_relu_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(ModelNew, self).__init__()
        
        # Initialize weights and biases for squeeze layer
        self.squeeze_weight = nn.Parameter(torch.randn(squeeze_channels, in_channels, 1, 1))
        self.squeeze_bias = nn.Parameter(torch.zeros(squeeze_channels))
        
        # Initialize weights and biases for expand layers
        self.expand1x1_weight = nn.Parameter(torch.randn(expand1x1_channels, squeeze_channels, 1, 1))
        self.expand1x1_bias = nn.Parameter(torch.zeros(expand1x1_channels))
        
        self.expand3x3_weight = nn.Parameter(torch.randn(expand3x3_channels, squeeze_channels, 3, 3))
        self.expand3x3_bias = nn.Parameter(torch.zeros(expand3x3_channels))
        
        # Initialize weights with kaiming uniform
        nn.init.kaiming_uniform_(self.squeeze_weight, a=0, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.expand1x1_weight, a=0, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.expand3x3_weight, a=0, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        # Fused squeeze conv + ReLU
        x = conv1x1_relu_module.conv1x1_relu_cuda(
            x.contiguous(),
            self.squeeze_weight.view(self.squeeze_weight.size(0), self.squeeze_weight.size(1)),
            self.squeeze_bias
        )
        
        # Fused expand1x1 conv + ReLU
        expand1x1_out = conv1x1_relu_module.conv1x1_relu_cuda(
            x,
            self.expand1x1_weight.view(self.expand1x1_weight.size(0), self.expand1x1_weight.size(1)),
            self.expand1x1_bias
        )
        
        # Fused expand3x3 conv + ReLU
        expand3x3_out = conv3x3_relu_module.conv3x3_relu_cuda(
            x,
            self.expand3x3_weight,
            self.expand3x3_bias
        )
        
        # Concatenate outputs
        return torch.cat([expand1x1_out, expand3x3_out], 1)

def get_inputs():
    batch_size = 128
    num_input_features = 3
    height, width = 256, 256
    return [torch.rand(batch_size, num_input_features, height, width)]

def get_init_inputs():
    num_input_features = 3
    squeeze_channels = 6
    expand1x1_channels = 64
    expand3x3_channels = 64
    return [num_input_features, squeeze_channels, expand1x1_channels, expand3x3_channels]