import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Fused BatchNorm2d + ReLU kernel
bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bn_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int height,
    int width,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * height * width;
    
    if (idx < total_size) {
        int w = idx % width;
        int h = (idx / width) % height;
        int c = (idx / (width * height)) % channels;
        int b = idx / (width * height * channels);
        
        float val = input[idx];
        float mean = running_mean[c];
        float var = running_var[c];
        float scale = weight[c];
        float shift = bias[c];
        
        // BatchNorm
        float normalized = (val - mean) / sqrtf(var + eps);
        float bn_out = normalized * scale + shift;
        
        // ReLU
        output[idx] = fmaxf(0.0f, bn_out);
    }
}

torch::Tensor bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps) {
    
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto output = torch::zeros_like(input);
    
    int total_size = batch_size * channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    bn_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width, eps
    );
    
    return output;
}
"""

bn_relu_cpp_source = "torch::Tensor bn_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor running_mean, torch::Tensor running_var, float eps);"

# Fused Conv2d 1x1 + AvgPool2d kernel
conv_avgpool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1x1_avgpool_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * out_channels * out_height * out_width;
    
    if (idx < total_size) {
        int ow = idx % out_width;
        int oh = (idx / out_width) % out_height;
        int oc = (idx / (out_width * out_height)) % out_channels;
        int b = idx / (out_width * out_height * out_channels);
        
        // Calculate input coordinates for average pooling
        int ih_start = oh * 2;
        int iw_start = ow * 2;
        
        float sum = 0.0f;
        
        // Perform 1x1 convolution + average pooling in one go
        for (int ic = 0; ic < in_channels; ic++) {
            float conv_sum = 0.0f;
            
            // Average pool 2x2 region
            for (int kh = 0; kh < 2; kh++) {
                for (int kw = 0; kw < 2; kw++) {
                    int ih = ih_start + kh;
                    int iw = iw_start + kw;
                    
                    if (ih < in_height && iw < in_width) {
                        int input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                        conv_sum += input[input_idx];
                    }
                }
            }
            
            // Average the pooled values and apply 1x1 convolution weight
            conv_sum /= 4.0f;
            sum += conv_sum * weight[oc * in_channels + ic];
        }
        
        output[idx] = sum;
    }
}

torch::Tensor conv1x1_avgpool_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int out_channels) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    
    int out_height = in_height / 2;
    int out_width = in_width / 2;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                              input.options());
    
    int total_size = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    conv1x1_avgpool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_height, in_width, out_height, out_width
    );
    
    return output;
}
"""

conv_avgpool_cpp_source = "torch::Tensor conv1x1_avgpool_cuda(torch::Tensor input, torch::Tensor weight, int out_channels);"

# Load the custom CUDA kernels
bn_relu_module = load_inline(
    name="bn_relu",
    cpp_sources=bn_relu_cpp_source,
    cuda_sources=bn_relu_source,
    functions=["bn_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

conv_avgpool_module = load_inline(
    name="conv_avgpool",
    cpp_sources=conv_avgpool_cpp_source,
    cuda_sources=conv_avgpool_source,
    functions=["conv1x1_avgpool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(ModelNew, self).__init__()
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        
        # Initialize BatchNorm parameters
        self.bn_weight = nn.Parameter(torch.ones(num_input_features))
        self.bn_bias = nn.Parameter(torch.zeros(num_input_features))
        self.register_buffer('bn_running_mean', torch.zeros(num_input_features))
        self.register_buffer('bn_running_var', torch.ones(num_input_features))
        self.bn_eps = 1e-5
        
        # Initialize Conv1x1 weight
        self.conv_weight = nn.Parameter(torch.randn(num_output_features, num_input_features) * 0.01)
        
        self.bn_relu_module = bn_relu_module
        self.conv_avgpool_module = conv_avgpool_module
        
    def forward(self, x):
        x = x.cuda()
        
        # Fused BatchNorm + ReLU
        x = self.bn_relu_module.bn_relu_cuda(
            x.contiguous(),
            self.bn_weight,
            self.bn_bias,
            self.bn_running_mean,
            self.bn_running_var,
            self.bn_eps
        )
        
        # Fused Conv1x1 + AvgPool
        x = self.conv_avgpool_module.conv1x1_avgpool_cuda(
            x.contiguous(),
            self.conv_weight.view(-1),
            self.num_output_features
        )
        
        return x

batch_size = 128
num_input_features = 32
num_output_features = 64
height, width = 256, 256

def get_inputs():
    return [torch.rand(batch_size, num_input_features, height, width)]

def get_init_inputs():
    return [num_input_features, num_output_features]