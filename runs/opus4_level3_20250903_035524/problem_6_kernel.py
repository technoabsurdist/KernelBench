import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused 1x1 convolutions and concatenation
inception_kernels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 1x1 convolution kernel optimized for inception module
__global__ void conv1x1_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width) {
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = batch_size * out_channels * height * width;
    
    if (out_idx < total_out) {
        int w = out_idx % width;
        int h = (out_idx / width) % height;
        int oc = (out_idx / (width * height)) % out_channels;
        int b = out_idx / (width * height * out_channels);
        
        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
            int in_idx = b * in_channels * height * width + ic * height * width + h * width + w;
            int weight_idx = oc * in_channels + ic;
            sum += input[in_idx] * weight[weight_idx];
        }
        
        if (bias != nullptr) {
            sum += bias[oc];
        }
        output[out_idx] = sum;
    }
}

// Optimized 3x3 convolution kernel
__global__ void conv3x3_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width) {
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = batch_size * out_channels * height * width;
    
    if (out_idx < total_out) {
        int w = out_idx % width;
        int h = (out_idx / width) % height;
        int oc = (out_idx / (width * height)) % out_channels;
        int b = out_idx / (width * height * out_channels);
        
        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = -1; kh <= 1; kh++) {
                for (int kw = -1; kw <= 1; kw++) {
                    int h_in = h + kh;
                    int w_in = w + kw;
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        int in_idx = b * in_channels * height * width + ic * height * width + h_in * width + w_in;
                        int weight_idx = oc * in_channels * 9 + ic * 9 + (kh + 1) * 3 + (kw + 1);
                        sum += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[oc];
        }
        output[out_idx] = sum;
    }
}

// Optimized 5x5 convolution kernel
__global__ void conv5x5_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width) {
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = batch_size * out_channels * height * width;
    
    if (out_idx < total_out) {
        int w = out_idx % width;
        int h = (out_idx / width) % height;
        int oc = (out_idx / (width * height)) % out_channels;
        int b = out_idx / (width * height * out_channels);
        
        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = -2; kh <= 2; kh++) {
                for (int kw = -2; kw <= 2; kw++) {
                    int h_in = h + kh;
                    int w_in = w + kw;
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        int in_idx = b * in_channels * height * width + ic * height * width + h_in * width + w_in;
                        int weight_idx = oc * in_channels * 25 + ic * 25 + (kh + 2) * 5 + (kw + 2);
                        sum += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[oc];
        }
        output[out_idx] = sum;
    }
}

// Fused max pooling + 1x1 conv kernel
__global__ void maxpool_conv1x1_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width) {
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = batch_size * out_channels * height * width;
    
    if (out_idx < total_out) {
        int w = out_idx % width;
        int h = (out_idx / width) % height;
        int oc = (out_idx / (width * height)) % out_channels;
        int b = out_idx / (width * height * out_channels);
        
        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
            float max_val = -1e30f;
            for (int kh = -1; kh <= 1; kh++) {
                for (int kw = -1; kw <= 1; kw++) {
                    int h_in = h + kh;
                    int w_in = w + kw;
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        int in_idx = b * in_channels * height * width + ic * height * width + h_in * width + w_in;
                        max_val = fmaxf(max_val, input[in_idx]);
                    }
                }
            }
            int weight_idx = oc * in_channels + ic;
            sum += max_val * weight[weight_idx];
        }
        
        if (bias != nullptr) {
            sum += bias[oc];
        }
        output[out_idx] = sum;
    }
}

torch::Tensor conv1x1_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());
    
    int total_out = batch_size * out_channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_out + block_size - 1) / block_size;
    
    conv1x1_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, height, width
    );
    
    return output;
}

torch::Tensor conv3x3_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());
    
    int total_out = batch_size * out_channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_out + block_size - 1) / block_size;
    
    conv3x3_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, height, width
    );
    
    return output;
}

torch::Tensor conv5x5_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());
    
    int total_out = batch_size * out_channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_out + block_size - 1) / block_size;
    
    conv5x5_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, height, width
    );
    
    return output;
}

torch::Tensor maxpool_conv1x1_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());
    
    int total_out = batch_size * out_channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_out + block_size - 1) / block_size;
    
    maxpool_conv1x1_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, height, width
    );
    
    return output;
}
"""

inception_cpp_source = """
torch::Tensor conv1x1_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor conv3x3_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor conv5x5_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor maxpool_conv1x1_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

inception_module = load_inline(
    name="inception_module",
    cpp_sources=inception_cpp_source,
    cuda_sources=inception_kernels_source,
    functions=["conv1x1_cuda", "conv3x3_cuda", "conv5x5_cuda", "maxpool_conv1x1_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(ModelNew, self).__init__()
        
        # Initialize weights for 1x1 convolution branch
        self.weight_1x1 = nn.Parameter(torch.randn(out_1x1, in_channels))
        self.bias_1x1 = nn.Parameter(torch.zeros(out_1x1))
        
        # Initialize weights for 3x3 branch
        self.weight_reduce_3x3 = nn.Parameter(torch.randn(reduce_3x3, in_channels))
        self.bias_reduce_3x3 = nn.Parameter(torch.zeros(reduce_3x3))
        self.weight_3x3 = nn.Parameter(torch.randn(out_3x3, reduce_3x3, 3, 3))
        self.bias_3x3 = nn.Parameter(torch.zeros(out_3x3))
        
        # Initialize weights for 5x5 branch
        self.weight_reduce_5x5 = nn.Parameter(torch.randn(reduce_5x5, in_channels))
        self.bias_reduce_5x5 = nn.Parameter(torch.zeros(reduce_5x5))
        self.weight_5x5 = nn.Parameter(torch.randn(out_5x5, reduce_5x5, 5, 5))
        self.bias_5x5 = nn.Parameter(torch.zeros(out_5x5))
        
        # Initialize weights for pooling branch
        self.weight_pool_proj = nn.Parameter(torch.randn(pool_proj, in_channels))
        self.bias_pool_proj = nn.Parameter(torch.zeros(pool_proj))
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.weight_1x1)
        nn.init.xavier_uniform_(self.weight_reduce_3x3)
        nn.init.xavier_uniform_(self.weight_3x3)
        nn.init.xavier_uniform_(self.weight_reduce_5x5)
        nn.init.xavier_uniform_(self.weight_5x5)
        nn.init.xavier_uniform_(self.weight_pool_proj)
        
        self.inception_module = inception_module
    
    def forward(self, x):
        # 1x1 convolution branch
        branch1x1 = self.inception_module.conv1x1_cuda(x, self.weight_1x1, self.bias_1x1)
        
        # 3x3 convolution branch
        reduce_3x3 = self.inception_module.conv1x1_cuda(x, self.weight_reduce_3x3, self.bias_reduce_3x3)
        branch3x3 = self.inception_module.conv3x3_cuda(reduce_3x3, self.weight_3x3.view(self.weight_3x3.size(0), -1), self.bias_3x3)
        
        # 5x5 convolution branch
        reduce_5x5 = self.inception_module.conv1x1_cuda(x, self.weight_reduce_5x5, self.bias_reduce_5x5)
        branch5x5 = self.inception_module.conv5x5_cuda(reduce_5x5, self.weight_5x5.view(self.weight_5x5.size(0), -1), self.bias_5x5)
        
        # Max pooling branch
        branch_pool = self.inception_module.maxpool_conv1x1_cuda(x, self.weight_pool_proj, self.bias_pool_proj)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)