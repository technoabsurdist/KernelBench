import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused Conv2d + ReLU CUDA kernel
conv_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cudnn.h>

template <typename scalar_t>
__global__ void conv_relu_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding) {
    
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_output_size = batch_size * out_channels * out_height * out_width;
    
    if (out_idx < total_output_size) {
        const int w = out_idx % out_width;
        const int h = (out_idx / out_width) % out_height;
        const int c = (out_idx / (out_width * out_height)) % out_channels;
        const int n = out_idx / (out_width * out_height * out_channels);
        
        scalar_t sum = 0;
        
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int ih = h * stride - padding + kh;
                    const int iw = w * stride - padding + kw;
                    
                    if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                        const int in_idx = n * in_channels * in_height * in_width + 
                                          ic * in_height * in_width + 
                                          ih * in_width + iw;
                        const int weight_idx = c * in_channels * kernel_size * kernel_size + 
                                             ic * kernel_size * kernel_size + 
                                             kh * kernel_size + kw;
                        sum += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[c];
        }
        
        // Fused ReLU
        output[out_idx] = sum > 0 ? sum : 0;
    }
}

torch::Tensor conv_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                              input.options());
    
    const int total_output_size = batch_size * out_channels * out_height * out_width;
    const int threads = 256;
    const int blocks = (total_output_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_relu_cuda", ([&] {
        conv_relu_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            in_height, in_width, out_height, out_width,
            kernel_size, stride, padding
        );
    }));
    
    return output;
}
"""

conv_relu_cpp_source = """
torch::Tensor conv_relu_cuda(torch::Tensor input, torch::Tensor weight, 
                             torch::Tensor bias, int stride, int padding);
"""

# MaxPool2d CUDA kernel
maxpool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void maxpool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride) {
    
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_output_size = batch_size * channels * out_height * out_width;
    
    if (out_idx < total_output_size) {
        const int w = out_idx % out_width;
        const int h = (out_idx / out_width) % out_height;
        const int c = (out_idx / (out_width * out_height)) % channels;
        const int n = out_idx / (out_width * out_height * channels);
        
        scalar_t max_val = -1e10;
        
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int ih = h * stride + kh;
                const int iw = w * stride + kw;
                
                if (ih < in_height && iw < in_width) {
                    const int in_idx = n * channels * in_height * in_width + 
                                      c * in_height * in_width + 
                                      ih * in_width + iw;
                    max_val = fmaxf(max_val, input[in_idx]);
                }
            }
        }
        
        output[out_idx] = max_val;
    }
}

torch::Tensor maxpool2d_cuda(torch::Tensor input, int kernel_size, int stride) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    const int out_height = (in_height - kernel_size) / stride + 1;
    const int out_width = (in_width - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, channels, out_height, out_width}, 
                              input.options());
    
    const int total_output_size = batch_size * channels * out_height * out_width;
    const int threads = 256;
    const int blocks = (total_output_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "maxpool2d_cuda", ([&] {
        maxpool2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, channels, in_height, in_width,
            out_height, out_width, kernel_size, stride
        );
    }));
    
    return output;
}
"""

maxpool_cpp_source = """
torch::Tensor maxpool2d_cuda(torch::Tensor input, int kernel_size, int stride);
"""

# Linear + ReLU fused kernel
linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

template <typename scalar_t>
__global__ void add_bias_relu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ bias,
    const int batch_size,
    const int out_features) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = batch_size * out_features;
    
    if (idx < total_size) {
        const int feature_idx = idx % out_features;
        scalar_t val = output[idx] + bias[feature_idx];
        output[idx] = val > 0 ? val : 0;
    }
}

torch::Tensor linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);
    
    auto output = torch::mm(input, weight.t());
    
    const int total_size = batch_size * out_features;
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "add_bias_relu", ([&] {
        add_bias_relu_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            batch_size, out_features
        );
    }));
    
    return output;
}
"""

linear_relu_cpp_source = """
torch::Tensor linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Load CUDA kernels
conv_relu_cuda = load_inline(
    name="conv_relu_cuda",
    cpp_sources=conv_relu_cpp_source,
    cuda_sources=conv_relu_source,
    functions=["conv_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

maxpool_cuda = load_inline(
    name="maxpool_cuda",
    cpp_sources=maxpool_cpp_source,
    cuda_sources=maxpool_source,
    functions=["maxpool2d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

linear_relu_cuda = load_inline(
    name="linear_relu_cuda",
    cpp_sources=linear_relu_cpp_source,
    cuda_sources=linear_relu_source,
    functions=["linear_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # Initialize weights for custom kernels
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        
        self.dropout = nn.Dropout(p=0.0)
        
    def forward(self, x):
        # Block 1
        x = conv_relu_cuda.conv_relu_cuda(x, self.conv1_1.weight, self.conv1_1.bias, 1, 1)
        x = conv_relu_cuda.conv_relu_cuda(x, self.conv1_2.weight, self.conv1_2.bias, 1, 1)
        x = maxpool_cuda.maxpool2d_cuda(x, 2, 2)
        
        # Block 2
        x = conv_relu_cuda.conv_relu_cuda(x, self.conv2_1.weight, self.conv2_1.bias, 1, 1)
        x = conv_relu_cuda.conv_relu_cuda(x, self.conv2_2.weight, self.conv2_2.bias, 1, 1)
        x = maxpool_cuda.maxpool2d_cuda(x, 2, 2)
        
        # Block 3
        x = conv_relu_cuda.conv_relu_cuda(x, self.conv3_1.weight, self.conv3_1.bias, 1, 1)
        x = conv_relu_cuda.conv_relu_cuda(x, self.conv3_2.weight, self.conv3_2.bias, 1, 1)
        x = conv_relu_cuda.conv_relu_cuda(x, self.conv3_3.weight, self.conv3_3.bias, 1, 1)
        x = conv_relu_cuda.conv_relu_cuda(x, self.conv3_4.weight, self.conv3_4.bias, 1, 1)
        x = maxpool_cuda.maxpool2d_cuda(x, 2, 2)
        
        # Block 4
        x = conv_relu_cuda.conv_relu_cuda(x, self.conv4_1.weight, self.conv4_1.bias, 1, 1)
        x = conv_relu_cuda.conv_relu_cuda(x, self.conv4_2.weight, self.conv4_2.bias, 1, 1)
        x = conv_relu_cuda.conv_relu_cuda(x, self.conv4_3.weight, self.conv4_3.bias, 1, 1)
        x = conv_relu_cuda.conv_relu_cuda(x, self.conv4_4.weight, self.conv4_4.bias, 1, 1)
        x = maxpool_cuda.maxpool2d_cuda(x, 2, 2)
        
        # Block 5
        x = conv_relu_cuda.conv_relu_cuda(x, self.conv5_1.weight, self.conv5_1.bias, 1, 1)
        x = conv_relu_cuda.conv_relu_cuda(x, self.conv5_2.weight, self.conv5_2.bias, 1, 1)
        x = conv_relu_cuda.conv_relu_cuda(x, self.conv5_3.weight, self.conv5_3.bias, 1, 1)
        x = conv_relu_cuda.conv_relu_cuda(x, self.conv5_4.weight, self.conv5_4.bias, 1, 1)
        x = maxpool_cuda.maxpool2d_cuda(x, 2, 2)
        
        # Classifier
        x = torch.flatten(x, 1)
        x = linear_relu_cuda.linear_relu_cuda(x, self.fc1.weight, self.fc1.bias)
        x = self.dropout(x)
        x = linear_relu_cuda.linear_relu_cuda(x, self.fc2.weight, self.fc2.bias)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def get_inputs():
    return [torch.rand(10, 3, 224, 224).cuda()]

def get_init_inputs():
    return [1000]