import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Conv2d + ReLU
conv_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_relu_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, 
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int kernel_size,
    int out_height, int out_width, int stride, int pad) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_size = batch_size * out_channels * out_height * out_width;
    
    if (idx < total_output_size) {
        int w_out = idx % out_width;
        int h_out = (idx / out_width) % out_height;
        int c_out = (idx / (out_width * out_height)) % out_channels;
        int batch = idx / (out_channels * out_height * out_width);
        
        float sum = 0.0f;
        
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int h_in = h_out * stride - pad + kh;
                    int w_in = w_out * stride - pad + kw;
                    
                    if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                        int input_idx = batch * (in_channels * in_height * in_width) +
                                       c_in * (in_height * in_width) +
                                       h_in * in_width + w_in;
                        int weight_idx = c_out * (in_channels * kernel_size * kernel_size) +
                                        c_in * (kernel_size * kernel_size) +
                                        kh * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[c_out];
        }
        
        // Apply ReLU
        output[idx] = fmaxf(sum, 0.0f);
    }
}

torch::Tensor conv2d_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                int stride, int padding) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                              input.options());
    
    int total_output_size = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_output_size + block_size - 1) / block_size;
    
    conv2d_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_height, in_width, kernel_size,
        out_height, out_width, stride, padding
    );
    
    return output;
}
"""

conv_relu_cpp_source = "torch::Tensor conv2d_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding);"

# Custom CUDA kernel for fused Linear + ReLU
linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void linear_relu_kernel(const float* input, const float* weight, const float* bias,
                                    float* output, int batch_size, int in_features, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * out_features;
    
    if (idx < total_size) {
        int batch = idx / out_features;
        int out_idx = idx % out_features;
        
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += input[batch * in_features + i] * weight[out_idx * in_features + i];
        }
        
        if (bias != nullptr) {
            sum += bias[out_idx];
        }
        
        // Apply ReLU
        output[idx] = fmaxf(sum, 0.0f);
    }
}

torch::Tensor linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    const int block_size = 256;
    int total_size = batch_size * out_features;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    linear_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_features, out_features
    );
    
    return output;
}
"""

linear_relu_cpp_source = "torch::Tensor linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Compile the inline CUDA kernels
conv_relu = load_inline(
    name="conv_relu",
    cpp_sources=conv_relu_cpp_source,
    cuda_sources=conv_relu_source,
    functions=["conv2d_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

linear_relu = load_inline(
    name="linear_relu",
    cpp_sources=linear_relu_cpp_source,
    cuda_sources=linear_relu_source,
    functions=["linear_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, num_classes):
        super(ModelNew, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
        
        # Custom CUDA operators
        self.conv_relu = conv_relu
        self.linear_relu = linear_relu
    
    def forward(self, x):
        # First convolutional layer with fused ReLU activation
        x = self.conv_relu.conv2d_relu_cuda(
            x, self.conv1.weight, self.conv1.bias, 1, 0
        )
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Second convolutional layer with fused ReLU activation
        x = self.conv_relu.conv2d_relu_cuda(
            x, self.conv2.weight, self.conv2.bias, 1, 0
        )
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 16*5*5)
        
        # First fully connected layer with fused ReLU activation
        x = self.linear_relu.linear_relu_cuda(x, self.fc1.weight, self.fc1.bias)
        
        # Second fully connected layer with fused ReLU activation
        x = self.linear_relu.linear_relu_cuda(x, self.fc2.weight, self.fc2.bias)
        
        # Final fully connected layer
        x = F.linear(x, self.fc3.weight, self.fc3.bias)
        
        return x

def get_inputs():
    return [torch.rand(4096, 1, 32, 32).cuda()]

def get_init_inputs():
    return [20]