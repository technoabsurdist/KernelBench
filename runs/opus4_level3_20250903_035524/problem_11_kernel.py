import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Conv2d + ReLU fusion
conv_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cudnn.h>

__global__ void add_relu_bias_kernel(float* output, const float* bias, int size, int channels, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int c = (idx / spatial_size) % channels;
        float val = output[idx] + bias[c];
        output[idx] = fmaxf(val, 0.0f);
    }
}

torch::Tensor conv2d_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                                int stride, int padding, int dilation, int groups) {
    // Use cuDNN for convolution
    auto output = torch::conv2d(input, weight, {}, stride, padding, dilation, groups);
    
    // Fuse bias add and ReLU
    auto size = output.numel();
    auto channels = output.size(1);
    auto spatial_size = output.size(2) * output.size(3);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    add_relu_bias_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        size, channels, spatial_size
    );
    
    return output;
}
"""

conv_relu_cpp_source = """
torch::Tensor conv2d_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                int stride, int padding, int dilation, int groups);
"""

# Define the custom CUDA kernel for Linear + ReLU fusion
linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void add_bias_relu_kernel(float* output, const float* bias, int batch_size, int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * features;
    
    if (idx < total) {
        int feature_idx = idx % features;
        float val = output[idx] + bias[feature_idx];
        output[idx] = fmaxf(val, 0.0f);
    }
}

torch::Tensor linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    // Flatten input if needed
    input = input.contiguous().view({batch_size, in_features});
    
    // Perform matrix multiplication using cuBLAS (through PyTorch)
    auto output = torch::mm(input, weight.t());
    
    // Fuse bias addition and ReLU
    const int block_size = 256;
    const int total_elements = batch_size * out_features;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    add_bias_relu_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size,
        out_features
    );
    
    return output;
}
"""

linear_relu_cpp_source = """
torch::Tensor linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
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

class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.stride = stride
        self.padding = padding
        
        # Initialize weights
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bias, 0)
        
    def forward(self, x):
        return conv_relu.conv2d_relu_cuda(x, self.weight, self.bias, self.stride, self.padding, 1, 1)

class LinearReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # Initialize weights
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bias, 0)
        
    def forward(self, x):
        return linear_relu.linear_relu_cuda(x, self.weight, self.bias)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # VGG16 architecture with fused Conv+ReLU operations
        self.features = nn.Sequential(
            # Block 1
            ConvReLU(3, 64, kernel_size=3, padding=1),
            ConvReLU(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            ConvReLU(64, 128, kernel_size=3, padding=1),
            ConvReLU(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            ConvReLU(128, 256, kernel_size=3, padding=1),
            ConvReLU(256, 256, kernel_size=3, padding=1),
            ConvReLU(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            ConvReLU(256, 512, kernel_size=3, padding=1),
            ConvReLU(512, 512, kernel_size=3, padding=1),
            ConvReLU(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            ConvReLU(512, 512, kernel_size=3, padding=1),
            ConvReLU(512, 512, kernel_size=3, padding=1),
            ConvReLU(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers with fused Linear+ReLU
        self.fc1 = LinearReLU(512 * 7 * 7, 4096)
        self.fc2 = LinearReLU(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def get_inputs():
    batch_size = 10
    return [torch.rand(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [1000]