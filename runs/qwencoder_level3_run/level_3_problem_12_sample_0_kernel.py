import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + ReLU
conv2d_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void conv2d_relu_kernel(const float* input, const float* weight, const float* bias, 
                                   float* output, int batch_size, int in_channels, int in_height, int in_width,
                                   int out_channels, int out_height, int out_width,
                                   int kernel_size, int padding, int stride) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (out_idx >= total_elements) return;
    
    int w = out_idx % out_width;
    int h = (out_idx / out_width) % out_height;
    int c = (out_idx / (out_width * out_height)) % out_channels;
    int n = out_idx / (out_width * out_height * out_channels);
    
    float sum = 0.0f;
    
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int in_h = h * stride + kh - padding;
                int in_w = w * stride + kw - padding;
                
                if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                    int input_idx = n * (in_channels * in_height * in_width) + 
                                    ic * (in_height * in_width) + 
                                    in_h * in_width + in_w;
                    int weight_idx = c * (in_channels * kernel_size * kernel_size) + 
                                     ic * (kernel_size * kernel_size) + 
                                     kh * kernel_size + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    sum += bias[c];
    output[out_idx] = fmaxf(0.0f, sum); // ReLU activation
}

torch::Tensor conv2d_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                               int kernel_size, int padding, int stride) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(0);
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv2d_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, in_height, in_width,
        out_channels, out_height, out_width, kernel_size, padding, stride
    );
    
    return output;
}
"""

conv2d_relu_cpp_source = """
torch::Tensor conv2d_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                               int kernel_size, int padding, int stride);
"""

# Compile the inline CUDA code for Conv2d + ReLU fusion
conv2d_relu = load_inline(
    name="conv2d_relu",
    cpp_sources=conv2d_relu_cpp_source,
    cuda_sources=conv2d_relu_source,
    functions=["conv2d_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for fused Linear + ReLU
linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void linear_relu_kernel(const float* input, const float* weight, const float* bias,
                                   float* output, int batch_size, int in_features, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx >= total_elements) return;
    
    int out_idx = idx % out_features;
    int batch_idx = idx / out_features;
    
    float sum = 0.0f;
    for (int i = 0; i < in_features; i++) {
        sum += input[batch_idx * in_features + i] * weight[out_idx * in_features + i];
    }
    sum += bias[out_idx];
    
    output[idx] = fmaxf(0.0f, sum); // ReLU activation
}

torch::Tensor linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    const int total_elements = batch_size * out_features;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    linear_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_features, out_features
    );
    
    return output;
}
"""

linear_relu_cpp_source = """
torch::Tensor linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code for Linear + ReLU fusion
linear_relu = load_inline(
    name="linear_relu",
    cpp_sources=linear_relu_cpp_source,
    cuda_sources=linear_relu_source,
    functions=["linear_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(Conv2dReLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Initialize weights
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bias, 0)
        
        self.conv2d_relu = conv2d_relu

    def forward(self, x):
        return self.conv2d_relu.conv2d_relu_cuda(x, self.weight, self.bias, 
                                                 self.kernel_size, self.padding, self.stride)

class LinearReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearReLU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # Initialize weights
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.bias, 0)
        
        self.linear_relu = linear_relu

    def forward(self, x):
        return self.linear_relu.linear_relu_cuda(x, self.weight, self.bias)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Optimized VGG19 model with custom CUDA kernels.
        """
        super(ModelNew, self).__init__()
        
        # VGG19 architecture with fused Conv2d+ReLU layers
        self.features = nn.Sequential(
            # Block 1
            Conv2dReLU(3, 64, kernel_size=3, padding=1),
            Conv2dReLU(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            Conv2dReLU(64, 128, kernel_size=3, padding=1),
            Conv2dReLU(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            Conv2dReLU(128, 256, kernel_size=3, padding=1),
            Conv2dReLU(256, 256, kernel_size=3, padding=1),
            Conv2dReLU(256, 256, kernel_size=3, padding=1),
            Conv2dReLU(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            Conv2dReLU(256, 512, kernel_size=3, padding=1),
            Conv2dReLU(512, 512, kernel_size=3, padding=1),
            Conv2dReLU(512, 512, kernel_size=3, padding=1),
            Conv2dReLU(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            Conv2dReLU(512, 512, kernel_size=3, padding=1),
            Conv2dReLU(512, 512, kernel_size=3, padding=1),
            Conv2dReLU(512, 512, kernel_size=3, padding=1),
            Conv2dReLU(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            LinearReLU(512 * 7 * 7, 4096),
            nn.Dropout(p=0.0),
            LinearReLU(4096, 4096),
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass of the optimized VGG19 model.

        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, 3, 224, 224).cuda()]

def get_init_inputs():
    return [num_classes]