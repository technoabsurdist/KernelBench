import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + ReLU
conv2d_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void conv2d_relu_kernel(
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
    int stride,
    int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % out_channels;
    int b = idx / (out_width * out_height * out_channels);
    
    float sum = 0.0f;
    
    for (int ic = 0; ic < in_channels; ic++) {
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_x = w * stride - padding + kx;
                int in_y = h * stride - padding + ky;
                
                if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
                    int input_idx = b * (in_channels * in_height * in_width) +
                                   ic * (in_height * in_width) +
                                   in_y * in_width + in_x;
                                   
                    int weight_idx = c * (in_channels * kernel_size * kernel_size) +
                                    ic * (kernel_size * kernel_size) +
                                    ky * kernel_size + kx;
                                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    sum += bias[c];
    output[idx] = fmaxf(0.0f, sum); // ReLU activation
}

torch::Tensor conv2d_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding
) {
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int in_height = input_sizes[2];
    int in_width = input_sizes[3];
    
    int out_channels = weight_sizes[0];
    int kernel_size = weight_sizes[2];
    
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    const int num_elements = batch_size * out_channels * out_height * out_width;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    
    conv2d_relu_kernel<<<num_blocks, block_size>>>(
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
        stride,
        padding
    );
    
    return output;
}
"""

conv2d_relu_cpp_source = """
torch::Tensor conv2d_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding
);
"""

# Define the custom CUDA kernel for fused Linear + ReLU
linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void linear_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx >= total_elements) return;
    
    int out_idx = idx % out_features;
    int batch_idx = idx / out_features;
    
    float sum = 0.0f;
    
    for (int i = 0; i < in_features; i++) {
        int input_idx = batch_idx * in_features + i;
        int weight_idx = out_idx * in_features + i;
        sum += input[input_idx] * weight[weight_idx];
    }
    
    sum += bias[out_idx];
    output[idx] = fmaxf(0.0f, sum); // ReLU activation
}

torch::Tensor linear_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_features = input_sizes[1];
    int out_features = weight_sizes[0];
    
    auto output = torch::zeros({batch_size, out_features}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    const int num_elements = batch_size * out_features;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    
    linear_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

linear_relu_cpp_source = """
torch::Tensor linear_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
);
"""

# Compile the inline CUDA code
conv2d_relu = load_inline(
    name="conv2d_relu",
    cpp_sources=conv2d_relu_cpp_source,
    cuda_sources=conv2d_relu_source,
    functions=["conv2d_relu_cuda"],
    verbose=False,
)

linear_relu = load_inline(
    name="linear_relu",
    cpp_sources=linear_relu_cpp_source,
    cuda_sources=linear_relu_source,
    functions=["linear_relu_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: The number of output classes (default is 1000 for ImageNet)
        """
        super(ModelNew, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        
        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        
        # Fifth convolutional layer
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.dropout1 = nn.Dropout(p=0.0)
        
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.dropout2 = nn.Dropout(p=0.0)
        
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)
        
        # Load custom CUDA extensions
        self.conv2d_relu = conv2d_relu
        self.linear_relu = linear_relu
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # Fused Conv2d + ReLU operations
        x = self.conv2d_relu.conv2d_relu_cuda(x, self.conv1.weight, self.conv1.bias, 4, 2)
        x = self.maxpool1(x)
        
        x = self.conv2d_relu.conv2d_relu_cuda(x, self.conv2.weight, self.conv2.bias, 1, 2)
        x = self.maxpool2(x)
        
        x = self.conv2d_relu.conv2d_relu_cuda(x, self.conv3.weight, self.conv3.bias, 1, 1)
        
        x = self.conv2d_relu.conv2d_relu_cuda(x, self.conv4.weight, self.conv4.bias, 1, 1)
        
        x = self.conv2d_relu.conv2d_relu_cuda(x, self.conv5.weight, self.conv5.bias, 1, 1)
        x = self.maxpool3(x)
        
        x = torch.flatten(x, 1)
        
        # Fused Linear + ReLU operations
        x = self.linear_relu.linear_relu_cuda(x, self.fc1.weight, self.fc1.bias)
        x = self.dropout1(x)
        
        x = self.linear_relu.linear_relu_cuda(x, self.fc2.weight, self.fc2.bias)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x