import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Conv2d + ReLU
conv_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cudnn.h>

torch::Tensor conv2d_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups) {
    
    // Perform standard conv2d
    auto output = torch::conv2d(
        input, weight, bias,
        {stride_h, stride_w},
        {pad_h, pad_w},
        {dilation_h, dilation_w},
        groups
    );
    
    // Apply ReLU in-place
    return output.relu_();
}
"""

conv_relu_cpp_source = """
torch::Tensor conv2d_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups);
"""

# Custom CUDA kernel for fused Linear + ReLU
linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void add_bias_relu_kernel(float* output, const float* bias, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx < total_elements) {
        int feature_idx = idx % out_features;
        float val = output[idx] + bias[feature_idx];
        output[idx] = fmaxf(0.0f, val);
    }
}

torch::Tensor linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    // Allocate output tensor
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    // Perform matrix multiplication using cuBLAS
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        out_features,
        batch_size,
        in_features,
        &alpha,
        weight.data_ptr<float>(),
        in_features,
        input.data_ptr<float>(),
        in_features,
        &beta,
        output.data_ptr<float>(),
        out_features
    );
    
    // Add bias and apply ReLU
    const int threads = 256;
    const int blocks = (batch_size * out_features + threads - 1) / threads;
    add_bias_relu_kernel<<<blocks, threads>>>(
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
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

linear_relu = load_inline(
    name="linear_relu",
    cpp_sources=linear_relu_cpp_source,
    cuda_sources=linear_relu_source,
    functions=["linear_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
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
        
        self.conv_relu = conv_relu
        self.linear_relu = linear_relu
    
    def forward(self, x):
        # Conv1 + ReLU (fused)
        x = self.conv_relu.conv2d_relu_cuda(
            x, self.conv1.weight, self.conv1.bias,
            4, 4, 2, 2, 1, 1, 1
        )
        x = self.maxpool1(x)
        
        # Conv2 + ReLU (fused)
        x = self.conv_relu.conv2d_relu_cuda(
            x, self.conv2.weight, self.conv2.bias,
            1, 1, 2, 2, 1, 1, 1
        )
        x = self.maxpool2(x)
        
        # Conv3 + ReLU (fused)
        x = self.conv_relu.conv2d_relu_cuda(
            x, self.conv3.weight, self.conv3.bias,
            1, 1, 1, 1, 1, 1, 1
        )
        
        # Conv4 + ReLU (fused)
        x = self.conv_relu.conv2d_relu_cuda(
            x, self.conv4.weight, self.conv4.bias,
            1, 1, 1, 1, 1, 1, 1
        )
        
        # Conv5 + ReLU (fused)
        x = self.conv_relu.conv2d_relu_cuda(
            x, self.conv5.weight, self.conv5.bias,
            1, 1, 1, 1, 1, 1, 1
        )
        x = self.maxpool3(x)
        
        x = torch.flatten(x, 1)
        
        # FC1 + ReLU (fused)
        x = self.linear_relu.linear_relu_cuda(x, self.fc1.weight, self.fc1.bias)
        x = self.dropout1(x)
        
        # FC2 + ReLU (fused)
        x = self.linear_relu.linear_relu_cuda(x, self.fc2.weight, self.fc2.bias)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x

batch_size = 1024
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]