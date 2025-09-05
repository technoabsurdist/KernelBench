import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Conv2d + ReLU
conv2d_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

torch::Tensor conv2d_relu_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                                  int stride, int padding) {
    // Conv2d
    auto output = torch::conv2d(input, weight, bias, {stride, stride}, {padding, padding});
    
    // ReLU fused operation
    auto size = output.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    relu_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), size);
        
    return output;
}
"""

conv2d_relu_cpp_source = """
torch::Tensor conv2d_relu_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                                  int stride, int padding);
"""

# Compile the inline CUDA code for Conv2d + ReLU fusion
conv2d_relu = load_inline(
    name="conv2d_relu",
    cpp_sources=conv2d_relu_cpp_source,
    cuda_sources=conv2d_relu_source,
    functions=["conv2d_relu_forward"],
    verbose=False,
)

# Custom CUDA kernel for inception module concatenation
inception_concat_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void copy_channels_kernel(const float* src, float* dst, int channels, int spatial_size, int dst_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = channels * spatial_size;
    if (idx < total_elements) {
        int channel = idx / spatial_size;
        int spatial_idx = idx % spatial_size;
        dst[dst_offset * spatial_size + channel * spatial_size + spatial_idx] = src[idx];
    }
}

torch::Tensor inception_concat(torch::Tensor branch1x1, torch::Tensor branch3x3, 
                               torch::Tensor branch5x5, torch::Tensor branch_pool) {
    // Get dimensions
    auto batch_size = branch1x1.size(0);
    auto h = branch1x1.size(2);
    auto w = branch1x1.size(3);
    auto spatial_size = h * w;
    
    auto c1 = branch1x1.size(1);
    auto c2 = branch3x3.size(1);
    auto c3 = branch5x5.size(1);
    auto c4 = branch_pool.size(1);
    auto total_channels = c1 + c2 + c3 + c4;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, total_channels, h, w}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(branch1x1.device()));
    
    // Copy each branch to the appropriate location in output
    const int block_size = 256;
    int num_blocks;
    
    // Copy branch1x1
    num_blocks = (c1 * spatial_size + block_size - 1) / block_size;
    copy_channels_kernel<<<num_blocks, block_size>>>(
        branch1x1.data_ptr<float>(), output.data_ptr<float>(), c1, spatial_size, 0);
    
    // Copy branch3x3
    num_blocks = (c2 * spatial_size + block_size - 1) / block_size;
    copy_channels_kernel<<<num_blocks, block_size>>>(
        branch3x3.data_ptr<float>(), output.data_ptr<float>(), c2, spatial_size, c1);
    
    // Copy branch5x5
    num_blocks = (c3 * spatial_size + block_size - 1) / block_size;
    copy_channels_kernel<<<num_blocks, block_size>>>(
        branch5x5.data_ptr<float>(), output.data_ptr<float>(), c3, spatial_size, c1+c2);
    
    // Copy branch_pool
    num_blocks = (c4 * spatial_size + block_size - 1) / block_size;
    copy_channels_kernel<<<num_blocks, block_size>>>(
        branch_pool.data_ptr<float>(), output.data_ptr<float>(), c4, spatial_size, c1+c2+c3);
    
    return output;
}
"""

inception_concat_cpp_source = """
torch::Tensor inception_concat(torch::Tensor branch1x1, torch::Tensor branch3x3, 
                               torch::Tensor branch5x5, torch::Tensor branch_pool);
"""

# Compile the inline CUDA code for inception concatenation
inception_concat = load_inline(
    name="inception_concat",
    cpp_sources=inception_concat_cpp_source,
    cuda_sources=inception_concat_source,
    functions=["inception_concat"],
    verbose=False,
)

class OptimizedInceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(OptimizedInceptionModule, self).__init__()
        
        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # 3x3 convolution branch
        self.branch3x3_1 = nn.Conv2d(in_channels, reduce_3x3, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        
        # 5x5 convolution branch
        self.branch5x5_1 = nn.Conv2d(in_channels, reduce_5x5, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        
        # Max pooling branch
        self.branch_pool_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_conv = nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        
        # Custom CUDA functions
        self.conv2d_relu = conv2d_relu
        self.inception_concat = inception_concat
    
    def forward(self, x):
        # 1x1 branch with fused conv+relu
        branch1x1 = self.conv2d_relu.conv2d_relu_forward(
            x, self.branch1x1.weight, self.branch1x1.bias, 1, 0)
        
        # 3x3 branch with fused conv+relu
        branch3x3 = self.conv2d_relu.conv2d_relu_forward(
            x, self.branch3x3_1.weight, self.branch3x3_1.bias, 1, 0)
        branch3x3 = self.conv2d_relu.conv2d_relu_forward(
            branch3x3, self.branch3x3_2.weight, self.branch3x3_2.bias, 1, 1)
        
        # 5x5 branch with fused conv+relu
        branch5x5 = self.conv2d_relu.conv2d_relu_forward(
            x, self.branch5x5_1.weight, self.branch5x5_1.bias, 1, 0)
        branch5x5 = self.conv2d_relu.conv2d_relu_forward(
            branch5x5, self.branch5x5_2.weight, self.branch5x5_2.bias, 1, 2)
        
        # Pooling branch
        branch_pool = self.branch_pool_pool(x)
        branch_pool = self.conv2d_relu.conv2d_relu_forward(
            branch_pool, self.branch_pool_conv.weight, self.branch_pool_conv.bias, 1, 0)
        
        # Custom concatenation
        return self.inception_concat.inception_concat(branch1x1, branch3x3, branch5x5, branch_pool)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception3a = OptimizedInceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = OptimizedInceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception4a = OptimizedInceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = OptimizedInceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = OptimizedInceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = OptimizedInceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = OptimizedInceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception5a = OptimizedInceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = OptimizedInceptionModule(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(1024, num_classes)
        
        # Custom CUDA functions
        self.conv2d_relu = conv2d_relu
    
    def forward(self, x):
        # Initial layers with fused conv+relu
        x = self.conv2d_relu.conv2d_relu_forward(
            x, self.conv1.weight, self.conv1.bias, 2, 3)
        x = self.maxpool1(x)
        
        x = self.conv2d_relu.conv2d_relu_forward(
            x, self.conv2.weight, self.conv2.bias, 1, 0)
        
        x = self.conv2d_relu.conv2d_relu_forward(
            x, self.conv3.weight, self.conv3.bias, 1, 1)
        x = self.maxpool2(x)
        
        # Inception modules
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x