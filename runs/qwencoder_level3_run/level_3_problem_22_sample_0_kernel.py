import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels
conv_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void relu6_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fminf(fmaxf(0.0f, data[idx]), 6.0f);
    }
}

__global__ void add_tensors(float* a, float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor fused_conv_bn_relu(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                                torch::Tensor running_mean, torch::Tensor running_var, 
                                torch::Tensor gamma, torch::Tensor beta, 
                                int64_t stride, int64_t padding) {
    // Convolution
    torch::Tensor conv_out = torch::conv2d(input, weight, bias, {stride, stride}, {padding, padding});
    
    // Batch norm: (x - mean) / sqrt(var + eps) * gamma + beta
    float eps = 1e-5;
    torch::Tensor normalized = (conv_out - running_mean.view({1, -1, 1, 1})) / 
                              torch::sqrt(running_var.view({1, -1, 1, 1}) + eps);
    torch::Tensor bn_out = normalized * gamma.view({1, -1, 1, 1}) + beta.view({1, -1, 1, 1});
    
    // ReLU
    auto size = bn_out.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    relu_kernel<<<num_blocks, block_size>>>(bn_out.data_ptr<float>(), size);
    
    return bn_out;
}

torch::Tensor fused_conv_bn_relu6(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                                 torch::Tensor running_mean, torch::Tensor running_var, 
                                 torch::Tensor gamma, torch::Tensor beta, 
                                 int64_t stride, int64_t padding) {
    // Convolution
    torch::Tensor conv_out = torch::conv2d(input, weight, bias, {stride, stride}, {padding, padding});
    
    // Batch norm
    float eps = 1e-5;
    torch::Tensor normalized = (conv_out - running_mean.view({1, -1, 1, 1})) / 
                              torch::sqrt(running_var.view({1, -1, 1, 1}) + eps);
    torch::Tensor bn_out = normalized * gamma.view({1, -1, 1, 1}) + beta.view({1, -1, 1, 1});
    
    // ReLU6
    auto size = bn_out.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    relu6_kernel<<<num_blocks, block_size>>>(bn_out.data_ptr<float>(), size);
    
    return bn_out;
}

torch::Tensor fused_conv_bn(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                           torch::Tensor running_mean, torch::Tensor running_var, 
                           torch::Tensor gamma, torch::Tensor beta, 
                           int64_t stride, int64_t padding) {
    // Convolution
    torch::Tensor conv_out = torch::conv2d(input, weight, bias, {stride, stride}, {padding, padding});
    
    // Batch norm
    float eps = 1e-5;
    torch::Tensor normalized = (conv_out - running_mean.view({1, -1, 1, 1})) / 
                              torch::sqrt(running_var.view({1, -1, 1, 1}) + eps);
    torch::Tensor bn_out = normalized * gamma.view({1, -1, 1, 1}) + beta.view({1, -1, 1, 1});
    
    return bn_out;
}

torch::Tensor residual_add(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    add_tensors<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), 
                                           out.data_ptr<float>(), size);
    
    return out;
}
"""

conv_bn_relu_cpp_source = """
torch::Tensor fused_conv_bn_relu(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                                torch::Tensor running_mean, torch::Tensor running_var, 
                                torch::Tensor gamma, torch::Tensor beta, 
                                int64_t stride, int64_t padding);

torch::Tensor fused_conv_bn_relu6(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                                 torch::Tensor running_mean, torch::Tensor running_var, 
                                 torch::Tensor gamma, torch::Tensor beta, 
                                 int64_t stride, int64_t padding);

torch::Tensor fused_conv_bn(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                           torch::Tensor running_mean, torch::Tensor running_var, 
                           torch::Tensor gamma, torch::Tensor beta, 
                           int64_t stride, int64_t padding);

torch::Tensor residual_add(torch::Tensor a, torch::Tensor b);
"""

# Compile the inline CUDA code
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=conv_bn_relu_cpp_source,
    cuda_sources=conv_bn_relu_source,
    functions=["fused_conv_bn_relu", "fused_conv_bn_relu6", "fused_conv_bn", "residual_add"],
    verbose=False,
)

class FusedConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(FusedConvBNReLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return custom_ops.fused_conv_bn_relu(
            x, self.conv.weight, self.conv.bias if self.conv.bias is not None else torch.tensor([]),
            self.bn.running_mean, self.bn.running_var, self.bn.weight, self.bn.bias,
            self.stride, self.padding
        )

class FusedConvBNReLU6(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(FusedConvBNReLU6, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return custom_ops.fused_conv_bn_relu6(
            x, self.conv.weight, self.conv.bias if self.conv.bias is not None else torch.tensor([]),
            self.bn.running_mean, self.bn.running_var, self.bn.weight, self.bn.bias,
            self.stride, self.padding
        )

class FusedConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(FusedConvBN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return custom_ops.fused_conv_bn(
            x, self.conv.weight, self.conv.bias if self.conv.bias is not None else torch.tensor([]),
            self.bn.running_mean, self.bn.running_var, self.bn.weight, self.bn.bias,
            self.stride, self.padding
        )

class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(FusedMBConv, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        if expand_ratio != 1:
            self.expand_conv = FusedConvBNReLU(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0)
        
        self.depthwise_conv = FusedConvBNReLU6(hidden_dim, hidden_dim, kernel_size=kernel_size, 
                                              stride=stride, padding=(kernel_size-1)//2)
        
        self.project_conv = FusedConvBN(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        identity = x
        
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        
        if self.use_residual:
            x = custom_ops.residual_add(x, identity)
        
        return x

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = FusedConvBNReLU(3, 32, kernel_size=3, stride=2, padding=1)
        
        # MBConv blocks
        self.blocks = nn.Sequential(
            # MBConv1 (32, 16, 1, 1)
            FusedMBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            # MBConv6 (16, 24, 2, 6)
            FusedMBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (24, 24, 1, 6)
            FusedMBConv(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (24, 40, 2, 6)
            FusedMBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (40, 40, 1, 6)
            FusedMBConv(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (40, 80, 2, 6)
            FusedMBConv(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (80, 80, 1, 6)
            FusedMBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (80, 112, 1, 6)
            FusedMBConv(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 112, 1, 6)
            FusedMBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 192, 2, 6)
            FusedMBConv(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            FusedMBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            FusedMBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 320, 1, 6)
            FusedMBConv(192, 320, kernel_size=3, stride=1, expand_ratio=6)
        )
        
        # Final convolutional layer
        self.conv2 = FusedConvBNReLU(320, 1280, kernel_size=1, stride=1, padding=0)
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x