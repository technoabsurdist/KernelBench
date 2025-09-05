import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv + relu
fused_conv_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

torch::Tensor fused_conv_relu_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding) {
    // Perform convolution using PyTorch's built-in function
    auto conv_output = torch::conv2d(input, weight, bias, {stride, stride}, {padding, padding});
    
    // Apply ReLU activation in-place
    auto size = conv_output.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    relu_kernel<<<num_blocks, block_size>>>(conv_output.data_ptr<float>(), size);
    
    return conv_output;
}
"""

fused_conv_relu_cpp_source = """
torch::Tensor fused_conv_relu_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding);
"""

# Compile the inline CUDA code for fused conv + relu
fused_conv_relu = load_inline(
    name="fused_conv_relu",
    cpp_sources=fused_conv_relu_cpp_source,
    cuda_sources=fused_conv_relu_source,
    functions=["fused_conv_relu_forward"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class FusedConvReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        return fused_conv_relu.fused_conv_relu_forward(input, weight, bias, stride, padding)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        
        # Use PyTorch's built-in conv2d backward for gradient computation
        grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride=(stride, stride), padding=(padding, padding))
        grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=(stride, stride), padding=(padding, padding))
        
        if bias is not None:
            grad_bias = grad_output.sum((0, 2, 3))
        else:
            grad_bias = None
            
        return grad_input, grad_weight, grad_bias, None, None

class FusedConvReLU2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(FusedConvReLU2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        
    def forward(self, x):
        return FusedConvReLU.apply(x, self.weight, self.bias, self.stride, self.padding)

class FireModuleNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModuleNew, self).__init__()
        
        self.squeeze = FusedConvReLU2d(in_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = FusedConvReLU2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = FusedConvReLU2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat([
            self.expand1x1(x),
            self.expand3x3(x)
        ], 1)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        self.features = nn.Sequential(
            FusedConvReLU2d(3, 96, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModuleNew(96, 16, 64, 64),
            FireModuleNew(128, 16, 64, 64),
            FireModuleNew(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModuleNew(256, 32, 128, 128),
            FireModuleNew(256, 48, 192, 192),
            FireModuleNew(384, 48, 192, 192),
            FireModuleNew(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModuleNew(512, 64, 256, 256),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            FusedConvReLU2d(512, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)