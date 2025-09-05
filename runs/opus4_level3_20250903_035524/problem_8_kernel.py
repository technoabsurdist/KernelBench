import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Fused BatchNorm + ReLU kernel
bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bn_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int HW,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * HW;
    
    if (idx < total) {
        int c = (idx / HW) % C;
        
        float x = input[idx];
        float m = mean[c];
        float v = var[c];
        float w = weight[c];
        float b = bias[c];
        
        float y = (x - m) / sqrtf(v + eps) * w + b;
        output[idx] = fmaxf(0.0f, y);  // ReLU
    }
}

torch::Tensor bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps) {
    
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    auto HW = H * W;
    
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (N * C * HW + threads - 1) / threads;
    
    bn_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, HW, eps
    );
    
    return output;
}
"""

bn_relu_cpp_source = """
torch::Tensor bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps);
"""

# Fused residual add + ReLU kernel
add_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_relu_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = fmaxf(0.0f, a[idx] + b[idx]);
    }
}

torch::Tensor add_relu_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto output = torch::empty_like(a);
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    add_relu_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

add_relu_cpp_source = """
torch::Tensor add_relu_cuda(torch::Tensor a, torch::Tensor b);
"""

# Load the custom CUDA kernels
bn_relu_module = load_inline(
    name="bn_relu",
    cpp_sources=bn_relu_cpp_source,
    cuda_sources=bn_relu_source,
    functions=["bn_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

add_relu_module = load_inline(
    name="add_relu",
    cpp_sources=add_relu_cpp_source,
    cuda_sources=add_relu_source,
    functions=["add_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride
        self.bn_relu = bn_relu_module
        self.add_relu = add_relu_module

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # Fused BN + ReLU
        out = self.bn_relu.bn_relu_cuda(
            out,
            self.bn1.running_mean,
            self.bn1.running_var,
            self.bn1.weight,
            self.bn1.bias,
            self.bn1.eps
        )

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Fused add + ReLU
        out = self.add_relu.add_relu_cuda(out, identity)

        return out

in_channels = 3
out_channels = 64
stride = 1
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, in_channels, 224, 224).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, stride]