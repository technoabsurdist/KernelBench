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
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float* __restrict__ output,
    int N, int C, int HW, float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * HW;
    
    if (idx < total) {
        int c = (idx / HW) % C;
        
        float x = input[idx];
        float m = mean[c];
        float v = var[c];
        float g = gamma[c];
        float b = beta[c];
        
        float norm = (x - m) / sqrtf(v + eps);
        float bn_out = g * norm + b;
        output[idx] = fmaxf(bn_out, 0.0f);  // ReLU
    }
}

torch::Tensor bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
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
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, HW, eps
    );
    
    return output;
}
"""

bn_relu_cpp_source = """
torch::Tensor bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
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
        output[idx] = fmaxf(a[idx] + b[idx], 0.0f);
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

add_relu_cpp_source = "torch::Tensor add_relu_cuda(torch::Tensor a, torch::Tensor b);"

# Custom adaptive average pooling kernel
adaptive_avgpool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void adaptive_avgpool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    
    if (idx < total) {
        int n = idx / C;
        int c = idx % C;
        
        float sum = 0.0f;
        int offset = n * C * H * W + c * H * W;
        
        for (int i = 0; i < H * W; i++) {
            sum += input[offset + i];
        }
        
        output[idx] = sum / (H * W);
    }
}

torch::Tensor adaptive_avgpool_cuda(torch::Tensor input) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    
    auto output = torch::zeros({N, C, 1, 1}, input.options());
    
    const int threads = 256;
    const int blocks = (N * C + threads - 1) / threads;
    
    adaptive_avgpool_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W
    );
    
    return output;
}
"""

adaptive_avgpool_cpp_source = "torch::Tensor adaptive_avgpool_cuda(torch::Tensor input);"

# Load custom kernels
bn_relu_module = load_inline(
    name="bn_relu",
    cpp_sources=bn_relu_cpp_source,
    cuda_sources=bn_relu_source,
    functions=["bn_relu_cuda"],
    verbose=False,
)

add_relu_module = load_inline(
    name="add_relu",
    cpp_sources=add_relu_cpp_source,
    cuda_sources=add_relu_source,
    functions=["add_relu_cuda"],
    verbose=False,
)

adaptive_avgpool_module = load_inline(
    name="adaptive_avgpool",
    cpp_sources=adaptive_avgpool_cpp_source,
    cuda_sources=adaptive_avgpool_source,
    functions=["adaptive_avgpool_cuda"],
    verbose=False,
)

class FusedBNReLU(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.eps = 1e-5
        self.momentum = 0.1
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x):
        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            return bn_relu_module.bn_relu_cuda(x, self.weight, self.bias, mean, var, self.eps)
        else:
            return bn_relu_module.bn_relu_cuda(x, self.weight, self.bias, self.running_mean, self.running_var, self.eps)

class BottleneckNew(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1_relu = FusedBNReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2_relu = FusedBNReLU(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1_relu(out)

        out = self.conv2(out)
        out = self.bn2_relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = add_relu_module.add_relu_cuda(out, identity)

        return out

class ModelNew(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_relu = FusedBNReLU(self.in_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = BottleneckNew

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1_relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = adaptive_avgpool_module.adaptive_avgpool_cuda(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def get_inputs():
    return [torch.rand(10, 3, 224, 224)]

def get_init_inputs():
    return [[3, 4, 23, 3], 1000]