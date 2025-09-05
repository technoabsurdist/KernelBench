import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Fused BatchNorm + ReLU kernel
fused_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void fused_bn_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int spatial_size,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * spatial_size;
    
    if (idx < total_size) {
        int c = (idx / spatial_size) % channels;
        
        float mean = running_mean[c];
        float var = running_var[c];
        float w = weight[c];
        float b = bias[c];
        
        float x = input[idx];
        float norm = (x - mean) / sqrtf(var + eps);
        float bn_out = norm * w + b;
        output[idx] = fmaxf(bn_out, 0.0f);  // ReLU
    }
}

torch::Tensor fused_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps) {
    
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto spatial_size = height * width;
    
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (batch_size * channels * spatial_size + block_size - 1) / block_size;
    
    fused_bn_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        spatial_size,
        eps
    );
    
    return output;
}
"""

fused_bn_relu_cpp_source = "torch::Tensor fused_bn_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor running_mean, torch::Tensor running_var, float eps);"

# Fused residual add + ReLU kernel
fused_add_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_add_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    float* __restrict__ output,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = x[idx] + residual[idx];
        output[idx] = fmaxf(sum, 0.0f);
    }
}

torch::Tensor fused_add_relu_cuda(torch::Tensor x, torch::Tensor residual) {
    auto size = x.numel();
    auto output = torch::empty_like(x);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_add_relu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        residual.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

fused_add_relu_cpp_source = "torch::Tensor fused_add_relu_cuda(torch::Tensor x, torch::Tensor residual);"

# Optimized adaptive average pooling for (1,1) output
adaptive_avgpool_1x1_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void adaptive_avgpool_1x1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int height,
    int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_channels = batch_size * channels;
    
    if (idx < total_channels) {
        int spatial_size = height * width;
        int offset = idx * spatial_size;
        
        float sum = 0.0f;
        for (int i = 0; i < spatial_size; i++) {
            sum += input[offset + i];
        }
        output[idx] = sum / spatial_size;
    }
}

torch::Tensor adaptive_avgpool_1x1_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto output = torch::zeros({batch_size, channels, 1, 1}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * channels + block_size - 1) / block_size;
    
    adaptive_avgpool_1x1_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width
    );
    
    return output;
}
"""

adaptive_avgpool_1x1_cpp_source = "torch::Tensor adaptive_avgpool_1x1_cuda(torch::Tensor input);"

# Compile the inline CUDA code
fused_bn_relu = load_inline(
    name="fused_bn_relu",
    cpp_sources=fused_bn_relu_cpp_source,
    cuda_sources=fused_bn_relu_source,
    functions=["fused_bn_relu_cuda"],
    verbose=False,
)

fused_add_relu = load_inline(
    name="fused_add_relu",
    cpp_sources=fused_add_relu_cpp_source,
    cuda_sources=fused_add_relu_source,
    functions=["fused_add_relu_cuda"],
    verbose=False,
)

adaptive_avgpool_1x1 = load_inline(
    name="adaptive_avgpool_1x1",
    cpp_sources=adaptive_avgpool_1x1_cpp_source,
    cuda_sources=adaptive_avgpool_1x1_source,
    functions=["adaptive_avgpool_1x1_cuda"],
    verbose=False,
)


class OptimizedBatchNormReLU(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = 1e-5
        self.momentum = 0.1
        
    def forward(self, x):
        if self.training:
            # Use standard PyTorch during training for proper statistics update
            bn_out = F.batch_norm(x, self.running_mean, self.running_var, 
                                 self.weight, self.bias, self.training, 
                                 self.momentum, self.eps)
            return F.relu(bn_out, inplace=True)
        else:
            return fused_bn_relu.fused_bn_relu_cuda(
                x, self.weight, self.bias, self.running_mean, self.running_var, self.eps
            )


class BasicBlockNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlockNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = fused_add_relu.fused_add_relu_cuda(out, identity)
        return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = OptimizedBatchNormReLU(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlockNew, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlockNew, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlockNew, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlockNew, 512, 2, stride=2)

        self.fc = nn.Linear(512 * BasicBlockNew.expansion, num_classes)

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
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = adaptive_avgpool_1x1.adaptive_avgpool_1x1_cuda(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def get_inputs():
    return [torch.rand(2, 3, 224, 224).cuda()]


def get_init_inputs():
    return [1000]