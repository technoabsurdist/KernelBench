import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Fused BatchNorm2D + ReLU kernel
fused_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void fused_bn_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int spatial_size,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * spatial_size;
    
    if (idx < total_size) {
        int c = (idx / spatial_size) % channels;
        
        float mean_val = running_mean[c];
        float var_val = running_var[c];
        float gamma = weight[c];
        float beta = bias[c];
        
        float x = input[idx];
        float x_normalized = (x - mean_val) / sqrtf(var_val + eps);
        float y = gamma * x_normalized + beta;
        
        // ReLU activation
        output[idx] = fmaxf(0.0f, y);
    }
}

torch::Tensor fused_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps) {
    
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto spatial_size = height * width;
    
    auto output = torch::zeros_like(input);
    
    int total_size = batch_size * channels * spatial_size;
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    fused_bn_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        spatial_size,
        eps
    );
    
    return output;
}
"""

fused_bn_relu_cpp_source = """
torch::Tensor fused_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps);
"""

# Fused adaptive average pooling + flatten kernel
fused_adaptive_avgpool_flatten_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_adaptive_avgpool_flatten_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int height,
    int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * channels) {
        int b = idx / channels;
        int c = idx % channels;
        
        float sum = 0.0f;
        int offset = b * channels * height * width + c * height * width;
        
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                sum += input[offset + h * width + w];
            }
        }
        
        output[idx] = sum / (height * width);
    }
}

torch::Tensor fused_adaptive_avgpool_flatten_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto output = torch::zeros({batch_size, channels}, input.options());
    
    int total_size = batch_size * channels;
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    fused_adaptive_avgpool_flatten_kernel<<<num_blocks, block_size>>>(
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

fused_adaptive_avgpool_flatten_cpp_source = """
torch::Tensor fused_adaptive_avgpool_flatten_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
fused_bn_relu = load_inline(
    name="fused_bn_relu",
    cpp_sources=fused_bn_relu_cpp_source,
    cuda_sources=fused_bn_relu_source,
    functions=["fused_bn_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

fused_adaptive_avgpool_flatten = load_inline(
    name="fused_adaptive_avgpool_flatten",
    cpp_sources=fused_adaptive_avgpool_flatten_cpp_source,
    cuda_sources=fused_adaptive_avgpool_flatten_source,
    functions=["fused_adaptive_avgpool_flatten_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class FusedBatchNormReLU(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(FusedBatchNormReLU, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x):
        if self.training:
            # Use standard PyTorch during training for proper statistics update
            x = F.batch_norm(x, self.running_mean, self.running_var, 
                           self.weight, self.bias, self.training, 
                           self.momentum, self.eps)
            return F.relu(x, inplace=True)
        else:
            # Use fused kernel during inference
            return fused_bn_relu.fused_bn_relu_cuda(
                x.contiguous(), 
                self.running_mean, 
                self.running_var,
                self.weight, 
                self.bias, 
                self.eps
            )

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            FusedBatchNormReLU(in_features),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer, self).__init__()
        self.bn_relu = FusedBatchNormReLU(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn_relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelNew, self).__init__()

        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_relu1 = FusedBatchNormReLU(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense blocks and transitions
        num_features = 64
        block_layers = [6, 12, 48, 32]

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        # Final layers
        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)
        self.fused_pool = fused_adaptive_avgpool_flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn_relu1(x)
        x = self.maxpool(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.final_bn(x)
        x = F.relu(x, inplace=True)
        
        # Use fused adaptive pooling + flatten
        x = self.fused_pool.fused_adaptive_avgpool_flatten_cuda(x.contiguous())
        
        x = self.classifier(x)
        return x

def get_inputs():
    batch_size = 10
    height, width = 224, 224
    return [torch.rand(batch_size, 3, height, width)]

def get_init_inputs():
    return [32, 10]