import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused 1x1 convolution + ReLU
conv1x1_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1x1_relu_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int height, int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * height * width;
    
    if (idx < total_output_elements) {
        int w = idx % width;
        int h = (idx / width) % height;
        int oc = (idx / (width * height)) % out_channels;
        int b = idx / (width * height * out_channels);
        
        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
            int input_idx = b * in_channels * height * width + ic * height * width + h * width + w;
            int weight_idx = oc * in_channels + ic;
            sum += input[input_idx] * weight[weight_idx];
        }
        
        if (bias != nullptr) {
            sum += bias[oc];
        }
        
        output[idx] = fmaxf(sum, 0.0f); // ReLU activation
    }
}

torch::Tensor conv1x1_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());
    
    int total_elements = batch_size * out_channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv1x1_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, height, width
    );
    
    return output;
}
"""

conv1x1_relu_cpp_source = "torch::Tensor conv1x1_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Custom CUDA kernel for optimized concatenation
concat_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void concat_kernel(
    const float* input1, const float* input2,
    float* output, int batch_size, int channels1, int channels2,
    int height, int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * (channels1 + channels2) * height * width;
    
    if (idx < total_elements) {
        int w = idx % width;
        int h = (idx / width) % height;
        int c = (idx / (width * height)) % (channels1 + channels2);
        int b = idx / (width * height * (channels1 + channels2));
        
        if (c < channels1) {
            int input1_idx = b * channels1 * height * width + c * height * width + h * width + w;
            output[idx] = input1[input1_idx];
        } else {
            int c2 = c - channels1;
            int input2_idx = b * channels2 * height * width + c2 * height * width + h * width + w;
            output[idx] = input2[input2_idx];
        }
    }
}

torch::Tensor concat_cuda(torch::Tensor input1, torch::Tensor input2) {
    auto batch_size = input1.size(0);
    auto channels1 = input1.size(1);
    auto channels2 = input2.size(1);
    auto height = input1.size(2);
    auto width = input1.size(3);
    
    auto output = torch::zeros({batch_size, channels1 + channels2, height, width}, input1.options());
    
    int total_elements = batch_size * (channels1 + channels2) * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    concat_kernel<<<num_blocks, block_size>>>(
        input1.data_ptr<float>(), input2.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels1, channels2, height, width
    );
    
    return output;
}
"""

concat_cpp_source = "torch::Tensor concat_cuda(torch::Tensor input1, torch::Tensor input2);"

# Load custom CUDA kernels
conv1x1_relu_module = load_inline(
    name="conv1x1_relu",
    cpp_sources=conv1x1_relu_cpp_source,
    cuda_sources=conv1x1_relu_source,
    functions=["conv1x1_relu_cuda"],
    verbose=False,
)

concat_module = load_inline(
    name="concat",
    cpp_sources=concat_cpp_source,
    cuda_sources=concat_source,
    functions=["concat_cuda"],
    verbose=False,
)

class FireModuleOptimized(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModuleOptimized, self).__init__()
        
        self.squeeze_weight = nn.Parameter(torch.randn(squeeze_channels, in_channels))
        self.squeeze_bias = nn.Parameter(torch.zeros(squeeze_channels))
        
        self.expand1x1_weight = nn.Parameter(torch.randn(expand1x1_channels, squeeze_channels))
        self.expand1x1_bias = nn.Parameter(torch.zeros(expand1x1_channels))
        
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.squeeze_weight)
        nn.init.kaiming_normal_(self.expand1x1_weight)
        nn.init.constant_(self.squeeze_bias, 0)
        nn.init.constant_(self.expand1x1_bias, 0)
    
    def forward(self, x):
        # Use custom CUDA kernel for squeeze (1x1 conv + relu)
        x = conv1x1_relu_module.conv1x1_relu_cuda(x, self.squeeze_weight, self.squeeze_bias)
        
        # Use custom CUDA kernel for expand1x1 (1x1 conv + relu)
        expand1x1_out = conv1x1_relu_module.conv1x1_relu_cuda(x, self.expand1x1_weight, self.expand1x1_bias)
        
        # Regular Conv3x3 + ReLU for expand3x3
        expand3x3_out = self.expand3x3_activation(self.expand3x3(x))
        
        # Use custom CUDA kernel for concatenation
        return concat_module.concat_cuda(expand1x1_out, expand3x3_out)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModuleOptimized(96, 16, 64, 64),
            FireModuleOptimized(128, 16, 64, 64),
            FireModuleOptimized(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModuleOptimized(256, 32, 128, 128),
            FireModuleOptimized(256, 48, 192, 192),
            FireModuleOptimized(384, 48, 192, 192),
            FireModuleOptimized(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModuleOptimized(512, 64, 256, 256),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

def get_inputs():
    batch_size = 64
    input_channels = 3
    height = 512
    width = 512
    return [torch.rand(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [1000]