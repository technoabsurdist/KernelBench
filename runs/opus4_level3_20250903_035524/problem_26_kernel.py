import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for channel shuffle
channel_shuffle_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void channel_shuffle_kernel(
    const float* input, 
    float* output, 
    int batch_size, 
    int channels, 
    int height, 
    int width, 
    int groups) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        int w = idx % width;
        int h = (idx / width) % height;
        int c = (idx / (width * height)) % channels;
        int b = idx / (width * height * channels);
        
        int channels_per_group = channels / groups;
        int group_id = c / channels_per_group;
        int channel_in_group = c % channels_per_group;
        
        int new_c = channel_in_group * groups + group_id;
        int new_idx = b * channels * height * width + new_c * height * width + h * width + w;
        
        output[new_idx] = input[idx];
    }
}

torch::Tensor channel_shuffle_cuda(torch::Tensor x, int groups) {
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    
    auto output = torch::empty_like(x);
    
    int total_elements = batch_size * channels * height * width;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    channel_shuffle_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width, groups
    );
    
    return output;
}
"""

channel_shuffle_cpp_source = "torch::Tensor channel_shuffle_cuda(torch::Tensor x, int groups);"

# Custom CUDA kernel for fused conv-bn-relu
fused_conv_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cudnn.h>

__global__ void fused_bn_relu_kernel(
    float* output,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float eps,
    int batch_size,
    int channels,
    int spatial_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * spatial_size;
    
    if (idx < total) {
        int c = (idx / spatial_size) % channels;
        
        float mean = running_mean[c];
        float var = running_var[c];
        float scale = gamma[c];
        float bias = beta[c];
        
        float value = output[idx];
        value = (value - mean) / sqrtf(var + eps);
        value = value * scale + bias;
        
        // ReLU
        output[idx] = fmaxf(value, 0.0f);
    }
}

torch::Tensor fused_conv_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps,
    int stride,
    int padding,
    int groups) {
    
    // Perform convolution using cudnn
    auto output = torch::conv2d(input, weight, bias, {stride, stride}, {padding, padding}, {1, 1}, groups);
    
    // Apply fused BN+ReLU
    int batch_size = output.size(0);
    int channels = output.size(1);
    int height = output.size(2);
    int width = output.size(3);
    int spatial_size = height * width;
    
    int total = batch_size * channels * spatial_size;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    
    fused_bn_relu_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        batch_size,
        channels,
        spatial_size
    );
    
    return output;
}
"""

fused_conv_bn_relu_cpp_source = """
torch::Tensor fused_conv_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps,
    int stride,
    int padding,
    int groups);
"""

# Compile inline CUDA code
channel_shuffle_module = load_inline(
    name="channel_shuffle",
    cpp_sources=channel_shuffle_cpp_source,
    cuda_sources=channel_shuffle_source,
    functions=["channel_shuffle_cuda"],
    verbose=True,
)

fused_conv_bn_relu_module = load_inline(
    name="fused_conv_bn_relu",
    cpp_sources=fused_conv_bn_relu_cpp_source,
    cuda_sources=fused_conv_bn_relu_source,
    functions=["fused_conv_bn_relu_cuda"],
    verbose=True,
)

class ChannelShuffleOptimized(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffleOptimized, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        return channel_shuffle_module.channel_shuffle_cuda(x.contiguous(), self.groups)

class FusedConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(FusedConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.padding = padding
        self.groups = groups
    
    def forward(self, x):
        return fused_conv_bn_relu_module.fused_conv_bn_relu_cuda(
            x.contiguous(),
            self.conv.weight,
            torch.zeros(self.conv.weight.size(0), device=x.device, dtype=x.dtype),
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.weight,
            self.bn.bias,
            self.bn.eps,
            self.stride,
            self.padding,
            self.groups
        )

class ShuffleNetUnitNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitNew, self).__init__()
        
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        # First 1x1 group convolution with fused bn+relu
        self.conv1_fused = FusedConvBnRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups)
        
        # Depthwise 3x3 convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # Optimized channel shuffle
        self.shuffle = ChannelShuffleOptimized(groups)
        
        # Second 1x1 group convolution with fused bn+relu
        self.conv3_fused = FusedConvBnRelu(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups)
        
        # Shortcut connection
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.conv1_fused(x)
        out = self.bn2(self.conv2(out))
        out = self.shuffle(out)
        out = self.conv3_fused(out)
        
        out += self.shortcut(x)
        return out

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        super(ModelNew, self).__init__()
        
        self.conv1_fused = FusedConvBnRelu(3, stages_out_channels[0], kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)
        
        self.conv5_fused = FusedConvBnRelu(stages_out_channels[3], 1024, kernel_size=1, stride=1, padding=0)
        
        self.fc = nn.Linear(1024, num_classes)
    
    def _make_stage(self, in_channels, out_channels, repeats, groups):
        layers = []
        layers.append(ShuffleNetUnitNew(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnitNew(out_channels, out_channels, groups))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1_fused(x)
        x = self.maxpool(x)
        
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.conv5_fused(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def get_inputs():
    batch_size = 10
    input_channels = 3
    height = 224
    width = 224
    return [torch.rand(batch_size, input_channels, height, width)]

def get_init_inputs():
    num_classes = 1000
    return [num_classes]