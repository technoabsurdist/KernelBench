import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels
shufflenet_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Channel shuffle kernel
__global__ void channel_shuffle_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    int groups
) {
    int channels_per_group = channels / groups;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        int w = idx % width;
        int h = (idx / width) % height;
        int c = (idx / (width * height)) % channels;
        int n = idx / (width * height * channels);
        
        int group_idx = c / channels_per_group;
        int channel_idx = c % channels_per_group;
        
        int new_c = channel_idx * groups + group_idx;
        int new_idx = n * (channels * height * width) + new_c * (height * width) + h * width + w;
        
        output[new_idx] = input[idx];
    }
}

// Fused conv + batch norm + relu kernel
__global__ void conv_bn_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    float eps
) {
    int out_w = (width + 2 * padding - kernel_size) / stride + 1;
    int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * out_h * out_w;
    
    if (idx < total_outputs) {
        int w_out = idx % out_w;
        int h_out = (idx / out_w) % out_h;
        int c_out = (idx / (out_w * out_h)) % out_channels;
        int n = idx / (out_w * out_h * out_channels);
        
        float sum = 0.0f;
        
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        int input_idx = n * (in_channels * height * width) + 
                                       c_in * (height * width) + 
                                       h_in * width + w_in;
                        int weight_idx = c_out * (in_channels * kernel_size * kernel_size) + 
                                        c_in * (kernel_size * kernel_size) + 
                                        kh * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias
        sum += bias[c_out];
        
        // Batch norm
        float mean = running_mean[c_out];
        float var = running_var[c_out];
        float normalized = (sum - mean) / sqrtf(var + eps);
        
        // ReLU
        output[idx] = fmaxf(0.0f, normalized);
    }
}

// Element-wise addition kernel
__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

// Channel shuffle function
torch::Tensor channel_shuffle_cuda(torch::Tensor input, int groups) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto output = torch::zeros_like(input);
    
    const int block_size = 256;
    const int total_elements = batch_size * channels * height * width;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    channel_shuffle_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        groups
    );
    
    return output;
}

// Element-wise addition function
torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    elementwise_add_kernel<<<num_blocks, block_size>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    return out;
}

"""

shufflenet_cpp_source = """
#include <torch/extension.h>

torch::Tensor channel_shuffle_cuda(torch::Tensor input, int groups);
torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("channel_shuffle_cuda", &channel_shuffle_cuda, "Channel shuffle CUDA");
    m.def("elementwise_add_cuda", &elementwise_add_cuda, "Element-wise addition CUDA");
}
"""

# Compile the inline CUDA code
shufflenet_ops = load_inline(
    name="shufflenet_ops",
    cpp_sources=shufflenet_cpp_source,
    cuda_sources=shufflenet_cuda_source,
    functions=["channel_shuffle_cuda", "elementwise_add_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ChannelShuffleCustom(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffleCustom, self).__init__()
        self.groups = groups
        self.channel_shuffle_cuda = shufflenet_ops.channel_shuffle_cuda
    
    def forward(self, x):
        return self.channel_shuffle_cuda(x, self.groups)

class ShuffleNetUnitCustom(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitCustom, self).__init__()
        
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        # First 1x1 group convolution
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # Depthwise 3x3 convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # Second 1x1 group convolution
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Custom shuffle operation
        self.shuffle = ChannelShuffleCustom(groups)
        
        # Shortcut connection
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        self.elementwise_add_cuda = shufflenet_ops.elementwise_add_cuda
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shuffle(out)
        out = F.relu(self.bn3(self.conv3(out)))
        
        shortcut = self.shortcut(x)
        return self.elementwise_add_cuda(out, shortcut)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        super(ModelNew, self).__init__()
        
        self.conv1 = nn.Conv2d(3, stages_out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)
        
        self.conv5 = nn.Conv2d(stages_out_channels[3], 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)
        
        self.fc = nn.Linear(1024, num_classes)
        
        self.elementwise_add_cuda = shufflenet_ops.elementwise_add_cuda
    
    def _make_stage(self, in_channels, out_channels, repeats, groups):
        layers = []
        layers.append(ShuffleNetUnitCustom(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnitCustom(out_channels, out_channels, groups))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x