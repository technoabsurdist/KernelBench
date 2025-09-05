import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Fused Conv2D + BatchNorm + ReLU kernel
fused_conv_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cudnn.h>

__global__ void fused_conv_bn_relu_kernel(
    const float* input, const float* weight, const float* bias,
    const float* bn_weight, const float* bn_bias, 
    const float* running_mean, const float* running_var,
    float* output, 
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int stride, int padding,
    float eps) {
    
    int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_h * out_w;
    
    if (idx < total_elements) {
        int w_out = idx % out_w;
        int h_out = (idx / out_w) % out_h;
        int c_out = (idx / (out_w * out_h)) % out_channels;
        int b = idx / (out_channels * out_h * out_w);
        
        float sum = 0.0f;
        
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int h_in = h_out * stride - padding + kh;
                    int w_in = w_out * stride - padding + kw;
                    
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        int input_idx = b * (in_channels * height * width) + 
                                      c_in * (height * width) + h_in * width + w_in;
                        int weight_idx = c_out * (in_channels * kernel_size * kernel_size) +
                                       c_in * (kernel_size * kernel_size) + kh * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[c_out];
        }
        
        // BatchNorm
        float mean = running_mean[c_out];
        float var = running_var[c_out];
        float scale = bn_weight[c_out];
        float shift = bn_bias[c_out];
        
        sum = scale * (sum - mean) / sqrtf(var + eps) + shift;
        
        // ReLU
        output[idx] = fmaxf(0.0f, sum);
    }
}

torch::Tensor fused_conv_bn_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    int stride, int padding, float eps) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * out_h * out_w + block_size - 1) / block_size;
    
    fused_conv_bn_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding, eps
    );
    
    return output;
}
"""

fused_conv_bn_relu_cpp_source = """
torch::Tensor fused_conv_bn_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    int stride, int padding, float eps);
"""

fused_conv_bn_relu = load_inline(
    name="fused_conv_bn_relu",
    cpp_sources=fused_conv_bn_relu_cpp_source,
    cuda_sources=fused_conv_bn_relu_source,
    functions=["fused_conv_bn_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Fused Depthwise Conv + BatchNorm + ReLU kernel
fused_dwconv_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_dwconv_bn_relu_kernel(
    const float* input, const float* weight, const float* bias,
    const float* bn_weight, const float* bn_bias,
    const float* running_mean, const float* running_var,
    float* output,
    int batch_size, int channels, int height, int width,
    int kernel_size, int stride, int padding, float eps) {
    
    int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_h * out_w;
    
    if (idx < total_elements) {
        int w_out = idx % out_w;
        int h_out = (idx / out_w) % out_h;
        int c = (idx / (out_w * out_h)) % channels;
        int b = idx / (channels * out_h * out_w);
        
        float sum = 0.0f;
        
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = b * (channels * height * width) + 
                                  c * (height * width) + h_in * width + w_in;
                    int weight_idx = c * (kernel_size * kernel_size) + kh * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[c];
        }
        
        // BatchNorm
        float mean = running_mean[c];
        float var = running_var[c];
        float scale = bn_weight[c];
        float shift = bn_bias[c];
        
        sum = scale * (sum - mean) / sqrtf(var + eps) + shift;
        
        // ReLU
        output[idx] = fmaxf(0.0f, sum);
    }
}

torch::Tensor fused_dwconv_bn_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    int stride, int padding, float eps) {
    
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto kernel_size = weight.size(2);
    
    int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, channels, out_h, out_w}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * channels * out_h * out_w + block_size - 1) / block_size;
    
    fused_dwconv_bn_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width,
        kernel_size, stride, padding, eps
    );
    
    return output;
}
"""

fused_dwconv_bn_relu_cpp_source = """
torch::Tensor fused_dwconv_bn_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    int stride, int padding, float eps);
"""

fused_dwconv_bn_relu = load_inline(
    name="fused_dwconv_bn_relu",
    cpp_sources=fused_dwconv_bn_relu_cpp_source,
    cuda_sources=fused_dwconv_bn_relu_source,
    functions=["fused_dwconv_bn_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # Define the EfficientNetB2 architecture components
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Define the MBConv blocks
        self.mbconv1 = self._make_mbconv_block(32, 96, 1, 3)
        self.mbconv2 = self._make_mbconv_block(96, 144, 2, 6)
        self.mbconv3 = self._make_mbconv_block(144, 192, 2, 6)
        self.mbconv4 = self._make_mbconv_block(192, 288, 2, 6)
        self.mbconv5 = self._make_mbconv_block(288, 384, 1, 6)
        
        # Final layers
        self.conv_final = nn.Conv2d(384, 1408, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_final = nn.BatchNorm2d(1408)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1408, num_classes)
        
        # Custom kernels
        self.fused_conv_bn_relu = fused_conv_bn_relu
        self.fused_dwconv_bn_relu = fused_dwconv_bn_relu
    
    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        layers = []
        expanded_channels = in_channels * expand_ratio
        
        # Expansion phase
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(expanded_channels))
            layers.append(nn.ReLU(inplace=True))
        
        # Depthwise convolution
        layers.append(nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False))
        layers.append(nn.BatchNorm2d(expanded_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Squeeze and Excitation
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Conv2d(expanded_channels, expanded_channels // 4, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(expanded_channels // 4, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.Sigmoid())
        
        # Output phase
        layers.append(nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Use fused kernel for first conv + bn + relu
        x = self.fused_conv_bn_relu.fused_conv_bn_relu_cuda(
            x.contiguous(), self.conv1.weight, torch.zeros(32).cuda() if not self.conv1.bias else self.conv1.bias,
            self.bn1.weight, self.bn1.bias, self.bn1.running_mean, self.bn1.running_var,
            2, 1, 1e-5
        )
        
        # Process MBConv blocks
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        
        # Final layers with fused kernel
        x = self.fused_conv_bn_relu.fused_conv_bn_relu_cuda(
            x.contiguous(), self.conv_final.weight, torch.zeros(1408).cuda() if not self.conv_final.bias else self.conv_final.bias,
            self.bn_final.weight, self.bn_final.bias, self.bn_final.running_mean, self.bn_final.running_var,
            1, 0, 1e-5
        )
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def get_inputs():
    return [torch.rand(2, 3, 224, 224).cuda()]

def get_init_inputs():
    return [1000]