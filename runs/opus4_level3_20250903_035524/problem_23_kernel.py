import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Conv2d + BatchNorm + ReLU6 fusion
conv_bn_relu6_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<typename scalar_t>
__global__ void conv_bn_relu6_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bn_weight,
    const scalar_t* __restrict__ bn_bias,
    const scalar_t* __restrict__ bn_mean,
    const scalar_t* __restrict__ bn_var,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int pad_h, int pad_w, float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_size * out_channels * out_height * out_width;
    
    if (idx < total_output) {
        int w_out = idx % out_width;
        int h_out = (idx / out_width) % out_height;
        int c_out = (idx / (out_width * out_height)) % out_channels;
        int b = idx / (out_width * out_height * out_channels);
        
        scalar_t sum = 0;
        
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int h_in = h_out * stride_h - pad_h + kh;
                    int w_in = w_out * stride_w - pad_w + kw;
                    
                    if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                        int input_idx = b * in_channels * in_height * in_width +
                                       c_in * in_height * in_width +
                                       h_in * in_width + w_in;
                        int weight_idx = c_out * in_channels * kernel_h * kernel_w +
                                        c_in * kernel_h * kernel_w +
                                        kh * kernel_w + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Apply batch normalization
        scalar_t bn_scale = bn_weight[c_out] / sqrt(bn_var[c_out] + eps);
        scalar_t bn_shift = bn_bias[c_out] - bn_mean[c_out] * bn_scale;
        sum = sum * bn_scale + bn_shift;
        
        // Apply ReLU6
        output[idx] = fminf(fmaxf(sum, (scalar_t)0), (scalar_t)6);
    }
}

torch::Tensor conv_bn_relu6_cuda(
    torch::Tensor input, torch::Tensor weight,
    torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor bn_mean, torch::Tensor bn_var,
    int stride_h, int stride_w, int pad_h, int pad_w, float eps) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);
    
    int out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    int total_output = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_output + block_size - 1) / block_size;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_bn_relu6_cuda", ([&] {
        conv_bn_relu6_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bn_weight.data_ptr<scalar_t>(),
            bn_bias.data_ptr<scalar_t>(),
            bn_mean.data_ptr<scalar_t>(),
            bn_var.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            in_height, in_width, out_height, out_width,
            kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w, eps
        );
    }));
    
    return output;
}
"""

conv_bn_relu6_cpp_source = """
torch::Tensor conv_bn_relu6_cuda(
    torch::Tensor input, torch::Tensor weight,
    torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor bn_mean, torch::Tensor bn_var,
    int stride_h, int stride_w, int pad_h, int pad_w, float eps);
"""

# Custom CUDA kernel for depthwise Conv2d + BatchNorm + ReLU6 fusion
depthwise_conv_bn_relu6_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void depthwise_conv_bn_relu6_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bn_weight,
    const scalar_t* __restrict__ bn_bias,
    const scalar_t* __restrict__ bn_mean,
    const scalar_t* __restrict__ bn_var,
    scalar_t* __restrict__ output,
    int batch_size, int channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int pad_h, int pad_w, float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_size * channels * out_height * out_width;
    
    if (idx < total_output) {
        int w_out = idx % out_width;
        int h_out = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % channels;
        int b = idx / (out_width * out_height * channels);
        
        scalar_t sum = 0;
        
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride_h - pad_h + kh;
                int w_in = w_out * stride_w - pad_w + kw;
                
                if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                    int input_idx = b * channels * in_height * in_width +
                                   c * in_height * in_width +
                                   h_in * in_width + w_in;
                    int weight_idx = c * kernel_h * kernel_w + kh * kernel_w + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
        
        // Apply batch normalization
        scalar_t bn_scale = bn_weight[c] / sqrt(bn_var[c] + eps);
        scalar_t bn_shift = bn_bias[c] - bn_mean[c] * bn_scale;
        sum = sum * bn_scale + bn_shift;
        
        // Apply ReLU6
        output[idx] = fminf(fmaxf(sum, (scalar_t)0), (scalar_t)6);
    }
}

torch::Tensor depthwise_conv_bn_relu6_cuda(
    torch::Tensor input, torch::Tensor weight,
    torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor bn_mean, torch::Tensor bn_var,
    int stride_h, int stride_w, int pad_h, int pad_w, float eps) {
    
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);
    
    int out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    auto output = torch::zeros({batch_size, channels, out_height, out_width}, input.options());
    
    int total_output = batch_size * channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_output + block_size - 1) / block_size;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "depthwise_conv_bn_relu6_cuda", ([&] {
        depthwise_conv_bn_relu6_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bn_weight.data_ptr<scalar_t>(),
            bn_bias.data_ptr<scalar_t>(),
            bn_mean.data_ptr<scalar_t>(),
            bn_var.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, channels,
            in_height, in_width, out_height, out_width,
            kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w, eps
        );
    }));
    
    return output;
}
"""

depthwise_conv_bn_relu6_cpp_source = """
torch::Tensor depthwise_conv_bn_relu6_cuda(
    torch::Tensor input, torch::Tensor weight,
    torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor bn_mean, torch::Tensor bn_var,
    int stride_h, int stride_w, int pad_h, int pad_w, float eps);
"""

# Compile the inline CUDA code
conv_bn_relu6_module = load_inline(
    name="conv_bn_relu6",
    cpp_sources=conv_bn_relu6_cpp_source,
    cuda_sources=conv_bn_relu6_source,
    functions=["conv_bn_relu6_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

depthwise_conv_bn_relu6_module = load_inline(
    name="depthwise_conv_bn_relu6",
    cpp_sources=depthwise_conv_bn_relu6_cpp_source,
    cuda_sources=depthwise_conv_bn_relu6_source,
    functions=["depthwise_conv_bn_relu6_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class OptimizedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = round(in_channels * expand_ratio)
        
        # Expand
        self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.expand_bn = nn.BatchNorm2d(hidden_dim)
        
        # Depthwise
        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)
        
        # Project
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        self.stride = stride
    
    def forward(self, x):
        # Expand with fused conv+bn+relu6
        x = conv_bn_relu6_module.conv_bn_relu6_cuda(
            x, self.expand_conv.weight,
            self.expand_bn.weight, self.expand_bn.bias,
            self.expand_bn.running_mean, self.expand_bn.running_var,
            1, 1, 0, 0, self.expand_bn.eps
        )
        
        # Depthwise with fused conv+bn+relu6
        x = depthwise_conv_bn_relu6_module.depthwise_conv_bn_relu6_cuda(
            x, self.depthwise_conv.weight,
            self.depthwise_bn.weight, self.depthwise_bn.bias,
            self.depthwise_bn.running_mean, self.depthwise_bn.running_var,
            self.stride, self.stride, 1, 1, self.depthwise_bn.eps
        )
        
        # Project (no activation)
        x = F.conv2d(x, self.project_conv.weight, stride=1, padding=0)
        x = self.project_bn(x)
        
        return x

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Optimized MBConv blocks
        self.mbconv1 = OptimizedMBConv(32, 16, 1, 1)
        self.mbconv2 = OptimizedMBConv(16, 24, 2, 6)
        self.mbconv3 = OptimizedMBConv(24, 40, 2, 6)
        self.mbconv4 = OptimizedMBConv(40, 80, 2, 6)
        self.mbconv5 = OptimizedMBConv(80, 112, 1, 6)
        self.mbconv6 = OptimizedMBConv(112, 192, 2, 6)
        self.mbconv7 = OptimizedMBConv(192, 320, 1, 6)
        
        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def get_inputs():
    batch_size = 10
    input_shape = (3, 240, 240)
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return [1000]