import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Conv2d + BatchNorm2d + ReLU6
conv_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void conv_bn_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* bn_weight,
    const float* bn_bias,
    float* output,
    int batch,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int out_height,
    int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * out_channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % out_channels;
    int b = idx / (out_width * out_height * out_channels);
    
    float sum = 0.0f;
    
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_y = h * stride - padding + ky;
                int in_x = w * stride - padding + kx;
                
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    int input_idx = ((b * in_channels + ic) * in_height + in_y) * in_width + in_x;
                    int weight_idx = ((c * in_channels + ic) * kernel_size + ky) * kernel_size + kx;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    sum += bias[c];
    
    // Batch norm
    sum = bn_weight[c] * sum + bn_bias[c];
    
    // ReLU6
    sum = fmaxf(0.0f, fminf(6.0f, sum));
    
    output[idx] = sum;
}

torch::Tensor conv_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    int stride,
    int padding
) {
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch = input_sizes[0];
    int in_channels = input_sizes[1];
    int in_height = input_sizes[2];
    int in_width = input_sizes[3];
    
    int out_channels = weight_sizes[0];
    int kernel_size = weight_sizes[2];
    
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::zeros({batch, out_channels, out_height, out_width}, options);
    
    const int block_size = 256;
    int total_elements = batch * out_channels * out_height * out_width;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv_bn_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_size,
        stride,
        padding,
        out_height,
        out_width
    );
    
    return output;
}
"""

conv_bn_relu_cpp_source = """
torch::Tensor conv_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    int stride,
    int padding
);
"""

# Compile the inline CUDA code
conv_bn_relu = load_inline(
    name="conv_bn_relu",
    cpp_sources=conv_bn_relu_cpp_source,
    cuda_sources=conv_bn_relu_source,
    functions=["conv_bn_relu_cuda"],
    verbose=False,
)

# Custom CUDA kernel for depthwise convolution + batch norm + ReLU6
depthwise_conv_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv_bn_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* bn_weight,
    const float* bn_bias,
    float* output,
    int batch,
    int channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int out_height,
    int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_width * out_height);
    
    float sum = 0.0f;
    
    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            int in_y = h * stride - padding + ky;
            int in_x = w * stride - padding + kx;
            
            if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                int input_idx = ((b * channels + c) * in_height + in_y) * in_width + in_x;
                int weight_idx = (c * kernel_size + ky) * kernel_size + kx;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    // Add bias
    sum += bias[c];
    
    // Batch norm
    sum = bn_weight[c] * sum + bn_bias[c];
    
    // ReLU6
    sum = fmaxf(0.0f, fminf(6.0f, sum));
    
    output[idx] = sum;
}

torch::Tensor depthwise_conv_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    int stride,
    int padding
) {
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch = input_sizes[0];
    int channels = input_sizes[1];
    int in_height = input_sizes[2];
    int in_width = input_sizes[3];
    
    int kernel_size = weight_sizes[2];
    
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::zeros({batch, channels, out_height, out_width}, options);
    
    const int block_size = 256;
    int total_elements = batch * channels * out_height * out_width;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    depthwise_conv_bn_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        in_height,
        in_width,
        kernel_size,
        stride,
        padding,
        out_height,
        out_width
    );
    
    return output;
}
"""

depthwise_conv_bn_relu_cpp_source = """
torch::Tensor depthwise_conv_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    int stride,
    int padding
);
"""

# Compile the inline CUDA code for depthwise convolution
depthwise_conv_bn_relu = load_inline(
    name="depthwise_conv_bn_relu",
    cpp_sources=depthwise_conv_bn_relu_cpp_source,
    cuda_sources=depthwise_conv_bn_relu_source,
    functions=["depthwise_conv_bn_relu_cuda"],
    verbose=False,
)

class FusedConvBnRelu(nn.Module):
    def __init__(self, conv, bn, relu=True):
        super(FusedConvBnRelu, self).__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu
        self.conv_bn_relu = conv_bn_relu
        
    def forward(self, x):
        if x.is_cuda and x.dtype == torch.float32:
            result = self.conv_bn_relu.conv_bn_relu_cuda(
                x,
                self.conv.weight,
                self.bn.bias,
                self.bn.weight,
                self.bn.bias,
                self.conv.stride[0],
                self.conv.padding[0]
            )
            return result
        else:
            x = self.conv(x)
            x = self.bn(x)
            if self.relu:
                x = F.relu6(x, inplace=True)
            return x

class FusedDepthwiseConvBnRelu(nn.Module):
    def __init__(self, conv, bn, relu=True):
        super(FusedDepthwiseConvBnRelu, self).__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu
        self.depthwise_conv_bn_relu = depthwise_conv_bn_relu
        
    def forward(self, x):
        if x.is_cuda and x.dtype == torch.float32 and self.conv.groups == self.conv.in_channels:
            result = self.depthwise_conv_bn_relu.depthwise_conv_bn_relu_cuda(
                x,
                self.conv.weight,
                self.bn.bias,
                self.bn.weight,
                self.bn.bias,
                self.conv.stride[0],
                self.conv.padding[0]
            )
            return result
        else:
            x = self.conv(x)
            x = self.bn(x)
            if self.relu:
                x = F.relu6(x, inplace=True)
            return x

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Optimized MobileNetV2 architecture with custom CUDA kernels.
        """
        super(ModelNew, self).__init__()
        
        def _make_divisible(v, divisor, min_value=None):
            """
            This function ensures that the number of channels is divisible by the divisor.
            """
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _inverted_residual_block(inp, oup, stride, expand_ratio):
            """
            Inverted Residual Block for MobileNetV2.
            """
            hidden_dim = int(inp * expand_ratio)
            use_res_connect = stride == 1 and inp == oup

            layers = []
            if expand_ratio != 1:
                # Pointwise convolution
                conv = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
                bn = nn.BatchNorm2d(hidden_dim)
                layers.append(FusedConvBnRelu(conv, bn))

            # Depthwise convolution
            depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
            depthwise_bn = nn.BatchNorm2d(hidden_dim)
            layers.append(FusedDepthwiseConvBnRelu(depthwise_conv, depthwise_bn))

            # Pointwise linear convolution
            pw_linear_conv = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
            pw_linear_bn = nn.BatchNorm2d(oup)
            layers.append(FusedConvBnRelu(pw_linear_conv, pw_linear_bn, relu=False))

            if use_res_connect:
                return nn.Sequential(*layers), True
            else:
                return nn.Sequential(*layers), False

        # MobileNetV2 architecture
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Building first layer
        first_conv = nn.Conv2d(3, input_channel, 3, 2, 1, bias=False)
        first_bn = nn.BatchNorm2d(input_channel)
        features = [FusedConvBnRelu(first_conv, first_bn)]

        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                block, use_res_connect = _inverted_residual_block(input_channel, output_channel, stride, expand_ratio=t)
                features.append(block)
                input_channel = output_channel

        # Building last several layers
        last_conv = nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False)
        last_bn = nn.BatchNorm2d(last_channel)
        features.append(FusedConvBnRelu(last_conv, last_bn))
        
        # Adaptive average pooling
        features.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*features)

        # Linear layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass of the optimized MobileNetV2 model.

        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x