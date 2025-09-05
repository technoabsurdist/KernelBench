import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Conv2d + BatchNorm2d + ReLU6
conv_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void conv_bn_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch_size * out_channels * out_height * out_width) return;

    int n = out_idx / (out_channels * out_height * out_width);
    int c = (out_idx / (out_height * out_width)) % out_channels;
    int h = (out_idx / out_width) % out_height;
    int w = out_idx % out_width;

    float sum = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = h * stride - padding + kh;
            int iw = w * stride - padding + kw;
            
            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                for (int ic = 0; ic < in_channels; ++ic) {
                    int input_idx = ((n * in_channels + ic) * in_height + ih) * in_width + iw;
                    int weight_idx = ((c * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    sum += bias[c];
    
    // Batch norm
    float mean = running_mean[c];
    float var = running_var[c];
    float eps = 1e-5;
    float normalized = (sum - mean) / sqrtf(var + eps);
    float bn_result = gamma[c] * normalized + beta[c];
    
    // ReLU6
    float relu_result = fmaxf(0.0f, fminf(6.0f, bn_result));
    
    output[out_idx] = relu_result;
}

torch::Tensor conv_bn_relu_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta,
    int kernel_size,
    int stride,
    int padding
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    
    auto out_channels = weight.size(0);
    auto out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    auto out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * out_height * out_width + block_size - 1) / block_size;
    
    conv_bn_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding
    );
    
    return output;
}
"""

conv_bn_relu_cpp_source = """
torch::Tensor conv_bn_relu_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta,
    int kernel_size,
    int stride,
    int padding
);
"""

# Compile the inline CUDA code
conv_bn_relu = load_inline(
    name="conv_bn_relu",
    cpp_sources=conv_bn_relu_cpp_source,
    cuda_sources=conv_bn_relu_source,
    functions=["conv_bn_relu_forward"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class FusedConvBnRelu(nn.Module):
    def __init__(self, conv, bn):
        super(FusedConvBnRelu, self).__init__()
        self.conv = conv
        self.bn = bn
        
    def forward(self, x):
        if not torch.cuda.is_available() or x.device.type != 'cuda':
            # Fallback to regular implementation if not on CUDA
            x = self.conv(x)
            x = self.bn(x)
            return F.relu6(x, inplace=True)
            
        return conv_bn_relu.conv_bn_relu_forward(
            x,
            self.conv.weight,
            self.conv.bias if self.conv.bias is not None else torch.zeros(self.conv.out_channels, device=x.device),
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.weight,
            self.bn.bias,
            self.conv.kernel_size[0],
            self.conv.stride[0],
            self.conv.padding[0]
        )

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB1 architecture implementation with custom CUDA optimizations.
        """
        super(ModelNew, self).__init__()
        
        # Initial convolutional layer with fused Conv-BN-ReLU
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.fused_conv1 = FusedConvBnRelu(self.conv1, self.bn1)
        
        # MBConv blocks
        self.mbconv1 = self._make_mbconv_block(32, 16, 1, 1)
        self.mbconv2 = self._make_mbconv_block(16, 24, 2, 6)
        self.mbconv3 = self._make_mbconv_block(24, 40, 2, 6)
        self.mbconv4 = self._make_mbconv_block(40, 80, 2, 6)
        self.mbconv5 = self._make_mbconv_block(80, 112, 1, 6)
        self.mbconv6 = self._make_mbconv_block(112, 192, 2, 6)
        self.mbconv7 = self._make_mbconv_block(192, 320, 1, 6)
        
        # Final convolutional layer with fused Conv-BN-ReLU
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.fused_conv2 = FusedConvBnRelu(self.conv2, self.bn2)
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
    
    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        """
        Creates a MBConv block.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param stride: Stride of the depthwise convolution.
        :param expand_ratio: Expansion ratio for the hidden layer.
        :return: A sequential MBConv block.
        """
        hidden_dim = round(in_channels * expand_ratio)
        
        layers = []
        
        # First pointwise conv with fused Conv-BN-ReLU
        conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        bn1 = nn.BatchNorm2d(hidden_dim)
        layers.append(FusedConvBnRelu(conv1, bn1))
        
        # Depthwise conv with fused Conv-BN-ReLU
        conv_dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        bn_dw = nn.BatchNorm2d(hidden_dim)
        layers.append(FusedConvBnRelu(conv_dw, bn_dw))
        
        # Last pointwise conv with only Conv-BN (no activation)
        conv_pw = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        bn_pw = nn.BatchNorm2d(out_channels)
        layers.append(conv_pw)
        layers.append(bn_pw)
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass of the EfficientNetB1 model.

        :param x: Input tensor, shape (batch_size, 3, 240, 240)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = self.fused_conv1(x)
        
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)
        
        x = self.fused_conv2(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x