import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv + batchnorm + relu6
fused_conv_bn_relu6_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_conv_bn_relu6_kernel(
    const float* input,
    const float* weight,
    const float* bn_weight,
    const float* bn_bias,
    const float* bn_mean,
    const float* bn_var,
    float* output,
    int batch,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    float eps
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch * out_channels * out_height * out_width) return;

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
                    int input_idx = n * (in_channels * in_height * in_width) +
                                    ic * (in_height * in_width) +
                                    ih * in_width + iw;
                    int weight_idx = c * (in_channels * kernel_size * kernel_size) +
                                     ic * (kernel_size * kernel_size) +
                                     kh * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // BatchNorm
    float mean = bn_mean[c];
    float var = bn_var[c];
    float weight_bn = bn_weight[c];
    float bias_bn = bn_bias[c];
    float normalized = (sum - mean) / sqrtf(var + eps);
    float bn_result = weight_bn * normalized + bias_bn;

    // ReLU6
    float relu6_result = fmaxf(0.0f, fminf(6.0f, bn_result));

    output[out_idx] = relu6_result;
}

__global__ void fused_conv_bn_kernel(
    const float* input,
    const float* weight,
    const float* bn_weight,
    const float* bn_bias,
    const float* bn_mean,
    const float* bn_var,
    float* output,
    int batch,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    float eps
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch * out_channels * out_height * out_width) return;

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
                    int input_idx = n * (in_channels * in_height * in_width) +
                                    ic * (in_height * in_width) +
                                    ih * in_width + iw;
                    int weight_idx = c * (in_channels * kernel_size * kernel_size) +
                                     ic * (kernel_size * kernel_size) +
                                     kh * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // BatchNorm
    float mean = bn_mean[c];
    float var = bn_var[c];
    float weight_bn = bn_weight[c];
    float bias_bn = bn_bias[c];
    float normalized = (sum - mean) / sqrtf(var + eps);
    float bn_result = weight_bn * normalized + bias_bn;

    output[out_idx] = bn_result;
}

torch::Tensor fused_conv_bn_relu6_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    int kernel_size,
    int stride,
    int padding,
    float eps
) {
    auto batch = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto out_channels = weight.size(0);
    auto out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    auto out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch, out_channels, out_height, out_width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    const int num_blocks = (batch * out_channels * out_height * out_width + block_size - 1) / block_size;
    
    fused_conv_bn_relu6_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        eps
    );
    
    return output;
}

torch::Tensor fused_conv_bn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    int kernel_size,
    int stride,
    int padding,
    float eps
) {
    auto batch = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto out_channels = weight.size(0);
    auto out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    auto out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch, out_channels, out_height, out_width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    const int num_blocks = (batch * out_channels * out_height * out_width + block_size - 1) / block_size;
    
    fused_conv_bn_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        eps
    );
    
    return output;
}
"""

fused_conv_bn_relu6_cpp_source = """
torch::Tensor fused_conv_bn_relu6_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    int kernel_size,
    int stride,
    int padding,
    float eps
);

torch::Tensor fused_conv_bn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    int kernel_size,
    int stride,
    int padding,
    float eps
);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_conv_bn_relu6",
    cpp_sources=fused_conv_bn_relu6_cpp_source,
    cuda_sources=fused_conv_bn_relu6_source,
    functions=["fused_conv_bn_relu6_cuda", "fused_conv_bn_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class FusedConvBnRelu6(nn.Module):
    def __init__(self, conv, bn, kernel_size, stride, padding):
        super(FusedConvBnRelu6, self).__init__()
        self.conv_weight = conv.weight
        self.bn_weight = bn.weight
        self.bn_bias = bn.bias
        self.bn_mean = bn.running_mean
        self.bn_var = bn.running_var
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.eps = bn.eps
        self.fused_conv_bn_relu6 = fused_ops

    def forward(self, x):
        return self.fused_conv_bn_relu6.fused_conv_bn_relu6_cuda(
            x,
            self.conv_weight,
            self.bn_weight,
            self.bn_bias,
            self.bn_mean,
            self.bn_var,
            self.kernel_size,
            self.stride,
            self.padding,
            self.eps
        )

class FusedConvBn(nn.Module):
    def __init__(self, conv, bn, kernel_size, stride, padding):
        super(FusedConvBn, self).__init__()
        self.conv_weight = conv.weight
        self.bn_weight = bn.weight
        self.bn_bias = bn.bias
        self.bn_mean = bn.running_mean
        self.bn_var = bn.running_var
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.eps = bn.eps
        self.fused_conv_bn = fused_ops

    def forward(self, x):
        return self.fused_conv_bn.fused_conv_bn_cuda(
            x,
            self.conv_weight,
            self.bn_weight,
            self.bn_bias,
            self.bn_mean,
            self.bn_var,
            self.kernel_size,
            self.stride,
            self.padding,
            self.eps
        )

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        Optimized MBConv block implementation with fused CUDA kernels.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Kernel size for the depthwise convolution.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the intermediate channels.
        """
        super(ModelNew, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        if expand_ratio != 1:
            # Fuse expand conv + batchnorm + relu6
            expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
            expand_bn = nn.BatchNorm2d(hidden_dim)
            # Initialize with dummy values to avoid errors
            nn.init.ones_(expand_bn.weight)
            nn.init.zeros_(expand_bn.bias)
            self.fused_expand = FusedConvBnRelu6(expand_conv, expand_bn, 1, 1, 0)
        
        # Fuse depthwise conv + batchnorm + relu6
        depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, 
                                  padding=(kernel_size-1)//2, groups=hidden_dim, bias=False)
        depthwise_bn = nn.BatchNorm2d(hidden_dim)
        # Initialize with dummy values to avoid errors
        nn.init.ones_(depthwise_bn.weight)
        nn.init.zeros_(depthwise_bn.bias)
        self.fused_depthwise = FusedConvBnRelu6(depthwise_conv, depthwise_bn, kernel_size, stride, (kernel_size-1)//2)
        
        # Fuse project conv + batchnorm
        project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        project_bn = nn.BatchNorm2d(out_channels)
        # Initialize with dummy values to avoid errors
        nn.init.ones_(project_bn.weight)
        nn.init.zeros_(project_bn.bias)
        self.fused_project = FusedConvBn(project_conv, project_bn, 1, 1, 0)
    
    def forward(self, x):
        """
        Forward pass of the optimized MBConv block.

        :param x: The input tensor, shape (batch_size, in_channels, H, W)
        :return: The output tensor, shape (batch_size, out_channels, H', W')
        """
        identity = x
        
        if hasattr(self, 'fused_expand'):
            x = self.fused_expand(x)
        
        x = self.fused_depthwise(x)
        x = self.fused_project(x)
        
        if self.use_residual:
            x += identity
        
        return x