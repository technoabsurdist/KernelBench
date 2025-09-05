import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2D + BatchNorm + ReLU
fused_conv_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cudnn.h>

__global__ void fused_bn_relu_kernel(
    float* output,
    const float* input,
    const float* gamma,
    const float* beta,
    const float* mean,
    const float* var,
    int batch_size,
    int channels,
    int height,
    int width,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        int w = idx % width;
        int h = (idx / width) % height;
        int c = (idx / (width * height)) % channels;
        int n = idx / (width * height * channels);
        
        float val = input[idx];
        float norm_val = (val - mean[c]) / sqrtf(var[c] + eps);
        float bn_out = gamma[c] * norm_val + beta[c];
        output[idx] = fmaxf(bn_out, 0.0f);  // ReLU
    }
}

torch::Tensor fused_conv_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    int stride,
    int padding,
    int dilation,
    int groups,
    float eps
) {
    // Perform convolution
    auto conv_output = torch::conv2d(
        input, weight, bias,
        /*stride=*/{stride, stride},
        /*padding=*/{padding, padding},
        /*dilation=*/{dilation, dilation},
        /*groups=*/groups
    );
    
    // Get dimensions
    int batch_size = conv_output.size(0);
    int channels = conv_output.size(1);
    int height = conv_output.size(2);
    int width = conv_output.size(3);
    int total_elements = batch_size * channels * height * width;
    
    // Allocate output
    auto output = torch::empty_like(conv_output);
    
    // Launch fused BN+ReLU kernel
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_bn_relu_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(),
        conv_output.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        batch_size, channels, height, width, eps
    );
    
    return output;
}
"""

fused_conv_bn_relu_cpp_source = """
torch::Tensor fused_conv_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    int stride,
    int padding,
    int dilation,
    int groups,
    float eps
);
"""

# Compile the inline CUDA code
fused_conv_bn_relu = load_inline(
    name="fused_conv_bn_relu",
    cpp_sources=fused_conv_bn_relu_cpp_source,
    cuda_sources=fused_conv_bn_relu_source,
    functions=["fused_conv_bn_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class FusedConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(FusedConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
    def forward(self, x):
        if self.training:
            # Use standard PyTorch during training for proper gradient computation
            return F.relu(self.bn(self.conv(x)), inplace=True)
        else:
            # Use fused kernel during inference
            return fused_conv_bn_relu.fused_conv_bn_relu_cuda(
                x,
                self.conv.weight,
                torch.zeros(self.conv.weight.size(0), device=x.device, dtype=x.dtype),  # No bias in conv
                self.bn.weight,
                self.bn.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                self.bn.eps
            )


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        """
        MobileNetV1 architecture implementation with custom CUDA kernels.

        :param num_classes: The number of output classes (default: 1000)
        :param input_channels: The number of input channels (default: 3 for RGB images)
        :param alpha: Width multiplier (default: 1.0)
        """
        super(ModelNew, self).__init__()
        
        def conv_bn(inp, oup, stride):
            return FusedConvBNReLU(inp, oup, 3, stride, 1, 1, 1)
        
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                FusedConvBNReLU(inp, inp, 3, stride, 1, 1, inp),  # Depthwise
                FusedConvBNReLU(inp, oup, 1, 1, 0, 1, 1),  # Pointwise
            )
        
        self.model = nn.Sequential(
            conv_bn(input_channels, int(32 * alpha), 2),
            conv_dw(int(32 * alpha), int(64 * alpha), 1),
            conv_dw(int(64 * alpha), int(128 * alpha), 2),
            conv_dw(int(128 * alpha), int(128 * alpha), 1),
            conv_dw(int(128 * alpha), int(256 * alpha), 2),
            conv_dw(int(256 * alpha), int(256 * alpha), 1),
            conv_dw(int(256 * alpha), int(512 * alpha), 2),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(1024 * alpha), 2),
            conv_dw(int(1024 * alpha), int(1024 * alpha), 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(int(1024 * alpha), num_classes)
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_channels, height, width)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_inputs():
    batch_size = 10
    input_channels = 3
    height = 224
    width = 224
    return [torch.rand(batch_size, input_channels, height, width).cuda()]

def get_init_inputs():
    num_classes = 1000
    input_channels = 3
    alpha = 1.0
    return [num_classes, input_channels, alpha]