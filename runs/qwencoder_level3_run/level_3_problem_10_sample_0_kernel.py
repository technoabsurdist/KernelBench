import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv-BN-ReLU
conv_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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
    int kernel_size,
    int stride,
    int padding,
    float eps
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * ((in_height + 2 * padding - kernel_size) / stride + 1) * ((in_width + 2 * padding - kernel_size) / stride + 1);
    
    if (out_idx >= total_outputs) return;
    
    int batch = out_idx / (out_channels * ((in_height + 2 * padding - kernel_size) / stride + 1) * ((in_width + 2 * padding - kernel_size) / stride + 1));
    int temp = out_idx % (out_channels * ((in_height + 2 * padding - kernel_size) / stride + 1) * ((in_width + 2 * padding - kernel_size) / stride + 1));
    int out_ch = temp / (((in_height + 2 * padding - kernel_size) / stride + 1) * ((in_width + 2 * padding - kernel_size) / stride + 1));
    temp = temp % (((in_height + 2 * padding - kernel_size) / stride + 1) * ((in_width + 2 * padding - kernel_size) / stride + 1));
    int out_h = temp / ((in_width + 2 * padding - kernel_size) / stride + 1);
    int out_w = temp % ((in_width + 2 * padding - kernel_size) / stride + 1);
    
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    float sum = 0.0f;
    
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int in_h = out_h * stride + kh - padding;
            int in_w = out_w * stride + kw - padding;
            
            if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                for (int ic = 0; ic < in_channels; ++ic) {
                    int input_idx = batch * (in_channels * in_height * in_width) + 
                                   ic * (in_height * in_width) + 
                                   in_h * in_width + in_w;
                                   
                    int weight_idx = out_ch * (in_channels * kernel_size * kernel_size) + 
                                    ic * (kernel_size * kernel_size) + 
                                    kh * kernel_size + kw;
                                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    sum += bias[out_ch];
    
    // Batch norm
    float mean = running_mean[out_ch];
    float var = running_var[out_ch];
    float gamma_val = gamma[out_ch];
    float beta_val = beta[out_ch];
    
    float normalized = (sum - mean) / sqrtf(var + eps);
    float bn_result = gamma_val * normalized + beta_val;
    
    // ReLU
    float relu_result = fmaxf(0.0f, bn_result);
    
    output[out_idx] = relu_result;
}

torch::Tensor conv_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta,
    int kernel_size,
    int stride,
    int padding,
    float eps
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto out_channels = weight.size(0);
    
    auto out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    auto out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    auto total_outputs = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_outputs + block_size - 1) / block_size;
    
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
        kernel_size,
        stride,
        padding,
        eps
    );
    
    return output;
}
"""

conv_bn_relu_cpp_source = """
torch::Tensor conv_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta,
    int kernel_size,
    int stride,
    int padding,
    float eps
);
"""

# Compile the inline CUDA code for fused Conv-BN-ReLU
conv_bn_relu = load_inline(
    name="conv_bn_relu",
    cpp_sources=conv_bn_relu_cpp_source,
    cuda_sources=conv_bn_relu_source,
    functions=["conv_bn_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for element-wise addition (residual connection)
elementwise_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

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

elementwise_add_cpp_source = """
torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);
"""

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class FusedConvBnRelu(nn.Module):
    def __init__(self, conv, bn):
        super(FusedConvBnRelu, self).__init__()
        self.conv = conv
        self.bn = bn
        self.conv_bn_relu = conv_bn_relu
        
    def forward(self, x):
        # Extract parameters from conv and bn
        weight = self.conv.weight
        bias = torch.zeros(weight.size(0), device=weight.device)
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        gamma = self.bn.weight
        beta = self.bn.bias
        eps = self.bn.eps
        
        # Get conv parameters
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        padding = self.conv.padding[0]
        
        return self.conv_bn_relu.conv_bn_relu_cuda(
            x, weight, bias, running_mean, running_var, gamma, beta,
            kernel_size, stride, padding, eps
        )

class BottleneckNew(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.stride = stride
        
        # Fuse conv-bn-relu for first two convolutions
        self.fused_conv1 = FusedConvBnRelu(self.conv1, self.bn1)
        self.fused_conv2 = FusedConvBnRelu(self.conv2, self.bn2)
        
        # Element-wise addition module
        self.elementwise_add = elementwise_add

    def forward(self, x):
        identity = x

        # Use fused conv-bn-relu
        out = self.fused_conv1(x)
        out = self.fused_conv2(out)

        # Third conv + bn (no ReLU)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Use custom element-wise addition
        out = self.elementwise_add.elementwise_add_cuda(out, identity)
        
        # Apply ReLU manually
        out = F.relu(out, inplace=True)

        return out

class ModelNew(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = BottleneckNew

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Fuse first conv-bn-relu
        self.fused_conv1 = FusedConvBnRelu(self.conv1, self.bn1)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Use fused conv-bn-relu for the first layer
        x = self.fused_conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x