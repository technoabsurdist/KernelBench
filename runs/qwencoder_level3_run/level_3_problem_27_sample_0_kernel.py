import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + BatchNorm2d + ReLU
conv_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void conv3x3_bn_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* bn_weight,
    const float* bn_bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
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
    
    // Convolution 3x3
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
                int ih = h * 2 + kh - 1; // Stride=2 for pooling effect
                int iw = w * 2 + kw - 1;
                
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    int input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                    int weight_idx = ((c * in_channels + ic) * 3 + kh) * 3 + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    sum += bias[c];
    
    // BatchNorm
    float mean = running_mean[c];
    float var = running_var[c];
    float eps = 1e-5f;
    float normalized = (sum - mean) / sqrtf(var + eps);
    float bn_out = bn_weight[c] * normalized + bn_bias[c];
    
    // ReLU
    float relu_out = fmaxf(0.0f, bn_out);
    
    output[idx] = relu_out;
}

torch::Tensor conv_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto batch = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    
    auto out_channels = weight.size(0);
    auto out_height = (in_height - 1) / 2; // Due to stride=2 in pooling
    auto out_width = (in_width - 1) / 2;
    
    auto output = torch::zeros({batch, out_channels, out_height, out_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    int total_elements = batch * out_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv3x3_bn_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_channels,
        in_height,
        in_width,
        out_channels,
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
    torch::Tensor running_mean,
    torch::Tensor running_var
);
"""

# Compile the inline CUDA code for fused Conv2d + BatchNorm2d + ReLU
conv_bn_relu = load_inline(
    name="conv_bn_relu",
    cpp_sources=conv_bn_relu_cpp_source,
    cuda_sources=conv_bn_relu_source,
    functions=["conv_bn_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class FusedConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusedConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv_bn_relu = conv_bn_relu

    def forward(self, x):
        return self.conv_bn_relu.conv_bn_relu_cuda(
            x,
            self.conv.weight,
            self.conv.bias,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var
        )

class ModelNew(nn.Module):
    def __init__(self, input_channels, stages, block_widths, output_classes):
        """
        :param input_channels: int, Number of input channels for the first layer
        :param stages: int, Number of stages in the RegNet architecture
        :param block_widths: List[int], Width (number of channels) for each block in the stages
        :param output_classes: int, Number of output classes for classification
        """
        super(ModelNew, self).__init__()

        self.stages = stages
        self.block_widths = block_widths
        
        layers = []
        current_channels = input_channels
        
        # Construct the stages with their respective blocks using fused operations
        for i in range(stages):
            layers.append(self._make_stage(current_channels, block_widths[i]))
            current_channels = block_widths[i]
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Final fully connected layer for classification
        self.fc = nn.Linear(block_widths[-1], output_classes)
    
    def _make_stage(self, in_channels, out_channels):
        """
        Creates a simple block for each stage with fused operations.
        :param in_channels: int, number of input channels
        :param out_channels: int, number of output channels
        :return: nn.Sequential block with fused convolutional layers
        """
        return nn.Sequential(
            FusedConvBnReLU(in_channels, out_channels),
            FusedConvBnReLU(out_channels, out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        Forward pass through the RegNet model.
        :param x: torch.Tensor of shape (batch_size, input_channels, height, width)
        :return: torch.Tensor of shape (batch_size, output_classes)
        """
        x = self.feature_extractor(x)
        x = torch.mean(x, dim=[2, 3])  # Global Average Pooling
        x = self.fc(x)
        return x