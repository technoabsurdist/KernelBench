import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused BatchNorm2d + ReLU
# This kernel is designed for inference mode (model.eval())
bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bn_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float eps,
    const int total_elements,
    const int C,
    const int spatial_dim) {

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x) {
        // Calculate channel index from the flat tensor index
        const int c = (idx / spatial_dim) % C;

        // Pre-calculate the scale and shift for the BN transformation
        // y = (x - mean) / sqrt(var + eps) * weight + bias
        // This can be rewritten as: y = x * (weight / sqrt(var + eps)) + (bias - mean * weight / sqrt(var + eps))
        // Let scale = weight / sqrt(var + eps)
        // Let shift = bias - mean * scale
        const float scale = weight[c] * rsqrtf(running_var[c] + eps);
        const float shift = bias[c] - running_mean[c] * scale;

        // Apply fused BatchNorm and ReLU
        const float bn_val = input[idx] * scale + shift;
        output[idx] = fmaxf(0.0f, bn_val);
    }
}

torch::Tensor bn_relu_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {

    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "Running Mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "Running Var must be a CUDA tensor");

    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    TORCH_CHECK(running_mean.is_contiguous(), "Running Mean must be contiguous");
    TORCH_CHECK(running_var.is_contiguous(), "Running Var must be contiguous");
    
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    const int total_elements = input.numel();
    const int spatial_dim = H * W;

    auto output = torch::empty_like(input);

    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    bn_relu_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        total_elements,
        C,
        spatial_dim
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

bn_relu_cpp_source = """
torch::Tensor bn_relu_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps);
"""

# Compile the inline CUDA code for Fused BatchNorm + ReLU
bn_relu = load_inline(
    name="bn_relu",
    cpp_sources=bn_relu_cpp_source,
    cuda_sources=bn_relu_source,
    functions=["bn_relu_forward_cuda"],
    verbose=False,
)

class FusedBNReLU(nn.Module):
    """
    A module that fuses BatchNorm2d and ReLU for inference.
    During training, it falls back to the standard PyTorch implementation.
    """
    def __init__(self, bn):
        super().__init__()
        # Store the original BatchNorm layer
        self.bn = bn

    def forward(self, x):
        if self.training:
            # Fallback to standard PyTorch ops during training
            return F.relu(self.bn(x))
        else:
            # Use the custom CUDA kernel for inference
            return bn_relu.bn_relu_forward_cuda(
                x,
                self.bn.weight,
                self.bn.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps
            )

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Optimized EfficientNetB2 architecture with fused BatchNorm+ReLU kernels.
        """
        super(ModelNew, self).__init__()
        
        # Define the EfficientNetB2 architecture components
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Fused BN+ReLU
        self.fused_bn_relu1 = FusedBNReLU(nn.BatchNorm2d(32))
        
        # Define the MBConv blocks using the new helper
        self.mbconv1 = self._make_mbconv_block(32, 96, 1, 3)
        self.mbconv2 = self._make_mbconv_block(96, 144, 2, 6)
        self.mbconv3 = self._make_mbconv_block(144, 192, 2, 6)
        self.mbconv4 = self._make_mbconv_block(192, 288, 2, 6)
        self.mbconv5 = self._make_mbconv_block(288, 384, 1, 6)
        
        # Final layers
        self.conv_final = nn.Conv2d(384, 1408, kernel_size=1, stride=1, padding=0, bias=False)
        # Fused BN+ReLU
        self.fused_bn_relu_final = FusedBNReLU(nn.BatchNorm2d(1408))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1408, num_classes)
    
    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        """
        Helper function to create a MBConv block with fused operations.
        """
        layers = []
        expanded_channels = in_channels * expand_ratio
        
        # Expansion phase
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
            # Fused BN+ReLU
            layers.append(FusedBNReLU(nn.BatchNorm2d(expanded_channels)))
        
        # Depthwise convolution
        layers.append(nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False))
        # Fused BN+ReLU
        layers.append(FusedBNReLU(nn.BatchNorm2d(expanded_channels)))
        
        # Squeeze and Excitation (left unchanged as it doesn't follow the BN->ReLU pattern)
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Conv2d(expanded_channels, expanded_channels // 4, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(expanded_channels // 4, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.Sigmoid())
        
        # Output phase (no ReLU, so no fusion)
        layers.append(nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass of the optimized EfficientNetB2 model.
        """
        x = self.fused_bn_relu1(self.conv1(x))
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.fused_bn_relu_final(self.conv_final(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x