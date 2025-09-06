import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from typing import Optional

# Define the custom CUDA kernel for a fused Conv2d -> BatchNorm2d -> ReLU operation.
# This is an inference-only optimization.
# The C++ part calls PyTorch's native convolution (which uses cuDNN)
# and the CUDA kernel then fuses the BatchNorm and ReLU operations, reducing
# kernel launch overhead and memory bandwidth by avoiding intermediate tensors.
fused_conv_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel to perform BatchNorm followed by ReLU
// Assumes NCHW tensor layout
__global__ void bn_relu_kernel_NCHW(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int total_elements,
    const int C, const int H, const int W,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float eps) {

    const int hw = H * W;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        const int channel = (idx / hw) % C;

        // Batch Normalization
        const float inv_std = rsqrtf(var[channel] + eps);
        const float normalized = gamma[channel] * (input[idx] - mean[channel]) * inv_std + beta[channel];

        // ReLU
        output[idx] = fmaxf(0.0f, normalized);
    }
}

// C++ function that calls PyTorch's convolution and then our custom kernel
torch::Tensor conv_bn_relu_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const torch::Tensor& bn_gamma,
    const torch::Tensor& bn_beta,
    const torch::Tensor& bn_mean,
    const torch::Tensor& bn_var,
    const std::vector<long>& stride,
    const std::vector<long>& padding,
    const std::vector<long>& dilation,
    const long groups,
    const float bn_eps) {

    // Ensure input tensors are on CUDA
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bn_gamma.is_cuda(), "BatchNorm gamma must be a CUDA tensor");
    TORCH_CHECK(bn_beta.is_cuda(), "BatchNorm beta must be a CUDA tensor");
    TORCH_CHECK(bn_mean.is_cuda(), "BatchNorm mean must be a CUDA tensor");
    TORCH_CHECK(bn_var.is_cuda(), "BatchNorm var must be a CUDA tensor");

    // Step 1: Perform convolution using PyTorch's highly optimized implementation (e.g., cuDNN)
    torch::Tensor conv_out = at::convolution(input, weight, bias, stride, padding, dilation, false, {0, 0}, groups);

    // Step 2: Prepare for the custom BN+ReLU kernel
    auto output = torch::empty_like(conv_out);
    const int C = conv_out.size(1);
    const int H = conv_out.size(2);
    const int W = conv_out.size(3);
    const int total_elements = conv_out.numel();

    const int block_size = 1024;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    // Step 3: Launch the custom CUDA kernel
    bn_relu_kernel_NCHW<<<num_blocks, block_size>>>(
        conv_out.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements, C, H, W,
        bn_gamma.data_ptr<float>(),
        bn_beta.data_ptr<float>(),
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        bn_eps
    );
    
    // Check for any CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}
"""

# C++ source for function signature definition
fused_conv_bn_relu_cpp_source = """
torch::Tensor conv_bn_relu_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const torch::Tensor& bn_gamma,
    const torch::Tensor& bn_beta,
    const torch::Tensor& bn_mean,
    const torch::Tensor& bn_var,
    const std::vector<long>& stride,
    const std::vector<long>& padding,
    const std::vector<long>& dilation,
    const long groups,
    const float bn_eps);
"""

# JIT compile the custom CUDA operator
# This is done once when the module is first loaded.
fused_op_module = load_inline(
    name="fused_conv_bn_relu_op",
    cpp_sources=fused_conv_bn_relu_cpp_source,
    cuda_sources=fused_conv_bn_relu_source,
    functions=["conv_bn_relu_forward"],
    verbose=True,
)


class FusedConvBNReLU(nn.Module):
    """
    A module that fuses Conv2d, BatchNorm2d, and ReLU for inference.
    In training mode, it uses the standard separate layers to ensure
    correct behavior of BatchNorm.
    In eval mode, it uses a custom CUDA kernel.
    """
    def __init__(self, inp: int, oup: int, stride: int, kernel_size: int = 3, padding: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(inp, oup, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(oup)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Standard path for training to update BN stats correctly
            x = self.conv(x)
            x = self.bn(x)
            return F.relu(x, inplace=True)
        else:
            # Fused path for inference
            return fused_op_module.conv_bn_relu_forward(
                x,
                self.conv.weight,
                self.conv.bias,
                self.bn.weight,
                self.bn.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.conv.stride,
                self.conv.padding,
                self.conv.dilation,
                self.conv.groups,
                self.bn.eps
            )

class FusedDepthwiseSeparableConv(nn.Module):
    """
    The depthwise separable convolution block from MobileNetV1,
    but using the FusedConvBNReLU module.
    """
    def __init__(self, inp: int, oup: int, stride: int):
        super().__init__()
        # Depthwise convolution
        self.depthwise = FusedConvBNReLU(inp, inp, stride, kernel_size=3, padding=1, groups=inp, bias=False)
        # Pointwise convolution
        self.pointwise = FusedConvBNReLU(inp, oup, stride=1, kernel_size=1, padding=0, groups=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ModelNew(nn.Module):
    def __init__(self, num_classes: int = 1000, input_channels: int = 3, alpha: float = 1.0):
        """
        MobileNetV1 architecture with fused Conv-BN-ReLU operators for inference speedup.
        The model behaves identically to the original during training but uses custom
        CUDA kernels for faster inference when in .eval() mode.

        :param num_classes: The number of output classes (default: 1000)
        :param input_channels: The number of input channels (default: 3 for RGB images)
        :param alpha: Width multiplier (default: 1.0)
        """
        super(ModelNew, self).__init__()
        
        # Use our fused modules instead of the original nn.Sequential factories
        def conv_bn_fused(inp, oup, stride):
            return FusedConvBNReLU(inp, oup, stride, kernel_size=3, padding=1, bias=False)
        
        def conv_dw_fused(inp, oup, stride):
            return FusedDepthwiseSeparableConv(inp, oup, stride)
        
        # Helper to apply the width multiplier
        c = lambda x: int(x * alpha)

        self.model = nn.Sequential(
            conv_bn_fused(input_channels, c(32), 2),
            conv_dw_fused(c(32), c(64), 1),
            conv_dw_fused(c(64), c(128), 2),
            conv_dw_fused(c(128), c(128), 1),
            conv_dw_fused(c(128), c(256), 2),
            conv_dw_fused(c(256), c(256), 1),
            conv_dw_fused(c(256), c(512), 2),
            conv_dw_fused(c(512), c(512), 1),
            conv_dw_fused(c(512), c(512), 1),
            conv_dw_fused(c(512), c(512), 1),
            conv_dw_fused(c(512), c(512), 1),
            conv_dw_fused(c(512), c(512), 1),
            conv_dw_fused(c(512), c(1024), 2),
            conv_dw_fused(c(1024), c(1024), 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(c(1024), num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: The input tensor, shape (batch_size, input_channels, height, width)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x