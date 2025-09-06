import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# 1. Custom CUDA Kernel for Fused BatchNorm + Activation
# -----------------------------------------------------------------------------

fused_bn_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bn_activation_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    const float* __restrict__ bn_running_mean,
    const float* __restrict__ bn_running_var,
    const float eps,
    const float activation_upper_bound,
    const int C,
    const int H,
    const int W,
    const int total_elements) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }

    // Decompose index to find the channel index 'c'
    const int c = (idx / (H * W)) % C;

    // Fetch channel-wise BatchNorm parameters
    const float mean = bn_running_mean[c];
    const float var = bn_running_var[c];
    const float weight = bn_weight[c];
    const float bias = bn_bias[c];

    // --- Fused Operation ---
    // 1. Batch Normalization (Inference)
    const float inv_std = rsqrtf(var + eps);
    const float normalized = (input[idx] - mean) * inv_std;
    const float scaled = weight * normalized + bias;

    // 2. Activation (ReLU or ReLU6)
    float activated = fmaxf(0.0f, scaled);
    if (activation_upper_bound > 0.0f) {
        output[idx] = fminf(activated, activation_upper_bound);
    } else {
        output[idx] = activated;
    }
}

torch::Tensor fused_bn_activation_cuda(
    torch::Tensor input,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    double eps,
    float activation_upper_bound) {

    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(bn_weight.is_contiguous(), "bn_weight must be contiguous");
    TORCH_CHECK(bn_bias.is_contiguous(), "bn_bias must be contiguous");
    TORCH_CHECK(bn_running_mean.is_contiguous(), "bn_running_mean must be contiguous");
    TORCH_CHECK(bn_running_var.is_contiguous(), "bn_running_var must be contiguous");

    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    const auto total_elements = input.numel();

    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_bn_activation_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_running_mean.data_ptr<float>(),
        bn_running_var.data_ptr<float>(),
        static_cast<float>(eps),
        activation_upper_bound,
        C, H, W,
        total_elements
    );
    
    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

fused_bn_activation_cpp_source = """
torch::Tensor fused_bn_activation_cuda(
    torch::Tensor input,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    double eps,
    float activation_upper_bound);
"""

# JIT compile the custom CUDA kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_bn_activation_cpp_source,
    cuda_sources=fused_bn_activation_source,
    functions=["fused_bn_activation_cuda"],
    verbose=False,
)

# -----------------------------------------------------------------------------
# 2. PyTorch Modules for the New Architecture
# -----------------------------------------------------------------------------

class FusedConvBNActivation(nn.Module):
    """
    A module that fuses Convolution, BatchNorm, and an activation function.
    Uses a custom CUDA kernel for inference and standard PyTorch ops for training.
    """
    def __init__(self, conv, bn, activation_type='relu'):
        super().__init__()
        self.conv = conv
        self.bn = bn
        if activation_type == 'relu6':
            self.activation = nn.ReLU6(inplace=True)
            self.activation_upper_bound = 6.0
        elif activation_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
            self.activation_upper_bound = 0.0
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

    def forward(self, x):
        x = self.conv(x)
        if self.training:
            # Use standard PyTorch ops during training
            x = self.bn(x)
            x = self.activation(x)
            return x
        else:
            # Use custom fused CUDA kernel during inference
            return fused_op.fused_bn_activation_cuda(
                x,
                self.bn.weight,
                self.bn.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps,
                self.activation_upper_bound,
            )

class MBConvBlockFused(nn.Module):
    """
    An MBConv block implementation that uses the FusedConvBNActivation module.
    """
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = round(in_channels * expand_ratio)

        # Expansion phase
        self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.expand_bn = nn.BatchNorm2d(hidden_dim)
        self.fused_expand = FusedConvBNActivation(self.expand_conv, self.expand_bn, 'relu6')

        # Depthwise phase
        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)
        self.fused_depthwise = FusedConvBNActivation(self.depthwise_conv, self.depthwise_bn, 'relu6')

        # Projection phase (no activation)
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.fused_expand(x)
        x = self.fused_depthwise(x)
        x = self.project_bn(self.project_conv(x))
        return x

# -----------------------------------------------------------------------------
# 3. The Optimized ModelNew Architecture
# -----------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Optimized EfficientNetB1 architecture using fused operators.
        """
        super(ModelNew, self).__init__()
        
        # Initial convolutional layer (Stem)
        conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(32)
        self.stem = FusedConvBNActivation(conv1, bn1, activation_type='relu')
        
        # MBConv blocks
        self.mbconv1 = self._make_mbconv_block(32, 16, 1, 1)
        self.mbconv2 = self._make_mbconv_block(16, 24, 2, 6)
        self.mbconv3 = self._make_mbconv_block(24, 40, 2, 6)
        self.mbconv4 = self._make_mbconv_block(40, 80, 2, 6)
        self.mbconv5 = self._make_mbconv_block(80, 112, 1, 6)
        self.mbconv6 = self._make_mbconv_block(112, 192, 2, 6)
        self.mbconv7 = self._make_mbconv_block(192, 320, 1, 6)
        
        # Final convolutional layer (Head)
        conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        bn2 = nn.BatchNorm2d(1280)
        self.head = FusedConvBNActivation(conv2, bn2, activation_type='relu')
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
    
    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        """Creates a fused MBConv block."""
        return MBConvBlockFused(in_channels, out_channels, stride, expand_ratio)
    
    def forward(self, x):
        """
        Forward pass of the optimized EfficientNetB1 model.
        """
        x = self.stem(x)
        
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)
        
        x = self.head(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x