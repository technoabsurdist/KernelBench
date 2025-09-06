import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA source code for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused kernel for BatchNorm + optional Activation + optional Residual Add
// This kernel is designed for inference mode where BatchNorm uses running mean and var.
__global__ void fused_bn_activation_add_kernel(
    const float* input,
    float* output,
    const float* residual, // Can be nullptr if not adding
    const float* bn_weight,
    const float* bn_bias,
    const float* bn_mean,
    const float* bn_inv_std, // Pre-calculated 1.0f / sqrt(var + eps)
    const bool with_relu,
    const bool with_relu6,
    const bool with_add,
    const int total_elements,
    const int C,
    const int spatial_dim) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        // Calculate channel index
        const int c = (idx / spatial_dim) % C;

        // Apply BatchNorm
        float val = (input[idx] - bn_mean[c]) * bn_inv_std[c] * bn_weight[c] + bn_bias[c];

        // Apply activation if enabled
        if (with_relu) {
            val = fmaxf(0.0f, val);
        } else if (with_relu6) {
            val = fminf(fmaxf(0.0f, val), 6.0f);
        }

        // Apply residual add if enabled
        if (with_add) {
            val += residual[idx];
        }

        output[idx] = val;
    }
}

// Helper function to launch the kernel from C++ wrappers
void launch_fused_kernel(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor residual, // Can be an empty tensor
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    double bn_eps,
    bool with_relu,
    bool with_relu6,
    bool with_add) {

    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");

    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int total_elements = input.numel();
    const int spatial_dim = H * W;

    // Pre-calculate inverse standard deviation for BatchNorm
    auto inv_std = 1.0 / torch::sqrt(bn_var + bn_eps);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    const float* residual_ptr = with_add ? residual.data_ptr<float>() : nullptr;

    fused_bn_activation_add_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        residual_ptr,
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        with_relu,
        with_relu6,
        with_add,
        total_elements, C, spatial_dim
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

// C++ wrapper for BN + ReLU
torch::Tensor fused_bn_relu_cuda(
    torch::Tensor input, torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor bn_mean, torch::Tensor bn_var, double bn_eps) {
    auto output = torch::empty_like(input);
    launch_fused_kernel(input, output, torch::Tensor(), bn_weight, bn_bias, bn_mean, bn_var, bn_eps, true, false, false);
    return output;
}

// C++ wrapper for BN + ReLU6
torch::Tensor fused_bn_relu6_cuda(
    torch::Tensor input, torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor bn_mean, torch::Tensor bn_var, double bn_eps) {
    auto output = torch::empty_like(input);
    launch_fused_kernel(input, output, torch::Tensor(), bn_weight, bn_bias, bn_mean, bn_var, bn_eps, false, true, false);
    return output;
}

// C++ wrapper for BN + Add
torch::Tensor fused_bn_add_cuda(
    torch::Tensor input, torch::Tensor residual, torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor bn_mean, torch::Tensor bn_var, double bn_eps) {
    auto output = torch::empty_like(input);
    launch_fused_kernel(input, output, residual, bn_weight, bn_bias, bn_mean, bn_var, bn_eps, false, false, true);
    return output;
}

// C++ wrapper for just BN
torch::Tensor fused_bn_cuda(
    torch::Tensor input, torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor bn_mean, torch::Tensor bn_var, double bn_eps) {
    auto output = torch::empty_like(input);
    launch_fused_kernel(input, output, torch::Tensor(), bn_weight, bn_bias, bn_mean, bn_var, bn_eps, false, false, false);
    return output;
}
"""

# Define the C++ function signatures
fused_ops_cpp_source = """
torch::Tensor fused_bn_relu_cuda(torch::Tensor input, torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var, double bn_eps);
torch::Tensor fused_bn_relu6_cuda(torch::Tensor input, torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var, double bn_eps);
torch::Tensor fused_bn_add_cuda(torch::Tensor input, torch::Tensor residual, torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var, double bn_eps);
torch::Tensor fused_bn_cuda(torch::Tensor input, torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var, double bn_eps);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=[
        "fused_bn_relu_cuda",
        "fused_bn_relu6_cuda",
        "fused_bn_add_cuda",
        "fused_bn_cuda",
    ],
    verbose=True,
)


class MBConvNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, fused_ops):
        """
        MBConv block implementation using fused CUDA kernels for post-convolution operations.
        """
        super(MBConvNew, self).__init__()
        self.fused_ops = fused_ops
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.expand_ratio = expand_ratio
        hidden_dim = in_channels * expand_ratio
        
        if self.expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
            self.expand_bn = nn.BatchNorm2d(hidden_dim)
        
        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)
        
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        identity = x
        
        if self.expand_ratio != 1:
            x = self.expand_conv(x)
            bn = self.expand_bn
            # Fused BN + ReLU6
            x = self.fused_ops.fused_bn_relu6_cuda(x, bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps)
        
        x = self.depthwise_conv(x)
        bn = self.depthwise_bn
        # Fused BN + ReLU6
        x = self.fused_ops.fused_bn_relu6_cuda(x, bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps)
        
        x = self.project_conv(x)
        bn = self.project_bn
        if self.use_residual:
            # Fused BN + Residual Add
            x = self.fused_ops.fused_bn_add_cuda(x, identity, bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps)
        else:
            # Fused BN only
            x = self.fused_ops.fused_bn_cuda(x, bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps)
        
        return x


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Optimized EfficientNetB0 architecture using custom fused CUDA kernels.
        """
        super(ModelNew, self).__init__()
        
        self.fused_ops = fused_ops

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # MBConv blocks using the new fused implementation
        self.blocks = nn.Sequential(
            MBConvNew(32, 16, kernel_size=3, stride=1, expand_ratio=1, fused_ops=self.fused_ops),
            MBConvNew(16, 24, kernel_size=3, stride=2, expand_ratio=6, fused_ops=self.fused_ops),
            MBConvNew(24, 24, kernel_size=3, stride=1, expand_ratio=6, fused_ops=self.fused_ops),
            MBConvNew(24, 40, kernel_size=5, stride=2, expand_ratio=6, fused_ops=self.fused_ops),
            MBConvNew(40, 40, kernel_size=5, stride=1, expand_ratio=6, fused_ops=self.fused_ops),
            MBConvNew(40, 80, kernel_size=3, stride=2, expand_ratio=6, fused_ops=self.fused_ops),
            MBConvNew(80, 80, kernel_size=3, stride=1, expand_ratio=6, fused_ops=self.fused_ops),
            MBConvNew(80, 112, kernel_size=5, stride=1, expand_ratio=6, fused_ops=self.fused_ops),
            MBConvNew(112, 112, kernel_size=5, stride=1, expand_ratio=6, fused_ops=self.fused_ops),
            MBConvNew(112, 192, kernel_size=5, stride=2, expand_ratio=6, fused_ops=self.fused_ops),
            MBConvNew(192, 192, kernel_size=5, stride=1, expand_ratio=6, fused_ops=self.fused_ops),
            MBConvNew(192, 192, kernel_size=5, stride=1, expand_ratio=6, fused_ops=self.fused_ops),
            MBConvNew(192, 320, kernel_size=3, stride=1, expand_ratio=6, fused_ops=self.fused_ops)
        )
        
        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        # Initial Conv -> BN -> ReLU replaced with fused kernel
        x = self.conv1(x)
        bn1 = self.bn1
        x = self.fused_ops.fused_bn_relu_cuda(x, bn1.weight, bn1.bias, bn1.running_mean, bn1.running_var, bn1.eps)
        
        x = self.blocks(x)
        
        # Final Conv -> BN -> ReLU replaced with fused kernel
        x = self.conv2(x)
        bn2 = self.bn2
        x = self.fused_ops.fused_bn_relu_cuda(x, bn2.weight, bn2.bias, bn2.running_mean, bn2.running_var, bn2.eps)
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x