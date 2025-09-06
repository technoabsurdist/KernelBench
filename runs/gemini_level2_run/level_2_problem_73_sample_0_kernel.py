import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fusing BatchNorm, and scaling
fused_bn_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bn_scale_kernel(
    const float* input,
    const float* weight, // Fused weight: scaling_factor * gamma / sqrt(var + eps)
    const float* bias,   // Fused bias: scaling_factor * (beta - mean * gamma / sqrt(var + eps))
    float* output,
    int total_elements,
    int channels,
    int spatial_dim) // H * W
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        // Calculate channel index 'c' from the flat memory index 'idx'
        int c = (idx / spatial_dim) % channels;
        // Apply the fused scale and bias
        output[idx] = input[idx] * weight[c] + bias[c];
    }
}

torch::Tensor fused_bn_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor (NCHW)");

    const auto channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);

    const int total_elements = input.numel();
    const int spatial_dim = height * width;

    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_bn_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        channels,
        spatial_dim);

    // Check for any CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

fused_bn_scale_cpp_source = """
torch::Tensor fused_bn_scale_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# JIT compile the CUDA kernel
fused_bn_scale = load_inline(
    name="fused_bn_scale",
    cpp_sources=fused_bn_scale_cpp_source,
    cuda_sources=fused_bn_scale_source,
    functions=["fused_bn_scale_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses Batch Normalization and scaling into a single custom CUDA kernel for inference.
    During training, it behaves identically to the original model.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        # Keep the original layers
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

        # Load the custom CUDA kernel
        self.fused_op = fused_bn_scale

        # Buffers to store the fused parameters for inference.
        # These are part of the model's state but not trained.
        self.register_buffer('fused_weight', None)
        self.register_buffer('fused_bias', None)

    def forward(self, x):
        # 1. Convolution (unchanged)
        x = self.conv(x)

        # 2. Fused BatchNorm + Scaling
        if self.training:
            # In training mode, use the standard PyTorch layers
            # to ensure correct statistics are updated in BatchNorm.
            x = self.bn(x)
            x = x * self.scaling_factor
            # Invalidate cached parameters if we switch back to training
            self.fused_weight = None
            self.fused_bias = None
        else:
            # In inference mode, use the custom fused CUDA kernel.
            # The fused parameters are computed and cached on the first inference pass.
            if self.fused_weight is None or self.fused_bias is None:
                # Pre-calculate the fused parameters from the trained BatchNorm layer
                # y = gamma * (x - mean) / sqrt(var + eps) + beta
                # y_scaled = scaling_factor * y
                # y_scaled = (scaling_factor * gamma / std) * x + scaling_factor * (beta - mean * gamma / std)
                # Let fused_weight = (scaling_factor * gamma / std)
                # Let fused_bias = scaling_factor * (beta - mean * gamma / std)
                std = torch.sqrt(self.bn.running_var + self.bn.eps)
                gamma = self.bn.weight
                beta = self.bn.bias
                mean = self.bn.running_mean
                
                self.fused_weight = (self.scaling_factor * gamma / std).contiguous()
                self.fused_bias = (self.scaling_factor * (beta - mean * gamma / std)).contiguous()

            # Call the custom CUDA kernel
            x = self.fused_op.fused_bn_scale_cuda(x, self.fused_weight, self.fused_bias)

        return x