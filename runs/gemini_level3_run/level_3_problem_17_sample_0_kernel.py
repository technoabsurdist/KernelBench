import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Fused ReLU + Concatenation
# This kernel takes the raw outputs of two convolution layers,
# applies ReLU to each element, and writes the result into a
# single, concatenated output tensor. This fuses three operations
# (ReLU, ReLU, cat) into a single kernel launch.
fused_expand_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_expand_relu_cat_kernel(
    const float* in1,
    const float* in2,
    float* out,
    int N, int C1, int C2, int H, int W) {

    int total_size = N * (C1 + C2) * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_size) {
        return;
    }

    // Deconstruct the 1D output index 'idx' into 4D coordinates (n, c_out, h, w)
    int C_total = C1 + C2;
    int HW = H * W;
    int CHW_total = C_total * HW;

    int n = idx / CHW_total;
    int remainder = idx % CHW_total;
    int c_out = remainder / HW;
    remainder = remainder % HW;
    int h = remainder / W;
    int w = remainder % W;

    float val;

    if (c_out < C1) {
        // This thread corresponds to a location in the first input tensor
        int c_in1 = c_out;
        int in1_idx = n * (C1 * HW) + c_in1 * HW + h * W + w;
        val = in1[in1_idx];
    } else {
        // This thread corresponds to a location in the second input tensor
        int c_in2 = c_out - C1;
        int in2_idx = n * (C2 * HW) + c_in2 * HW + h * W + w;
        val = in2[in2_idx];
    }

    // Apply ReLU and write to the output tensor
    out[idx] = fmaxf(0.f, val);
}

torch::Tensor fused_expand_relu_cat_cuda(torch::Tensor in1, torch::Tensor in2) {
    TORCH_CHECK(in1.is_cuda(), "Input tensor 1 must be a CUDA tensor");
    TORCH_CHECK(in2.is_cuda(), "Input tensor 2 must be a CUDA tensor");
    
    // Ensure inputs are contiguous for direct memory access
    auto in1_c = in1.contiguous();
    auto in2_c = in2.contiguous();

    TORCH_CHECK(in1_c.dim() == 4, "Input tensor 1 must be 4D");
    TORCH_CHECK(in2_c.dim() == 4, "Input tensor 2 must be 4D");
    TORCH_CHECK(in1_c.size(0) == in2_c.size(0) && in1_c.size(2) == in2_c.size(2) && in1_c.size(3) == in2_c.size(3),
                "Input tensors must have the same batch size, height, and width");

    const auto N = in1_c.size(0);
    const auto C1 = in1_c.size(1);
    const auto H = in1_c.size(2);
    const auto W = in1_c.size(3);
    const auto C2 = in2_c.size(1);

    auto out = torch::empty({N, C1 + C2, H, W}, in1_c.options());

    const int total_size = out.numel();
    if (total_size == 0) {
        return out;
    }
    
    const int block_size = 1024;
    const int num_blocks = (total_size + block_size - 1) / block_size;

    fused_expand_relu_cat_kernel<<<num_blocks, block_size>>>(
        in1_c.data_ptr<float>(),
        in2_c.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C1, C2, H, W
    );
    
    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
"""

fused_expand_cpp_source = (
    "torch::Tensor fused_expand_relu_cat_cuda(torch::Tensor in1, torch::Tensor in2);"
)

# Compile the inline CUDA code. This is done once at module load time.
fused_expand_op = load_inline(
    name="fused_expand_op",
    cpp_sources=fused_expand_cpp_source,
    cuda_sources=fused_expand_source,
    functions=["fused_expand_relu_cat_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        """
        :param in_channels: Number of input channels
        :param squeeze_channels: Number of output channels for the squeeze layer
        :param expand1x1_channels: Number of output channels for the 1x1 expand layer
        :param expand3x3_channels: Number of output channels for the 3x3 expand layer
        """
        super(ModelNew, self).__init__()
        
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        # The activations for the expand layers and the final concatenation
        # are now handled by our custom fused CUDA kernel.
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)

        self.fused_expand_op = fused_expand_op.fused_expand_relu_cat_cuda

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, expand1x1_channels + expand3x3_channels, height, width)
        """
        x = self.squeeze_activation(self.squeeze(x))
        
        # Get the outputs of the convolutions *before* activation
        expand1x1_out = self.expand1x1(x)
        expand3x3_out = self.expand3x3(x)
        
        # Apply the fused ReLU and concatenation kernel
        return self.fused_expand_op(expand1x1_out, expand3x3_out)