import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for ConvTranspose2d
conv_transpose_2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

// Helper for integer division that rounds up
#define DIV_UP(a, b) (((a) + (b) - 1) / (b))

__global__ void conv_transpose2d_kernel_forward(
    const float* input,
    const float* weight,
    float* output,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int kH, const int kW,
    const int H_out, const int W_out,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dil_h, const int dil_w
) {
    // Using an input-centric approach (scatter)
    // Each thread handles one element from the input tensor
    int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_input_elements = (int64_t)N * C_in * H_in * W_in;

    if (index >= total_input_elements) {
        return;
    }

    // Deconstruct the 1D index to 4D input coordinates (n, c_in, h_in, w_in)
    const int w_in = index % W_in;
    const int h_in = (index / W_in) % H_in;
    const int c_in = (index / (W_in * H_in)) % C_in;
    const int n = index / (W_in * H_in * C_in);

    const float input_val = input[index];

    // Iterate over the output channels and the kernel
    for (int c_out = 0; c_out < C_out; ++c_out) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                // Calculate the corresponding output coordinates
                const int h_out = h_in * stride_h - pad_h + kh * dil_h;
                const int w_out = w_in * stride_w - pad_w + kw * dil_w;

                // Check if the output coordinates are within bounds
                if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out) {
                    // Get the weight value
                    // Weight layout: (C_in, C_out, kH, kW) for groups=1
                    const int64_t weight_idx = (int64_t)c_in * C_out * kH * kW +
                                               (int64_t)c_out * kH * kW +
                                               (int64_t)kh * kW +
                                               kw;
                    const float weight_val = weight[weight_idx];

                    // Calculate the output index
                    const int64_t output_idx = (int64_t)n * C_out * H_out * W_out +
                                               (int64_t)c_out * H_out * W_out +
                                               (int64_t)h_out * W_out +
                                               w_out;

                    // Atomically add the contribution to the output tensor
                    // This is necessary because multiple input threads can write to the same output location
                    atomicAdd(&output[output_idx], input_val * weight_val);
                }
            }
        }
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w,
    int dil_h, int dil_w
) {
    // Ensure tensors are on CUDA and contiguous
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    input = input.contiguous();
    weight = weight.contiguous();

    // Get dimensions
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    // Weight shape is (C_in, C_out, kH, kW) for groups=1
    TORCH_CHECK(weight.size(0) == C_in, "Weight C_in mismatch");
    const int C_out = weight.size(1);
    const int kH = weight.size(2);
    const int kW = weight.size(3);

    // Calculate output dimensions
    const int H_out = (H_in - 1) * stride_h - 2 * pad_h + dil_h * (kH - 1) + out_pad_h + 1;
    const int W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (kW - 1) + out_pad_w + 1;

    // Create output tensor, initialized to zeros
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    // Kernel launch configuration
    const int64_t total_input_elements = input.numel();
    if (total_input_elements == 0) {
        return output;
    }
    const int block_size = 256;
    const int grid_size = DIV_UP(total_input_elements, (int64_t)block_size);

    // Launch the kernel
    conv_transpose2d_kernel_forward<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, kH, kW,
        H_out, W_out,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w
    );
    
    // Check for any CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

conv_transpose_2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w,
    int dil_h, int dil_w
);
"""

# Compile the inline CUDA code
# This is done once when the module is imported.
custom_conv_transpose_2d = load_inline(
    name="custom_conv_transpose_2d",
    cpp_sources=conv_transpose_2d_cpp_source,
    cuda_sources=conv_transpose_2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution operation with a custom CUDA kernel.
    This implementation assumes groups=1 and bias=False.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of integers representing the kernel size (height, width).
        stride (tuple, optional): Tuple of integers representing the stride of the convolution. Defaults to (1, 1).
        padding (tuple, optional): Tuple of integers representing the padding applied to the input. Defaults to (0, 0).
        output_padding (tuple, optional): Tuple of integers representing the additional size added to one side of the output shape. Defaults to (0, 0).
        dilation (tuple, optional): Tuple of integers representing the spacing between kernel elements. Defaults to (1, 1).
        groups (int, optional): Must be 1 for this custom implementation.
        bias (bool, optional): Must be False for this custom implementation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        if groups != 1:
            raise NotImplementedError("Custom CUDA kernel only supports groups=1")
        if bias:
            raise NotImplementedError("Custom CUDA kernel does not support bias")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        
        # The weight tensor for ConvTranspose2d has shape (in_channels, out_channels, kH, kW) for groups=1
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, *kernel_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Standard PyTorch initialization for Conv layers
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return custom_conv_transpose_2d.conv_transpose2d_cuda(
            x, self.weight,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.output_padding[0], self.output_padding[1],
            self.dilation[0], self.dilation[1]
        )