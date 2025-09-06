import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for 2D Average Pooling
avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool2d_kernel(
    const float* input,
    float* output,
    const int batch_size,
    const int channels,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w,
    const int kernel_size,
    const int stride,
    const int padding) {

    // Calculate the global thread index for the output tensor
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int bc_idx = blockIdx.z * blockDim.z + threadIdx.z; // Combined batch and channel index

    // Check if the thread is within the output tensor bounds
    if (ow >= out_w || oh >= out_h || bc_idx >= batch_size * channels) {
        return;
    }

    // Decompose the combined index into batch and channel
    int b = bc_idx / channels;
    int c = bc_idx % channels;

    // Calculate the top-left corner of the pooling window in the input tensor
    int start_h = oh * stride - padding;
    int start_w = ow * stride - padding;

    float sum = 0.0f;
    int count = 0;

    // Iterate over the kernel window
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h = start_h + kh;
            int w = start_w + kw;

            // Check if the current position is within the valid input bounds (not padding)
            if (h >= 0 && h < in_h && w >= 0 && w < in_w) {
                // Calculate the linear index for the input tensor (NCHW format)
                int input_idx = b * channels * in_h * in_w +
                                c * in_h * in_w +
                                h * in_w +
                                w;
                sum += input[input_idx];
                count++;
            }
        }
    }

    // Calculate the average. Handle the case where the window is entirely in the padding.
    float avg = (count > 0) ? sum / count : 0.0f;

    // Calculate the linear index for the output tensor (NCHW format)
    int output_idx = b * channels * out_h * out_w +
                     c * out_h * out_w +
                     oh * out_w +
                     ow;

    output[output_idx] = avg;
}

torch::Tensor avg_pool2d_cuda(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding) {

    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (NCHW)");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");

    // Get input dimensions
    const int batch_size = x.size(0);
    const int channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);

    // Calculate output dimensions
    const int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    const int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    // Create the output tensor
    auto out = torch::zeros({batch_size, channels, out_h, out_w}, x.options());

    // Define grid and block dimensions for the CUDA kernel
    // Each thread computes one output pixel. The grid is 3D to cover (out_w, out_h, batch*channels)
    dim3 block_dim(16, 16, 1);
    dim3 grid_dim(
        (out_w + block_dim.x - 1) / block_dim.x,
        (out_h + block_dim.y - 1) / block_dim.y,
        (batch_size * channels + block_dim.z - 1) / block_dim.z
    );

    // Launch the kernel
    avg_pool2d_kernel<<<grid_dim, block_dim>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        channels,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_size,
        stride,
        padding
    );

    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

avg_pool_cpp_source = (
    "torch::Tensor avg_pool2d_cuda(torch::Tensor x, int kernel_size, int stride, int padding);"
)

# Compile the inline CUDA code for 2D Average Pooling
# This is done at the module level to avoid recompilation on every model instantiation
avg_pool_2d_custom = load_inline(
    name="avg_pool_2d_custom",
    cpp_sources=avg_pool_cpp_source,
    cuda_sources=avg_pool_source,
    functions=["avg_pool2d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Simple model that performs 2D Average Pooling using a custom CUDA kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the custom Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies custom 2D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return avg_pool_2d_custom.avg_pool2d_cuda(x, self.kernel_size, self.stride, self.padding)

batch_size = 16
channels = 64
height = 2048
width = 2048
kernel_size = 11

def get_inputs():
    x = torch.rand(batch_size, channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size]