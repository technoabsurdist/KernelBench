import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and its C++ wrapper for ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA Kernel for ConvTranspose2d forward pass (NCHW format)
// This kernel uses a "gather" approach, where each thread is responsible for computing a single output element.
__global__ void conv_transpose2d_forward_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels, int H_in, int W_in,
    int out_channels, int H_out, int W_out,
    int kernel_size, int stride, int padding) {

    // Calculate the global thread index for the output tensor's spatial dimensions (x, y)
    // and the combined batch/channel dimension (z).
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    int z_idx = blockIdx.z * blockDim.z + threadIdx.z;

    // The z-dimension of the grid maps to batch and output channels
    if (z_idx >= batch_size * out_channels) return;
    
    int k = z_idx % out_channels; // output channel index
    int n = z_idx / out_channels; // batch index

    // Boundary check for spatial dimensions
    if (x_out >= W_out || y_out >= H_out) {
        return;
    }

    float sum = 0.0f;

    // Iterate over input channels and the kernel to accumulate contributions
    for (int c = 0; c < in_channels; ++c) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                // This is the core logic of transposed convolution. We map an output pixel (y_out, x_out)
                // back to a potential source input pixel (y_in, x_in).
                int y_in_numerator = y_out + padding - ky;
                int x_in_numerator = x_out + padding - kx;

                // An input pixel contributes only if the mapping is exact (i.e., divisible by stride)
                if (y_in_numerator >= 0 && y_in_numerator % stride == 0 &&
                    x_in_numerator >= 0 && x_in_numerator % stride == 0) {

                    int y_in = y_in_numerator / stride;
                    int x_in = x_in_numerator / stride;

                    // Check if the calculated input coordinates are within bounds
                    if (y_in < H_in && x_in < W_in) {
                        // Calculate linear index for the input tensor
                        long long input_idx = (long long)n * in_channels * H_in * W_in +
                                              (long long)c * H_in * W_in +
                                              (long long)y_in * W_in +
                                              x_in;

                        // Calculate linear index for the weight tensor
                        // PyTorch weight for ConvTranspose2d is (in_channels, out_channels/groups, kH, kW)
                        // For groups=1, it's (in_channels, out_channels, kH, kW)
                        long long weight_idx = (long long)c * out_channels * kernel_size * kernel_size +
                                               (long long)k * kernel_size * kernel_size +
                                               (long long)ky * kernel_size +
                                               kx;

                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Calculate linear index for the output tensor and write the result
    long long output_idx = (long long)n * out_channels * H_out * W_out +
                           (long long)k * H_out * W_out +
                           (long long)y_out * W_out +
                           x_out;
    output[output_idx] = sum;
}

// C++ wrapper function to be called from PyTorch
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding) {

    // Ensure inputs are on the correct device and have a contiguous memory layout
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    input = input.contiguous();
    weight = weight.contiguous();

    // Extract dimensions from input tensors
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto H_in = input.size(2);
    const auto W_in = input.size(3);

    // Weight dimensions: (in_channels, out_channels, kH, kW)
    TORCH_CHECK(weight.size(0) == in_channels, "Weight in_channels mismatch");
    const auto out_channels = weight.size(1);
    const auto kernel_size_H = weight.size(2);
    const auto kernel_size_W = weight.size(3);
    TORCH_CHECK(kernel_size_H == kernel_size_W, "Custom kernel only supports square kernels");
    const auto kernel_size = kernel_size_H;

    // Calculate output dimensions using the standard formula for transposed convolution
    const long H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    const long W_out = (W_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    // Create an empty output tensor
    auto output = torch::zeros({batch_size, out_channels, H_out, W_out}, input.options());

    // Configure the CUDA kernel launch parameters
    dim3 threads_per_block(16, 16, 1);
    dim3 num_blocks(
        (W_out + threads_per_block.x - 1) / threads_per_block.x,
        (H_out + threads_per_block.y - 1) / threads_per_block.y,
        (batch_size * out_channels + threads_per_block.z - 1) / threads_per_block.z
    );

    // Launch the kernel
    conv_transpose2d_forward_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels, H_in, W_in,
        out_channels, H_out, W_out,
        kernel_size, stride, padding
    );

    return output;
}
"""

# Define the C++ source for the function signature
conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int output_padding);
"""

# Use load_inline to JIT compile the CUDA/C++ code
custom_conv_transpose2d_op = load_inline(
    name="custom_conv_transpose2d_op",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
)


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution using a custom CUDA kernel.
    This implementation replaces nn.ConvTranspose2d for the forward pass.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Must be 1 for this custom implementation.
        bias (bool, optional): Must be False for this custom implementation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # The custom kernel has limitations, so we check them here.
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
        
        # Manually define the weight parameter, similar to how a standard nn.Module would.
        # The shape for ConvTranspose2d weight is (in_channels, out_channels, kH, kW).
        self.weight = nn.Parameter(torch.Tensor(
            in_channels, out_channels, kernel_size, kernel_size
        ))
        
        # Initialize the weight parameter using a standard method.
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Use Kaiming uniform initialization, which is the default for nn.Conv2d/ConvTranspose2d.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the custom transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Call the JIT-compiled CUDA function.
        return custom_conv_transpose2d_op.conv_transpose2d_cuda(
            x, self.weight, self.stride, self.padding, self.output_padding
        )