import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for 1D convolution
conv1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

__global__ void conv1d_forward_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int length,
    const int kernel_size,
    const int length_out,
    const int stride,
    const int dilation,
    const bool has_bias) {

    // Each thread computes one output element
    // Grid is structured as (blocks for length_out, out_channels, batch_size)
    const int b = blockIdx.z;
    const int oc = blockIdx.y;
    const int l_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (l_out >= length_out) {
        return;
    }

    float acc = 0.0f;

    // Pre-calculate base offsets for the current output element to reduce redundant calculations
    const int out_base_idx = (b * out_channels + oc) * length_out;
    const int x_batch_offset = b * in_channels * length;
    const int weight_out_channel_offset = oc * in_channels * kernel_size;

    for (int ic = 0; ic < in_channels; ++ic) {
        const int x_channel_offset = x_batch_offset + ic * length;
        const int weight_in_channel_offset = weight_out_channel_offset + ic * kernel_size;
        for (int k = 0; k < kernel_size; ++k) {
            const int l_in = l_out * stride + k * dilation;
            // Note: No input boundary check is needed here because the output length `length_out`
            // is calculated such that the maximum `l_in` will always be within the input bounds.
            acc += x[x_channel_offset + l_in] * weight[weight_in_channel_offset + k];
        }
    }

    if (has_bias) {
        acc += bias[oc];
    }

    out[out_base_idx + l_out] = acc;
}

torch::Tensor conv1d_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int dilation) {

    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Input weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input x must be a float32 tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "Input weight must be a float32 tensor");
    
    // Ensure tensors are contiguous for predictable memory layout
    x = x.contiguous();
    weight = weight.contiguous();

    // Get dimensions from input tensors
    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto length = x.size(2);

    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2);

    // Calculate output length based on convolution parameters
    const int length_out = (length - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(length_out > 0, "Output length must be positive, but got ", length_out);

    // Create the output tensor
    auto out = torch::zeros({batch_size, out_channels, length_out}, x.options());

    // Handle optional bias tensor
    const bool has_bias = bias.defined();
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "Input bias must be a CUDA tensor");
        TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "Input bias must be a float32 tensor");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == out_channels, "Bias must be a 1D tensor of size out_channels");
        bias = bias.contiguous();
    }

    // Setup grid and block dimensions for the kernel launch
    const int block_size = 256;
    const dim3 threads(block_size, 1, 1);
    const dim3 blocks((length_out + block_size - 1) / block_size, out_channels, batch_size);

    // Launch the CUDA kernel
    conv1d_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        length,
        kernel_size,
        length_out,
        stride,
        dilation,
        has_bias
    );
    
    // Check for errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# C++ source for the function signature, required by load_inline
conv1d_cpp_source = """
torch::Tensor conv1d_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int dilation);
"""

# Compile the inline CUDA code. This happens only once when the module is imported.
conv1d_op = load_inline(
    name="conv1d_op",
    cpp_sources=conv1d_cpp_source,
    cuda_sources=conv1d_source,
    functions=["conv1d_forward_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Performs a standard 1D convolution operation using a custom CUDA kernel.
    This module is designed to be a drop-in replacement for a standard nn.Conv1d layer.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        # Define learnable parameters (weight and bias) so they are managed by nn.Module
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            # A common pattern for optional parameters in nn.Module
            self.register_parameter('bias', None)
        
        # Initialize parameters to match PyTorch's default Conv1d initialization
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Mimic the initialization of nn.Conv1d for consistency
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in > 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return conv1d_op.conv1d_forward_cuda(x, self.weight, self.bias, self.stride, self.dilation)