import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel and C++ wrapper for 2D convolution
conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_forward_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int N, int C_in, int H, int W,
    int C_out, int K,
    int H_out, int W_out,
    int stride, int padding, int dilation, int groups) {

    // Using a 2D grid of thread blocks and 2D thread blocks.
    // Each thread computes one output element.
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    // blockIdx.z handles batch and output channel
    const int c_out_n_combo = blockIdx.z;

    if (h_out >= H_out || w_out >= W_out) {
        return;
    }

    const int n = c_out_n_combo / C_out;
    const int c_out = c_out_n_combo % C_out;

    const int C_in_per_group = C_in / groups;
    const int group_idx = c_out / (C_out / groups);
    const int c_in_start = group_idx * C_in_per_group;
    const int c_in_end = c_in_start + C_in_per_group;

    float acc = 0.0f;

    // Iterate over input channels for the current group
    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        // Iterate over the kernel
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                const int h_in = h_out * stride - padding + kh * dilation;
                const int w_in = w_out * stride - padding + kw * dilation;

                // Boundary check for padding
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    long x_idx = n * C_in * H * W + c_in * H * W + h_in * W + w_in;
                    // The weight tensor is indexed by (c_out, c_in_group, kh, kw)
                    long w_idx = c_out * C_in_per_group * K * K + (c_in - c_in_start) * K * K + kh * K + kw;
                    acc += x[x_idx] * weight[w_idx];
                }
            }
        }
    }

    // Add bias if it exists
    if (bias != nullptr) {
        acc += bias[c_out];
    }

    // Write the result to the output tensor
    long out_idx = n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out;
    out[out_idx] = acc;
}

torch::Tensor conv2d_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "Weight tensor must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias tensor must be a CUDA tensor");
        TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "Bias tensor must be float32");
    }

    const int N = x.size(0);
    const int C_in = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);

    const int C_out = weight.size(0);
    const int K = weight.size(2); // Assuming square kernel

    const int H_out = (H + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    const int W_out = (W + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    auto out = torch::zeros({N, C_out, H_out, W_out}, x.options());

    const dim3 threads(16, 16);
    const dim3 blocks(
        (W_out + threads.x - 1) / threads.x,
        (H_out + threads.y - 1) / threads.y,
        (unsigned int)(N * C_out)
    );

    conv2d_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        N, C_in, H, W,
        C_out, K,
        H_out, W_out,
        stride, padding, dilation, groups
    );
    
    // Check for errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
"""

conv2d_cpp_source = """
torch::Tensor conv2d_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups);
"""

# Compile the inline CUDA code for the custom convolution
custom_conv2d = load_inline(
    name="custom_conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation using a custom CUDA kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Define learnable parameters, similar to nn.Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize weights and bias using standard methods
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return custom_conv2d.conv2d_forward_cuda(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )