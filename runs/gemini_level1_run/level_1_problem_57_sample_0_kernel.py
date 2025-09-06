import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ConvTranspose2d
conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <optional>

// CUDA kernel for the forward pass of 2D transposed convolution
__global__ void conv_transpose2d_forward_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int stride, int padding, int groups
) {
    // Each thread computes one output pixel (n, c_out, h_out, w_out)
    // Grid is 3D: (W_out, H_out, N * C_out)
    // Block is 2D: (TILE_W, TILE_H)
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int n_cout_plane = blockIdx.z;

    int n = n_cout_plane / C_out;
    int c_out = n_cout_plane % C_out;

    // Boundary check
    if (w_out >= W_out || h_out >= H_out || n >= N) {
        return;
    }

    // Determine the group for this output channel
    const int C_out_per_group = C_out / groups;
    const int C_in_per_group = C_in / groups;
    int g = c_out / C_out_per_group;

    // Input channel range for this group
    int c_in_start = g * C_in_per_group;
    int c_in_end = c_in_start + C_in_per_group;

    // Relative output channel index within the group
    int c_out_g = c_out % C_out_per_group;

    float sum = 0.0f;

    // This is a "gather" operation. For each output pixel, we iterate through
    // the kernel and the relevant input channels to gather contributions.
    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                // Map output coordinates to input coordinates
                int h_in_numerator = h_out + padding - kh;
                int w_in_numerator = w_out + padding - kw;

                // Check if the mapping is valid (integer division)
                if (h_in_numerator >= 0 && h_in_numerator % stride == 0 &&
                    w_in_numerator >= 0 && w_in_numerator % stride == 0) {

                    int h_in = h_in_numerator / stride;
                    int w_in = w_in_numerator / stride;

                    // Check if the input coordinates are within bounds
                    if (h_in < H_in && w_in < W_in) {
                        // Input index
                        long long input_idx = n * C_in * H_in * W_in +
                                              c_in * H_in * W_in +
                                              h_in * W_in +
                                              w_in;

                        // Weight index
                        // Weight shape: (C_in, C_out/groups, K, K)
                        long long weight_idx = c_in * C_out_per_group * K * K +
                                               c_out_g * K * K +
                                               kh * K +
                                               kw;

                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Calculate output index
    long long output_idx = n * C_out * H_out * W_out +
                           c_out * H_out * W_out +
                           h_out * W_out +
                           w_out;

    // Add bias if it exists
    if (bias != nullptr) {
        sum += bias[c_out];
    }

    output[output_idx] = sum;
}

// C++ wrapper function that launches the CUDA kernel
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias_opt,
    int stride,
    int padding,
    int output_padding,
    int groups
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be a CUDA tensor");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "Bias tensor must be a CUDA tensor");
    }
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "Weight must be a float32 tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor");

    // Get dimensions from input tensors
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    // Weight shape is (C_in, C_out/groups, K, K)
    TORCH_CHECK(weight.size(0) == C_in, "Weight in_channels mismatch");
    const int C_out = weight.size(1) * groups;
    const int K = weight.size(2);
    TORCH_CHECK(weight.size(3) == K, "Kernel must be square");

    // Calculate output shape using the formula for transposed convolution
    const int H_out = (H_in - 1) * stride - 2 * padding + K + output_padding;
    const int W_out = (W_in - 1) * stride - 2 * padding + K + output_padding;

    TORCH_CHECK(H_out > 0 && W_out > 0, "Invalid output dimensions calculated. Check stride, padding, and kernel size.");

    // Create the output tensor, initialized to zeros
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    // Get a raw pointer to the bias tensor, or nullptr if it doesn't exist
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value();
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C_out, "Bias shape mismatch");
        bias_ptr = bias.data_ptr<float>();
    }

    // Configure and launch the kernel
    const dim3 block_size(16, 16, 1); // 256 threads per block
    const dim3 grid_size(
        (W_out + block_size.x - 1) / block_size.x,
        (H_out + block_size.y - 1) / block_size.y,
        (long long)N * C_out
    );

    conv_transpose2d_forward_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K, stride, padding, groups
    );

    // Check for any CUDA errors during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

conv_transpose_cpp_source = """
#include <torch/extension.h>
#include <optional>

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias_opt,
    int stride,
    int padding,
    int output_padding,
    int groups
);
"""

# Compile the inline CUDA code
custom_conv_transpose = load_inline(
    name="custom_conv_transpose",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    A replacement for nn.ConvTranspose2d that uses a custom CUDA kernel.
    This module maintains its own weight and bias parameters, making it a
    trainable, drop-in replacement.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        # Define weight and bias as learnable parameters, same as nn.ConvTranspose2d
        self.weight = nn.Parameter(torch.Tensor(
            in_channels, out_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Use standard PyTorch initialization for compatibility
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using the custom CUDA kernel.
        """
        return custom_conv_transpose.conv_transpose2d_cuda(
            x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups
        )