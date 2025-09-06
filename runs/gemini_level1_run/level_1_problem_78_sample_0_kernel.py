import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for 2D transposed convolution
conv_transpose2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int N,
    const int C_in,
    const int H_in,
    const int W_in,
    const int C_out,
    const int H_out,
    const int W_out,
    const int kH,
    const int kW,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w
) {
    // Each thread computes one output element (n, c_out, h_out, w_out)
    // Using a 2D block and 3D grid
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int c_out = blockIdx.z % C_out;
    const int n = blockIdx.z / C_out;

    // Boundary check to avoid writing out of bounds
    if (n >= N || c_out >= C_out || h_out >= H_out || w_out >= W_out) {
        return;
    }

    float sum = 0.0f;

    // Iterate over input channels
    for (int c_in = 0; c_in < C_in; ++c_in) {
        // Iterate over the kernel height and width
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                // Calculate the corresponding input coordinates for this kernel tap
                int h_in_numerator = h_out + pad_h - kh;
                int w_in_numerator = w_out + pad_w - kw;

                // Check if this kernel tap contributes to the current output pixel.
                // This is the core condition for transposed convolution.
                if (h_in_numerator >= 0 && h_in_numerator % stride_h == 0 &&
                    w_in_numerator >= 0 && w_in_numerator % stride_w == 0) {

                    int h_in = h_in_numerator / stride_h;
                    int w_in = w_in_numerator / stride_w;

                    // Check if the calculated input coordinates are within the input tensor bounds
                    if (h_in < H_in && w_in < W_in) {
                        // Calculate flattened indices for input and weight tensors
                        // Input tensor format: (N, C_in, H_in, W_in)
                        long long input_idx = n * C_in * H_in * W_in +
                                              c_in * H_in * W_in +
                                              h_in * W_in +
                                              w_in;

                        // Weight tensor format: (C_in, C_out, kH, kW)
                        long long weight_idx = c_in * C_out * kH * kW +
                                               c_out * kH * kW +
                                               kh * kW +
                                               kw;

                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Add bias if it exists
    if (bias != nullptr) {
        sum += bias[c_out];
    }

    // Calculate the flattened index for the output tensor
    long long output_idx = n * C_out * H_out * W_out +
                           c_out * H_out * W_out +
                           h_out * W_out +
                           w_out;

    output[output_idx] = sum;
}

torch::Tensor conv_transpose2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<long> stride,
    std::vector<long> padding
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "Weight must be a float32 tensor");
    TORCH_CHECK(input.size(1) == weight.size(0), "Input channels must match weight's in_channels");

    // Get dimensions from input tensors
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    const int C_out = weight.size(1);
    const int kH = weight.size(2);
    const int kW = weight.size(3);

    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int pad_h = padding[0];
    const int pad_w = padding[1];

    // Calculate output dimensions using the standard formula for transposed convolution
    const int H_out = (H_in - 1) * stride_h - 2 * pad_h + kH;
    const int W_out = (W_in - 1) * stride_w - 2 * pad_w + kW;

    // Create the output tensor, initialized to zeros
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    // Setup grid and block dimensions for the CUDA kernel
    const int threads_x = 16;
    const int threads_y = 16;
    dim3 threads(threads_x, threads_y);

    const int blocks_x = (W_out + threads_x - 1) / threads_x;
    const int blocks_y = (H_out + threads_y - 1) / threads_y;
    const int blocks_z = N * C_out;
    dim3 blocks(blocks_x, blocks_y, blocks_z);

    // Get raw data pointers
    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    const float* bias_ptr = nullptr;
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C_out, "Bias must have size C_out");
        TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "Bias must be a float32 tensor");
        bias_ptr = bias.data_ptr<float>();
    }

    // Launch the CUDA kernel
    conv_transpose2d_kernel<<<blocks, threads>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        N, C_in, H_in, W_in, C_out, H_out, W_out,
        kH, kW, stride_h, stride_w, pad_h, pad_w
    );

    // Check for any CUDA errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<long> stride,
    std::vector<long> padding
);
"""

# JIT compile the custom CUDA operator
# This is done once when the module is imported.
conv_transpose2d_cuda_op = load_inline(
    name="conv_transpose2d_cuda",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_cuda_source,
    functions=["conv_transpose2d_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Performs a 2D transposed convolution operation using a custom CUDA kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (tuple, optional): Stride of the convolution (height, width). Defaults to (1, 1).
        padding (tuple, optional): Padding applied to the input (height, width). Defaults to (0, 0).
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Define learnable parameters, matching PyTorch's ConvTranspose2d weight shape
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize weights and bias using a method similar to PyTorch's default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Handle the optional bias. If it's None, pass an empty tensor to the C++ extension.
        bias_tensor = self.bias if self.bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        
        return conv_transpose2d_cuda_op.conv_transpose2d_forward_cuda(
            x, self.weight, bias_tensor, list(self.stride), list(self.padding)
        )