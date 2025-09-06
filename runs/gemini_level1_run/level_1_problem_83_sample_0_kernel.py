import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA and C++ source code for the custom depthwise convolution
# This kernel is specifically optimized for a vertical (K, 1) depthwise convolution.
# It assigns one thread to compute one output pixel.
depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void depthwise_conv2d_vertical_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    const int N, const int C, const int H, const int W,
    const int K, 
    const int stride, const int padding, const int dilation,
    const int H_out, const int W_out) {

    // Calculate the global thread index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = N * C * H_out * W_out;

    if (idx >= num_threads) {
        return;
    }

    // Decompose the 1D index into 4D coordinates for the output tensor
    const int w_out = idx % W_out;
    const int h_out = (idx / W_out) % H_out;
    const int c = (idx / (W_out * H_out)) % C;
    const int n = idx / (W_out * H_out * C);

    float acc = 0.0f;

    // The kernel is (K, 1), so we iterate vertically over the kernel height
    for (int k = 0; k < K; ++k) {
        // Calculate input coordinates corresponding to the output coordinate and kernel position
        const int h_in = h_out * stride + k * dilation - padding;
        // The kernel width is 1, so k_w is always 0.
        // The original PyTorch Conv2d uses symmetric stride and padding.
        const int w_in = w_out * stride - padding;

        // Boundary checks: only accumulate if the input coordinates are within the valid range
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            // Calculate flat indices for input and weight tensors
            const int input_idx = n * C * H * W + c * H * W + h_in * W + w_in;
            // Weight tensor shape is (C, 1, K, 1). For a contiguous tensor, this simplifies.
            const int weight_idx = c * K + k;
            acc += input[input_idx] * weight[weight_idx];
        }
    }

    // Add bias if it's provided (i.e., bias pointer is not null)
    if (bias != nullptr) {
        acc += bias[c];
    }

    // Write the final result to the output tensor
    output[idx] = acc;
}

torch::Tensor depthwise_conv2d_vertical_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int stride, 
    int padding, 
    int dilation) {

    // Ensure tensors are on the correct device and contiguous for correct memory access
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    
    input = input.contiguous();
    weight = weight.contiguous();

    // Get tensor dimensions
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int K = weight.size(2); // Kernel height

    // Calculate output dimensions based on standard convolution formula
    // Note: PyTorch's Conv2d applies stride and padding symmetrically when single ints are provided
    const int H_out = (H + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    const int W_out = (W + 2 * padding - dilation * (1 - 1) - 1) / stride + 1;

    // Create the output tensor, initialized to zeros
    auto output = torch::zeros({N, C, H_out, W_out}, input.options());

    // Handle optional bias tensor
    const float* bias_ptr = nullptr;
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
        bias = bias.contiguous();
        bias_ptr = bias.data_ptr<float>();
    }

    // Set up grid and block dimensions for the CUDA kernel
    const int total_output_elements = N * C * H_out * W_out;
    if (total_output_elements == 0) {
        return output;
    }
    const int block_size = 256;
    const int num_blocks = (total_output_elements + block_size - 1) / block_size;

    // Launch the kernel
    depthwise_conv2d_vertical_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C, H, W,
        K,
        stride, padding, dilation,
        H_out, W_out
    );
    
    // Check for any CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

# C++ function signature for the JIT compiler
depthwise_conv_cpp_source = """
torch::Tensor depthwise_conv2d_vertical_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int stride, 
    int padding, 
    int dilation);
"""

# JIT compile the custom CUDA kernel using load_inline
custom_depthwise_conv = load_inline(
    name="custom_depthwise_conv",
    cpp_sources=depthwise_conv_cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv2d_vertical_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with a square input and an asymmetric kernel,
    using a custom CUDA kernel for the convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Create learnable parameters for weight and bias, similar to nn.Conv2d
        # The weight shape for depthwise convolution with a (K, 1) kernel is (in_channels, 1, K, 1)
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, 1))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            # It's important to register 'bias' as None so it's handled correctly by PyTorch
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        return custom_depthwise_conv.depthwise_conv2d_vertical_cuda(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation
        )