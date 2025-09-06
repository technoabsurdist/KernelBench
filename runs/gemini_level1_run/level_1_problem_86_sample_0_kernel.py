import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math
from torch.nn import init

# Define the custom CUDA kernel for fused depthwise-separable convolution
fused_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_depthwise_pointwise_conv_kernel(
    const float* __restrict__ x,
    const float* __restrict__ depthwise_weight,
    const float* __restrict__ pointwise_weight,
    float* __restrict__ out,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int H_out, const int W_out,
    const int K, const int stride, const int padding, const int dilation
) {
    // Calculate global thread indices for the output tensor.
    // Each thread computes one output element (n, c_out, h_out, w_out).
    // The grid is structured so that each block processes a 2D tile of the output
    // for a specific batch element 'n' and output channel 'c_out'.
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int n_c_out_idx = blockIdx.z;

    // Decompose the z-index into batch and output channel indices
    const int n = n_c_out_idx / C_out;
    const int c_out = n_c_out_idx % C_out;

    // Boundary check to ensure threads do not write out of the output tensor's bounds
    if (w_out >= W_out || h_out >= H_out || n >= N) {
        return;
    }

    float acc = 0.0f;

    // This loop fuses the two convolutions.
    // It iterates over input channels to compute the dot product for the pointwise convolution.
    // Inside the loop, the value from the depthwise convolution is computed on-the-fly.
    for (int c_in = 0; c_in < C_in; ++c_in) {
        float depthwise_val = 0.0f;

        // Perform depthwise convolution for the current input channel (c_in)
        // at the spatial location (h_out, w_out).
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                // Calculate corresponding input coordinates based on stride, padding, and dilation
                const int h_in = h_out * stride - padding + kh * dilation;
                const int w_in = w_out * stride - padding + kw * dilation;

                // Check if the calculated input coordinates are within the valid input tensor bounds
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    // Calculate flat indices for input and depthwise weight tensors.
                    // Using long long to prevent potential integer overflow with large tensors.
                    const long long x_idx = (long long)n * C_in * H_in * W_in +
                                            (long long)c_in * H_in * W_in +
                                            (long long)h_in * W_in +
                                            w_in;
                    // The depthwise weight has shape (C_in, 1, K, K), but the '1' dimension
                    // doesn't affect the contiguous memory layout.
                    const int dw_idx = c_in * K * K + kh * K + kw;

                    depthwise_val += x[x_idx] * depthwise_weight[dw_idx];
                }
            }
        }

        // Perform the pointwise multiplication (1x1 convolution) and accumulate the result.
        // The pointwise weight has shape (C_out, C_in, 1, 1).
        const int pw_idx = c_out * C_in + c_in;
        acc += depthwise_val * pointwise_weight[pw_idx];
    }

    // Write the final computed value to the output tensor
    const long long out_idx = (long long)n * C_out * H_out * W_out +
                              (long long)c_out * H_out * W_out +
                              (long long)h_out * W_out +
                              w_out;
    out[out_idx] = acc;
}

// C++ wrapper function to be called from Python
torch::Tensor fused_conv_cuda(
    torch::Tensor x,
    torch::Tensor depthwise_weight,
    torch::Tensor pointwise_weight,
    int stride, int padding, int dilation
) {
    // Input validation checks
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(depthwise_weight.is_cuda(), "depthwise_weight must be a CUDA tensor");
    TORCH_CHECK(pointwise_weight.is_cuda(), "pointwise_weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(depthwise_weight.is_contiguous(), "depthwise_weight must be contiguous");
    TORCH_CHECK(pointwise_weight.is_contiguous(), "pointwise_weight must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input x must be a 4D tensor");

    // Get dimensions from input tensors
    const int N = x.size(0);
    const int C_in = x.size(1);
    const int H_in = x.size(2);
    const int W_in = x.size(3);

    const int C_out = pointwise_weight.size(0);
    const int K = depthwise_weight.size(2);

    // Calculate output dimensions using the standard convolution formula
    const int H_out = (H_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    const int W_out = (W_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    // Create an empty output tensor with the correct shape and on the same device as the input
    auto out = torch::empty({N, C_out, H_out, W_out}, x.options());

    // Configure kernel launch parameters
    const int TILE_DIM = 16; // A common tile size for 2D kernels
    const dim3 threads(TILE_DIM, TILE_DIM, 1);
    const dim3 blocks(
        (W_out + TILE_DIM - 1) / TILE_DIM,
        (H_out + TILE_DIM - 1) / TILE_DIM,
        (unsigned int)(N * C_out)
    );

    // Launch the CUDA kernel
    fused_depthwise_pointwise_conv_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        depthwise_weight.data_ptr<float>(),
        pointwise_weight.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K, stride, padding, dilation
    );
    
    // Check for any CUDA errors after kernel launch for debugging
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return out;
}
"""

fused_conv_cpp_source = """
torch::Tensor fused_conv_cuda(
    torch::Tensor x,
    torch::Tensor depthwise_weight,
    torch::Tensor pointwise_weight,
    int stride, int padding, int dilation
);
"""

# JIT compile the C++/CUDA code
fused_conv_module = load_inline(
    name="fused_conv",
    cpp_sources=fused_conv_cpp_source,
    cuda_sources=fused_conv_source,
    functions=["fused_conv_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation using a custom fused CUDA kernel.
    This implementation fuses the depthwise and pointwise convolutions into a single kernel
    to reduce memory bandwidth and kernel launch overhead.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
                               Note: The custom kernel does not currently support bias.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        if bias:
            raise NotImplementedError("Bias is not supported in this custom kernel.")
        if fused_conv_module is None:
            raise RuntimeError("CUDA module for fused convolution could not be loaded.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Define the weights as learnable parameters, matching the shapes used by nn.Conv2d
        # Depthwise weights: (in_channels, 1, kernel_size, kernel_size)
        # Pointwise weights: (out_channels, in_channels, 1, 1)
        self.depthwise_weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        self.pointwise_weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1))
        
        self.reset_parameters()
        
        # Store the compiled CUDA function for use in the forward pass
        self.fused_conv = fused_conv_module.fused_conv_cuda

    def reset_parameters(self) -> None:
        # Initialize weights using the same method as PyTorch's Conv2d for consistency
        init.kaiming_uniform_(self.depthwise_weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.pointwise_weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the fused depthwise-separable 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.fused_conv(
            x,
            self.depthwise_weight,
            self.pointwise_weight,
            self.stride,
            self.padding,
            self.dilation
        )