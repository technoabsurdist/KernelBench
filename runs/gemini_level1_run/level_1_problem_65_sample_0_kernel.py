import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and its C++ wrapper in a single string
conv_transpose2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Helper for 4D NCHW tensor access
#define TENSOR_IDX(n, c, h, w, C, H, W) (((n * C + c) * H + h) * W + w)

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int H_out, const int W_out,
    const int K_h, const int K_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w
) {
    // Each thread computes one output element (h_out, w_out) for a given (n, c_out) pair.
    // The grid is 3D: (W_out, H_out, N * C_out)
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int n_cout_idx = blockIdx.z;

    // Boundary check to avoid writing out of bounds
    if (w_out >= W_out || h_out >= H_out) {
        return;
    }

    const int n = n_cout_idx / C_out;
    const int c_out = n_cout_idx % C_out;

    float sum = 0.0f;

    // Iterate over input channels and kernel spatial dimensions
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K_h; ++kh) {
            for (int kw = 0; kw < K_w; ++kw) {
                // This is the core logic for transposed convolution's "gathering" approach.
                // We check if an input pixel, when convolved with this kernel position,
                // would contribute to the current output pixel.
                const int h_in_nom = h_out + pad_h - kh;
                const int w_in_nom = w_out + pad_w - kw;

                // Check for stride divisibility and calculate input coordinates
                if ((h_in_nom % stride_h == 0) && (w_in_nom % stride_w == 0)) {
                    const int h_in = h_in_nom / stride_h;
                    const int w_in = w_in_nom / stride_w;

                    // Check if the calculated input coordinates are within bounds
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        // PyTorch weight format for ConvTranspose2d is (C_in, C_out, K_h, K_w)
                        const long input_idx = TENSOR_IDX(n, c_in, h_in, w_in, C_in, H_in, W_in);
                        const long weight_idx = TENSOR_IDX(c_in, c_out, kh, kw, C_out, K_h, K_w);
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    const long output_idx = TENSOR_IDX(n, c_out, h_out, w_out, C_out, H_out, W_out);
    output[output_idx] = sum;
}

// C++ wrapper function that launches the CUDA kernel
torch::Tensor conv_transpose2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "Weight must be a float32 tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor");

    // Get dimensions from input tensors
    const auto N = input.size(0);
    const auto C_in = input.size(1);
    const auto H_in = input.size(2);
    const auto W_in = input.size(3);

    // Weight tensor shape is (C_in, C_out, K_h, K_w) for ConvTranspose2d
    TORCH_CHECK(weight.size(0) == C_in, "Weight C_in dimension mismatch");
    const auto C_out = weight.size(1);
    const auto K_h = weight.size(2);
    const auto K_w = weight.size(3);

    // Calculate output dimensions using the formula from PyTorch docs
    const long H_out = (H_in - 1) * stride_h - 2 * pad_h + K_h + out_pad_h;
    const long W_out = (W_in - 1) * stride_w - 2 * pad_w + K_w + out_pad_w;

    // Create output tensor, initialized to zeros
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    // Setup CUDA launch configuration
    // Use a 2D block for the spatial dimensions (H_out, W_out)
    // Use a 1D grid for the batch and output channel dimensions
    const dim3 threads(16, 16, 1);
    const dim3 blocks(
        (W_out + threads.x - 1) / threads.x,
        (H_out + threads.y - 1) / threads.y,
        N * C_out
    );

    // Launch the kernel
    conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        stride_h, stride_w,
        pad_h, pad_w
    );

    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

# C++ source for the function declaration (for PyTorch's binding mechanism)
conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w
);
"""

# Compile the inline CUDA code. This is done once when the Python module is imported.
custom_conv_transpose_op = load_inline(
    name="custom_conv_transpose_op",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_cuda_source,
    functions=["conv_transpose2d_cuda_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution using a custom CUDA kernel.
    This implementation replaces nn.ConvTranspose2d.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Must be 1 for this custom kernel. Defaults to 1.
        bias (bool, optional): Must be False for this custom kernel. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()

        # The custom kernel has limitations compared to the full PyTorch operator.
        if groups != 1:
            raise NotImplementedError("Custom CUDA kernel only supports groups=1.")
        if bias:
            raise NotImplementedError("Custom CUDA kernel does not support bias.")

        # Store convolution parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # Create the weight parameter, matching the shape of nn.ConvTranspose2d
        # Weight shape for ConvTranspose2d is (in_channels, out_channels, kH, kW)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size[0], kernel_size[1]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Use the same default initialization as PyTorch's ConvTranspose2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def _pair(self, val):
        """Helper to convert an int to a tuple of two ints."""
        if isinstance(val, int):
            return (val, val)
        return val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor from the custom kernel.
        """
        # Convert stride, padding, etc. to pairs (for height and width)
        stride_h, stride_w = self._pair(self.stride)
        pad_h, pad_w = self._pair(self.padding)
        out_pad_h, out_pad_w = self._pair(self.output_padding)

        # Call the custom CUDA function loaded via JIT compilation
        return custom_conv_transpose_op.conv_transpose2d_cuda_forward(
            x,
            self.weight,
            stride_h, stride_w,
            pad_h, pad_w,
            out_pad_h, out_pad_w
        )