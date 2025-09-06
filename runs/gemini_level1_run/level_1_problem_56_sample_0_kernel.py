import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel source for Conv2D using the im2col+GEMM strategy.
# The im2col part is implemented as a custom CUDA kernel, while the GEMM part
# leverages the highly optimized torch::matmul function (which calls cuBLAS).
conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// CUDA kernel for im2col, handles batch dimension.
// This kernel transforms patches of the input image into columns of a matrix.
__global__ void im2col_kernel_batched(const float* data_im,
    const int batch_size, const int in_channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int out_height, const int out_width,
    float* data_col) {

    // Use long long for tid and num_kernels to avoid overflow on large problems
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long num_kernels = (long long)batch_size * in_channels * kernel_h * kernel_w * out_height * out_width;

    if (tid >= num_kernels) {
        return;
    }

    // Decompose the 1D thread index (tid) into 6D coordinates (b, c_in, k_h, k_w, h_out, w_out)
    int w_out = tid % out_width;
    int h_out = (tid / out_width) % out_height;
    int k_w = (tid / ((long long)out_width * out_height)) % kernel_w;
    int k_h = (tid / ((long long)out_width * out_height * kernel_w)) % kernel_h;
    int c_in = (tid / ((long long)out_width * out_height * kernel_w * kernel_h)) % in_channels;
    int b = tid / ((long long)out_width * out_height * kernel_w * kernel_h * in_channels);

    // Calculate the corresponding input coordinates based on convolution parameters
    int h_in = h_out * stride_h - pad_h + k_h * dilation_h;
    int w_in = w_out * stride_w - pad_w + k_w * dilation_w;

    // Calculate the linear index for the destination 'col' matrix
    // The 'col' matrix has a logical shape of (B, C_in * K_h * K_w, H_out * W_out)
    long long C_col = (long long)in_channels * kernel_h * kernel_w;
    long long N_col = (long long)out_height * out_width;

    long long col_row = c_in * (kernel_h * kernel_w) + k_h * kernel_w + k_w;
    long long col_col = h_out * out_width + w_out;
    long long col_index = b * (C_col * N_col) + col_row * N_col + col_col;

    // Check if the calculated input coordinates are within the bounds of the input tensor
    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
        // If in bounds, copy the data from the input tensor
        long long im_plane_size = (long long)in_channels * height * width;
        long long im_index = b * im_plane_size + (c_in * height + h_in) * width + w_in;
        data_col[col_index] = data_im[im_index];
    } else {
        // If out of bounds (due to padding), write 0.0f
        data_col[col_index] = 0.0f;
    }
}

// C++ function to launch the im2col kernel
void im2col_launcher(const float* data_im,
    const int batch_size, const int in_channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int out_height, const int out_width,
    float* data_col) {

    long long num_kernels = (long long)batch_size * in_channels * kernel_h * kernel_w * out_height * out_width;
    const int block_size = 512; // A common and often efficient block size
    
    // Use long long for num_blocks calculation to be safe, then cast to int for launch
    long long num_blocks_ll = (num_kernels + block_size - 1) / block_size;
    TORCH_CHECK(num_blocks_ll < std::numeric_limits<int>::max(), "Too many blocks required for CUDA kernel launch.");
    int num_blocks = static_cast<int>(num_blocks_ll);

    im2col_kernel_batched<<<num_blocks, block_size>>>(
        data_im, batch_size, in_channels, height, width,
        kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, out_height, out_width, data_col
    );
    
    // Check for kernel launch errors for easier debugging
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error in im2col_launcher: ", cudaGetErrorString(err));
}


// The main C++ interface function that will be called from Python
torch::Tensor conv2d_im2col_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    // --- Input Validation ---
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input tensor must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "Weight tensor must be 4D (C_out, C_in/groups, K_h, K_w)");
    // This custom implementation currently only supports groups=1
    TORCH_CHECK(groups == 1, "Custom Conv2D kernel currently only supports groups=1");

    // Ensure contiguous memory layout for direct pointer access
    input = input.contiguous();
    weight = weight.contiguous();

    // --- Get Dimensions ---
    const int64_t batch_size = input.size(0);
    const int64_t in_channels = input.size(1);
    const int64_t in_height = input.size(2);
    const int64_t in_width = input.size(3);

    const int64_t out_channels = weight.size(0);
    const int64_t kernel_h = weight.size(2);
    const int64_t kernel_w = weight.size(3);

    const int64_t stride_h = stride[0];
    const int64_t stride_w = stride[1];
    const int64_t pad_h = padding[0];
    const int64_t pad_w = padding[1];
    const int64_t dilation_h = dilation[0];
    const int64_t dilation_w = dilation[1];

    // --- Calculate Output Dimensions ---
    const int64_t out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int64_t out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    TORCH_CHECK(out_height > 0 && out_width > 0, "Calculated output dimensions must be positive.");

    // --- Step 1: im2col ---
    // Create the intermediate 'col' buffer
    auto col_buffer = torch::zeros({batch_size, in_channels * kernel_h * kernel_w, out_height * out_width}, input.options());

    // Launch the custom im2col CUDA kernel
    im2col_launcher(
        input.data_ptr<float>(),
        batch_size, in_channels, in_height, in_width,
        kernel_h, kernel_w,
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
        out_height, out_width,
        col_buffer.data_ptr<float>()
    );

    // --- Step 2: GEMM (Matrix Multiplication) ---
    // Reshape weight for GEMM: (C_out, C_in * K_h * K_w)
    auto weight_reshaped = weight.view({out_channels, -1});

    // Perform batched matrix multiplication using torch::matmul
    // (1, C_out, C_in*K*K) @ (B, C_in*K*K, H_out*W_out) -> (B, C_out, H_out*W_out)
    auto output_reshaped = torch::matmul(weight_reshaped, col_buffer);

    // --- Step 3: Reshape and Add Bias ---
    // Reshape output to the final 4D tensor shape: (B, C_out, H_out, W_out)
    auto output = output_reshaped.view({batch_size, out_channels, out_height, out_width});

    // Add bias if it is provided
    if (bias.has_value()) {
        output = output + bias.value().view({1, out_channels, 1, 1});
    }

    return output;
}
"""

# Define the C++ function signature for the JIT compiler
conv2d_cpp_source = """
torch::Tensor conv2d_im2col_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups);
"""

# Compile the inline CUDA/C++ code
conv2d_im2col = load_inline(
    name="conv2d_im2col",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_cuda_source,
    functions=["conv2d_im2col_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution using a custom CUDA kernel based on the
    im2col + GEMM algorithm.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of two integers for kernel height and width.
        stride (tuple, optional): Stride for height and width. Defaults to (1, 1).
        padding (tuple, optional): Padding for height and width. Defaults to (0, 0).
        dilation (tuple, optional): Dilation for height and width. Defaults to (1, 1).
        groups (int, optional): Number of groups. Custom kernel only supports 1. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        if groups != 1:
            raise NotImplementedError("Custom Conv2D kernel only supports groups=1")

        # Store convolution parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Define learnable parameters (weight and optional bias)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            # It's important to register 'bias' as None if it's not used
            self.register_parameter('bias', None)

        # Store the compiled custom operator
        self.conv2d_op = conv2d_im2col

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # The C++ function expects lists for stride, padding, and dilation
        return self.conv2d_op.conv2d_im2col_cuda(
            x,
            self.weight,
            self.bias,
            list(self.stride),
            list(self.padding),
            list(self.dilation),
            self.groups
        )