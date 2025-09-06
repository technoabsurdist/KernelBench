import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel source for im2col-based convolution
conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Helper function to get a pixel from an NCHW tensor with padding.
// Returns 0.0f for out-of-bounds accesses, which corresponds to zero-padding.
__device__ float get_pixel(const float* data, int n, int c, int h, int w, int C, int H, int W) {
    if (h < 0 || h >= H || w < 0 || w >= W) {
        return 0.0f;
    }
    // Index calculation: (n * C + c) * H + h) * W + w
    return data[((n * C + c) * H + h) * W + w];
}

// im2col kernel: This kernel transforms patches of the input tensor into columns.
// The convolution operation can then be performed as a single matrix multiplication.
__global__ void im2col_kernel(
    const float* input_data,
    int batch_size, int in_channels, int in_height, int in_width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int out_height, int out_width,
    float* col_data) {

    // Each thread is responsible for computing one column of the im2col matrix.
    int col_index = blockIdx.x * blockDim.x + threadIdx.x;
    int col_size = in_channels * kernel_h * kernel_w;
    int total_cols = batch_size * out_height * out_width;

    if (col_index >= total_cols) {
        return;
    }

    // Decompose the column index to find the corresponding batch and output pixel location.
    int w_out = col_index % out_width;
    int h_out = (col_index / out_width) % out_height;
    int n = col_index / (out_width * out_height);

    // Calculate the starting position in the input tensor for this patch.
    int start_h = h_out * stride_h - pad_h;
    int start_w = w_out * stride_w - pad_w;

    // Pointer to the start of the current column in the output col_data buffer.
    float* current_col = col_data + col_size * col_index;

    // Iterate over input channels, kernel height, and kernel width to fill the column.
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = start_h + kh * dilation_h;
                int w_in = start_w + kw * dilation_w;

                *current_col = get_pixel(input_data, n, c_in, h_in, w_in, in_channels, in_height, in_width);
                current_col++;
            }
        }
    }
}

// Basic matrix multiplication kernel: C = A * B
// A is the reshaped weight matrix (M, K)
// B is the im2col matrix (K, N)
// C is the output matrix (M, N)
__global__ void matmul_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Kernel to add the bias vector to the output of the matrix multiplication.
// output: (out_channels, batch_size * out_height * out_width)
// bias: (out_channels)
__global__ void add_bias_kernel(
    float* output, const float* bias,
    int out_channels, int spatial_dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_channels * spatial_dim) {
        int c = idx / spatial_dim; // Get the output channel index
        output[idx] += bias[c];
    }
}


// C++ wrapper function that orchestrates the im2col, matmul, and bias addition.
torch::Tensor conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {

    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    }

    input = input.contiguous();
    weight = weight.contiguous();
    if (bias.defined()) {
        bias = bias.contiguous();
    }

    // Get tensor dimensions
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);

    const int out_channels = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    // For simplicity, this implementation assumes square kernels, strides, etc.
    const int stride_h = stride;
    const int stride_w = stride;
    const int pad_h = padding;
    const int pad_w = padding;
    const int dilation_h = dilation;
    const int dilation_w = dilation;

    // Calculate output dimensions
    const int out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    // --- 1. im2col ---
    const int col_rows = in_channels * kernel_h * kernel_w;
    const int col_cols = batch_size * out_height * out_width;
    auto col_buffer = torch::empty({col_rows, col_cols}, input.options());

    const int num_kernels = batch_size * out_height * out_width;
    const int threads_per_block_im2col = 256;
    const int num_blocks_im2col = (num_kernels + threads_per_block_im2col - 1) / threads_per_block_im2col;

    im2col_kernel<<<num_blocks_im2col, threads_per_block_im2col>>>(
        input.data_ptr<float>(),
        batch_size, in_channels, in_height, in_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        out_height, out_width,
        col_buffer.data_ptr<float>()
    );
    
    // --- 2. GEMM (Matrix Multiplication) ---
    auto weight_reshaped = weight.view({out_channels, -1});
    auto output_gemm = torch::empty({out_channels, col_cols}, input.options());

    const int M = out_channels;
    const int N = col_cols;
    const int K = col_rows;

    const dim3 threads_per_block_gemm(16, 16);
    const dim3 num_blocks_gemm(
        (N + threads_per_block_gemm.x - 1) / threads_per_block_gemm.x,
        (M + threads_per_block_gemm.y - 1) / threads_per_block_gemm.y
    );

    matmul_kernel<<<num_blocks_gemm, threads_per_block_gemm>>>(
        weight_reshaped.data_ptr<float>(),
        col_buffer.data_ptr<float>(),
        output_gemm.data_ptr<float>(),
        M, N, K
    );

    // --- 3. Add Bias (if provided) ---
    if (bias.defined()) {
        const int total_elements = out_channels * col_cols;
        const int threads_per_block_bias = 256;
        const int num_blocks_bias = (total_elements + threads_per_block_bias - 1) / threads_per_block_bias;

        add_bias_kernel<<<num_blocks_bias, threads_per_block_bias>>>(
            output_gemm.data_ptr<float>(),
            bias.data_ptr<float>(),
            out_channels,
            col_cols
        );
    }

    // --- 4. Reshape output to NCHW format ---
    auto output = output_gemm.view({out_channels, batch_size, out_height, out_width});
    return output.permute({1, 0, 2, 3}).contiguous();
}
"""

conv2d_cpp_source = "torch::Tensor conv2d_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int dilation);"

# Compile the inline CUDA code using torch.utils.cpp_extension
custom_conv2d_impl = load_inline(
    name="custom_conv2d_impl",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_forward_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation using a custom CUDA kernel.
    This implementation uses the im2col + GEMM algorithm.

    NOTE: This implementation does not support the 'groups' parameter. All other
    parameters match the behavior of nn.Conv2d.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Not supported. Must be 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        if groups != 1:
            raise NotImplementedError("Custom CUDA Conv2d kernel only supports groups=1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Define learnable parameters, same as nn.Conv2d, to be managed by PyTorch
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Use standard PyTorch initialization for conv layers
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
        return custom_conv2d_impl.conv2d_forward_cuda(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation
        )