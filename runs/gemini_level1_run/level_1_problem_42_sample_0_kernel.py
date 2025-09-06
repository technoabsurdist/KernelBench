import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for 2D Max Pooling
maxpool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h> // For FLT_MAX

// CUDA kernel for 2D Max Pooling
__global__ void maxpool2d_kernel(
    const float* input,
    float* output,
    const int N, const int C, const int H_in, const int W_in,
    const int H_out, const int W_out,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w) {

    // Calculate the global thread ID for the output tensor
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_outputs = N * C * H_out * W_out;

    if (index < total_outputs) {
        // De-linearize the index to get n, c, h_out, w_out
        const int w_out = index % W_out;
        const int h_out = (index / W_out) % H_out;
        const int c = (index / (W_out * H_out)) % C;
        const int n = index / (W_out * H_out * C);

        // Initialize max value to the most negative float
        float max_val = -FLT_MAX;

        // Calculate the top-left corner of the pooling window in the input tensor
        const int h_start = h_out * stride_h - pad_h;
        const int w_start = w_out * stride_w - pad_w;

        // Iterate over the kernel window
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                // Calculate the current input coordinates, considering dilation
                const int h_in = h_start + kh * dilation_h;
                const int w_in = w_start + kw * dilation_w;

                // Check if the coordinates are within the input bounds (after padding)
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    // Calculate the linear index for the input tensor
                    const int input_idx = n * C * H_in * W_in + c * H_in * W_in + h_in * W_in + w_in;
                    // Update the max value
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }
        // Write the result to the output tensor
        output[index] = max_val;
    }
}

// C++ wrapper function that will be called from Python
torch::Tensor maxpool2d_cuda(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    // Get input dimensions
    const int N = x.size(0);
    const int C = x.size(1);
    const int H_in = x.size(2);
    const int W_in = x.size(3);

    // For simplicity, assume square kernel, stride, padding, dilation as per nn.MaxPool2d with int args
    const int kernel_h = kernel_size;
    const int kernel_w = kernel_size;
    const int stride_h = stride;
    const int stride_w = stride;
    const int pad_h = padding;
    const int pad_w = padding;
    const int dilation_h = dilation;
    const int dilation_w = dilation;

    // Calculate output dimensions using the standard formula
    const int H_out = (H_in + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int W_out = (W_in + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    // Create the output tensor
    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    // Handle case with no output elements to avoid launching an empty kernel
    if (output.numel() == 0) {
        return output;
    }

    // Set up grid and block dimensions for the CUDA kernel
    const int total_outputs = N * C * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_outputs + block_size - 1) / block_size;

    // Launch the kernel
    maxpool2d_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        H_out, W_out,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w
    );

    // Check for any CUDA errors during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA error in maxpool2d_kernel: ", cudaGetErrorString(err));
    }

    return output;
}
"""

# C++ source for the function signature, required by load_inline
maxpool2d_cpp_source = """
torch::Tensor maxpool2d_cuda(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding,
    int dilation);
"""

# Compile the inline CUDA code using torch.utils.cpp_extension
# This creates a module that can be called from Python.
maxpool2d_custom_module = load_inline(
    name="maxpool2d_custom",
    cpp_sources=maxpool2d_cpp_source,
    cuda_sources=maxpool2d_source,
    functions=["maxpool2d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 2D using a custom CUDA kernel.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the custom Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # Store the compiled CUDA function
        self.maxpool_custom = maxpool2d_custom_module.maxpool2d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies custom Max Pooling 2D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D, shape (batch_size, channels, pooled_height, pooled_width).
        """
        # The custom kernel expects a contiguous CUDA tensor
        if not x.is_cuda:
            x = x.cuda()
        if not x.is_contiguous():
            x = x.contiguous()
            
        return self.maxpool_custom(x, self.kernel_size, self.stride, self.padding, self.dilation)