import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for 2D convolution
conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_forward_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int KH, int KW,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    // Grid is launched with (B, C_out, ceil(H_out * W_out / block_size))
    // Each thread computes one output element
    int b = blockIdx.x;
    int c_out = blockIdx.y;
    int out_spatial_idx = blockIdx.z * blockDim.x + threadIdx.x;

    if (out_spatial_idx >= H_out * W_out) {
        return;
    }

    int w_out = out_spatial_idx % W_out;
    int h_out = out_spatial_idx / W_out;

    // Initialize accumulator with bias if it exists
    float acc = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Iterate over input channels and kernel dimensions
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                // Calculate corresponding input coordinates
                int h_in = h_out * stride_h - pad_h + kh;
                int w_in = w_out * stride_w - pad_w + kw;

                // Boundary check for padding
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    // Calculate flat indices for input and weight tensors
                    long input_idx = (long)b * C_in * H_in * W_in +
                                     (long)c_in * H_in * W_in +
                                     (long)h_in * W_in +
                                     w_in;
                    long weight_idx = (long)c_out * C_in * KH * KW +
                                      (long)c_in * KH * KW +
                                      (long)kh * KW +
                                      kw;
                    
                    acc += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Calculate flat index for output tensor and write the result
    long output_idx = (long)b * C_out * H_out * W_out +
                      (long)c_out * H_out * W_out +
                      (long)h_out * W_out +
                      w_out;
    output[output_idx] = acc;
}

torch::Tensor conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "Weight must be a float32 tensor");
    
    // Ensure tensors are contiguous in memory
    input = input.contiguous();
    weight = weight.contiguous();
    if (bias.defined()) {
        bias = bias.contiguous();
    }

    // Get tensor dimensions
    const int B = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    const int C_out = weight.size(0);
    // C_in from weight should match input, but we trust the user for this simple implementation
    const int KH = weight.size(2);
    const int KW = weight.size(3);

    // Calculate output dimensions
    const int H_out = (H_in + 2 * pad_h - KH) / stride_h + 1;
    const int W_out = (W_in + 2 * pad_w - KW) / stride_w + 1;

    // Create the output tensor
    auto output = torch::zeros({B, C_out, H_out, W_out}, input.options());

    // Configure and launch the kernel
    const int block_size = 256;
    const int grid_z = (H_out * W_out + block_size - 1) / block_size;
    const dim3 grid(B, C_out, grid_z);
    const dim3 block(block_size);

    // Get raw data pointers
    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    // Launch the CUDA kernel
    conv2d_forward_kernel<<<grid, block>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        B, C_in, H_in, W_in,
        C_out, H_out, W_out,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w
    );
    
    // Check for errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

conv2d_cpp_source = (
    "torch::Tensor conv2d_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride_h, int stride_w, int pad_h, int pad_w);"
)

# Compile the inline CUDA code for 2D convolution
# This is done once when the module is imported.
conv2d_cuda_op = load_inline(
    name="conv2d_cuda_op",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_forward_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution using a custom CUDA kernel.
    This implementation replaces nn.Conv2d with a direct convolution kernel.
    NOTE: Dilation and groups are not supported in this custom implementation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        dilation (int or tuple, optional): Not supported. Must be 1.
        groups (int, optional): Not supported. Must be 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # This custom kernel does not support dilation or grouped convolution
        if dilation != 1 and dilation != (1, 1):
            raise NotImplementedError("Custom CUDA kernel does not support dilation > 1.")
        if groups != 1:
            raise NotImplementedError("Custom CUDA kernel does not support groups > 1.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Create learnable parameters (weight and optional bias)
        # Initialize them similarly to how PyTorch's Conv2d does for fair comparison
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Kaiming uniform initialization
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
        # Normalize stride and padding to be tuples of 2 integers
        _stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        _padding = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)

        # Call the custom CUDA function
        return conv2d_cuda_op.conv2d_forward_cuda(
            x, self.weight, self.bias,
            _stride[0], _stride[1],
            _padding[0], _padding[1]
        )