import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline
from torch.nn.modules.utils import _triple

# Define the custom CUDA kernel for 3D convolution
conv3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_forward_kernel(
    const float* input, const float* weight, const float* bias, float* output,
    const int N, const int C_in, const int D_in, const int H_in, const int W_in,
    const int C_out, const int D_out, const int H_out, const int W_out,
    const int K_d, const int K_h, const int K_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int dil_d, const int dil_h, const int dil_w) {

    // Calculate total number of output elements
    const long long total_outputs = (long long)N * C_out * D_out * H_out * W_out;
    
    // Get global thread index
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= total_outputs) {
        return;
    }

    // Decode 1D index to 5D output coordinates (n, c_out, d_out, h_out, w_out)
    const int w_out = index % W_out;
    const int h_out = (index / W_out) % H_out;
    const int d_out = (index / (W_out * H_out)) % D_out;
    const int c_out = (index / (W_out * H_out * D_out)) % C_out;
    const int n = index / (W_out * H_out * D_out * C_out);

    float acc = 0.0f;

    // This implementation assumes groups = 1
    // Loop over input channels
    for (int c_in = 0; c_in < C_in; ++c_in) {
        // Loop over kernel depth, height, and width
        for (int kd = 0; kd < K_d; ++kd) {
            for (int kh = 0; kh < K_h; ++kh) {
                for (int kw = 0; kw < K_w; ++kw) {
                    // Calculate corresponding input coordinates
                    const int d_in = d_out * stride_d - pad_d + kd * dil_d;
                    const int h_in = h_out * stride_h - pad_h + kh * dil_h;
                    const int w_in = w_out * stride_w - pad_w + kw * dil_w;

                    // Check if the input coordinates are within bounds
                    if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        // Calculate linear indices for input and weight tensors
                        const long long input_idx = (long long)n * C_in * D_in * H_in * W_in +
                                                    (long long)c_in * D_in * H_in * W_in +
                                                    (long long)d_in * H_in * W_in +
                                                    (long long)h_in * W_in +
                                                    w_in;
                        
                        const long long weight_idx = (long long)c_out * C_in * K_d * K_h * K_w +
                                                     (long long)c_in * K_d * K_h * K_w +
                                                     (long long)kd * K_h * K_w +
                                                     (long long)kh * K_w +
                                                     kw;
                        
                        acc += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Add bias if it exists
    if (bias != nullptr) {
        acc += bias[c_out];
    }

    // Write result to the output tensor
    output[index] = acc;
}

torch::Tensor conv3d_forward_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::IntArrayRef stride, torch::IntArrayRef padding, torch::IntArrayRef dilation) {

    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    }

    // Get dimensions from input and weight tensors
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    const int C_out = weight.size(0);
    const int K_d = weight.size(2);
    const int K_h = weight.size(3);
    const int K_w = weight.size(4);

    // Extract stride, padding, and dilation values
    const int stride_d = stride[0]; const int stride_h = stride[1]; const int stride_w = stride[2];
    const int pad_d = padding[0]; const int pad_h = padding[1]; const int pad_w = padding[2];
    const int dil_d = dilation[0]; const int dil_h = dilation[1]; const int dil_w = dilation[2];

    // Calculate output dimensions
    const int D_out = (D_in + 2 * pad_d - dil_d * (K_d - 1) - 1) / stride_d + 1;
    const int H_out = (H_in + 2 * pad_h - dil_h * (K_h - 1) - 1) / stride_h + 1;
    const int W_out = (W_in + 2 * pad_w - dil_w * (K_w - 1) - 1) / stride_w + 1;

    TORCH_CHECK(D_out > 0 && H_out > 0 && W_out > 0, "Output dimensions must be positive");

    // Create the output tensor
    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    // Setup grid and block sizes for CUDA kernel launch
    const long long total_outputs = (long long)N * C_out * D_out * H_out * W_out;
    const int block_size = 256;
    const int grid_size = (total_outputs + block_size - 1) / block_size;

    // Get data pointers
    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    // Launch the CUDA kernel
    conv3d_forward_kernel<<<grid_size, block_size>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        K_d, K_h, K_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w
    );
    
    // Check for any CUDA errors during kernel execution
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor conv3d_forward_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::IntArrayRef stride, torch::IntArrayRef padding, torch::IntArrayRef dilation);
"""

# Compile the inline CUDA code for 3D convolution
# This might take a moment the first time it's run.
conv3d_cuda_op = load_inline(
    name="conv3d_cuda_op",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_cuda_source,
    functions=["conv3d_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation using a custom CUDA kernel.
    This implementation assumes groups = 1.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_width, kernel_height, kernel_depth).
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections. Must be 1 for this custom implementation.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        if groups != 1:
            raise NotImplementedError("Custom CUDA kernel only supports groups=1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups

        # Define learnable parameters (weight and bias)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize weights and bias similar to PyTorch's default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor.
        """
        return conv3d_cuda_op.conv3d_forward_cuda(x, self.weight, self.bias, self.stride, self.padding, self.dilation)