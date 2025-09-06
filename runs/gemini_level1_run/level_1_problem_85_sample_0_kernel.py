import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for depthwise 2D convolution
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int C, int H, int W,
    int KH, int KW,
    int SH, int SW,
    int PH, int PW,
    int DH, int DW,
    int OH, int OW) {

    const int output_size = B * C * OH * OW;
    // Use a grid-stride loop to ensure all elements are processed,
    // regardless of the number of blocks and threads.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < output_size; i += gridDim.x * blockDim.x) {
        // Decompose 1D index i into 4D index (b, c, oh, ow)
        int ow = i % OW;
        int oh = (i / OW) % OH;
        int c = (i / (OW * OH)) % C;
        int b = i / (OW * OH * C);

        float acc = 0.0f;

        // Loop over the kernel dimensions
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                // Calculate the corresponding input coordinates
                int h_in = oh * SH + kh * DH - PH;
                int w_in = ow * SW + kw * DW - PW;

                // Boundary check for the input tensor
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    // Calculate flat indices for input and weight tensors
                    long long input_idx = (long long)b * C * H * W + (long long)c * H * W + (long long)h_in * W + w_in;
                    // For depthwise, weight index is (c, 0, kh, kw)
                    long long weight_idx = (long long)c * KH * KW + (long long)kh * KW + kw;
                    
                    acc += input[input_idx] * weight[weight_idx];
                }
            }
        }

        // Add bias if it exists
        if (bias != nullptr) {
            acc += bias[c];
        }

        output[i] = acc;
    }
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias, // Can be an undefined tensor
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w) {

    // Check that tensors are on the same CUDA device
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    
    // Ensure tensors are contiguous in memory for direct pointer access
    input = input.contiguous();
    weight = weight.contiguous();
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
        bias = bias.contiguous();
    }

    // Get input tensor dimensions
    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    // Get weight tensor dimensions (shape is [out_channels, 1, KH, KW])
    const int KH = weight.size(2);
    const int KW = weight.size(3);

    // Calculate output dimensions
    const int OH = (H + 2 * padding_h - dilation_h * (KH - 1) - 1) / stride_h + 1;
    const int OW = (W + 2 * padding_w - dilation_w * (KW - 1) - 1) / stride_w + 1;

    // Create an empty output tensor
    auto output = torch::empty({B, C, OH, OW}, input.options());

    const int output_size = B * C * OH * OW;
    if (output_size == 0) {
        return output;
    }

    // Configure kernel launch parameters
    const int block_size = 256;
    const int num_blocks = (output_size + block_size - 1) / block_size;

    // Get a raw pointer to the bias data, or nullptr if bias is not defined
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    // Launch the CUDA kernel
    depthwise_conv2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        B, C, H, W,
        KH, KW,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        OH, OW
    );
    
    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

# Define the C++ function signature for the JIT compiler
depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w);
"""

# Compile the inline CUDA code using torch.utils.cpp_extension.load_inline
# This is done once when the module is imported.
custom_depthwise_conv2d = load_inline(
    name="custom_depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and asymmetric kernel
    using a custom CUDA kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution. Must be equal to in_channels.
        kernel_size_h (int): Height of the convolution kernel.
        kernel_size_w (int): Width of the convolution kernel.
        stride_h (int, optional): Stride of the convolution in height dimension. Defaults to 1.
        stride_w (int, optional): Stride of the convolution in width dimension. Defaults to 1.
        padding_h (int, optional): Padding applied to the input in height dimension. Defaults to 0.
        padding_w (int, optional): Padding applied to the input in width dimension. Defaults to 0.
        dilation_h (int, optional): Spacing between kernel elements in height dimension. Defaults to 1.
        dilation_w (int, optional): Spacing between kernel elements in width dimension. Defaults to 1.
        groups (int, optional): Number of blocked connections. Must be equal to in_channels.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # For depthwise convolution, out_channels must be equal to in_channels and groups must be equal to in_channels
        if in_channels != out_channels:
            raise ValueError("in_channels must equal out_channels for depthwise convolution")
        if in_channels != groups:
            raise ValueError("groups must equal in_channels for depthwise convolution")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups

        # The weight tensor for a depthwise convolution has shape (out_channels, 1, kernel_height, kernel_width)
        self.weight = nn.Parameter(torch.Tensor(out_channels, 1, kernel_size_h, kernel_size_w))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Mimic the initialization of nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return custom_depthwise_conv2d.depthwise_conv2d_cuda(
            x,
            self.weight,
            self.bias,
            self.stride_h, self.stride_w,
            self.padding_h, self.padding_w,
            self.dilation_h, self.dilation_w
        )