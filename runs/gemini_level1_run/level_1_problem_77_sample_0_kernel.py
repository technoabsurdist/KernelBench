import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for 3D transposed convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for 3D transposed convolution.
// This is a naive "gather" implementation where each output element thread iterates
// over the relevant input and kernel elements.
__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    float* output,
    const float* bias,
    const int N, const int C_in, const int D_in, const int H_in, const int W_in,
    const int C_out, const int D_out, const int H_out, const int W_out,
    const int KD, const int KH, const int KW,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int dil_d, const int dil_h, const int dil_w
) {
    // Using a grid-stride loop to ensure all output elements are processed
    // regardless of the number of blocks launched.
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < (long long)N * C_out * D_out * H_out * W_out;
         index += (long long)blockDim.x * gridDim.x) {

        // Decompose the 1D index into 5D output coordinates
        int w_out = index % W_out;
        int h_out = (index / W_out) % H_out;
        int d_out = (index / (W_out * H_out)) % D_out;
        int c_out = (index / (W_out * H_out * D_out)) % C_out;
        int n = index / (W_out * H_out * D_out * C_out);

        float acc = 0.0f;

        // Iterate over the kernel and input channels
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kd = 0; kd < KD; ++kd) {
                for (int kh = 0; kh < KH; ++kh) {
                    for (int kw = 0; kw < KW; ++kw) {
                        // Core logic of the "gather" operation for transposed convolution.
                        int d_in_nom = d_out + pad_d - kd * dil_d;
                        int h_in_nom = h_out + pad_h - kh * dil_h;
                        int w_in_nom = w_out + pad_w - kw * dil_w;

                        // The input element contributes only if it's not a "zero" inserted by the stride.
                        if (d_in_nom % stride_d == 0 && h_in_nom % stride_h == 0 && w_in_nom % stride_w == 0) {
                            int d_in = d_in_nom / stride_d;
                            int h_in = h_in_nom / stride_h;
                            int w_in = w_in_nom / stride_w;

                            // Check if the calculated input coordinates are within the input tensor bounds.
                            if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                // Input tensor flat index: (N, C_in, D_in, H_in, W_in)
                                long long input_idx = (long long)n * C_in * D_in * H_in * W_in +
                                                    (long long)c_in * D_in * H_in * W_in +
                                                    (long long)d_in * H_in * W_in +
                                                    (long long)h_in * W_in +
                                                    w_in;

                                // Weight tensor flat index: (C_in, C_out, KD, KH, KW)
                                long long weight_idx = (long long)c_in * C_out * KD * KH * KW +
                                                     (long long)c_out * KD * KH * KW +
                                                     (long long)kd * KH * KW +
                                                     (long long)kh * KW +
                                                     kw;

                                acc += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }

        // Add bias if provided
        if (bias != nullptr) {
            acc += bias[c_out];
        }
        output[index] = acc;
    }
}

// C++ wrapper function that will be called from Python
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    }

    // Ensure tensors are contiguous in memory for predictable access
    input = input.contiguous();
    weight = weight.contiguous();
    if (bias.defined() && bias.numel() > 0) {
        bias = bias.contiguous();
    }

    // Get input dimensions
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    // Get weight dimensions: PyTorch stores them as (C_in, C_out, KD, KH, KW)
    const int C_out = weight.size(1);
    const int KD = weight.size(2);
    const int KH = weight.size(3);
    const int KW = weight.size(4);

    // Assume symmetric stride, padding, and dilation as per the original model's design
    const int stride_d = stride, stride_h = stride, stride_w = stride;
    const int pad_d = padding, pad_h = padding, pad_w = padding;
    const int dil_d = dilation, dil_h = dilation, dil_w = dilation;

    // Calculate output dimensions using the standard formula for transposed convolution
    // (output_padding is assumed to be 0)
    const int D_out = (D_in - 1) * stride_d - 2 * pad_d + dil_d * (KD - 1) + 1;
    const int H_out = (H_in - 1) * stride_h - 2 * pad_h + dil_h * (KH - 1) + 1;
    const int W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (KW - 1) + 1;

    // Create the output tensor, initialized to zeros
    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());
    const long long total_elements = output.numel();
    if (total_elements == 0) {
        return output;
    }

    // Get raw data pointers
    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const float* bias_ptr = (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr;

    // Configure and launch the kernel
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input_ptr, weight_ptr, output_ptr, bias_ptr,
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        KD, KH, KW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w
    );

    // Check for errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

# C++ source for the function signature, required by load_inline
conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation
);
"""

# Use torch.utils.cpp_extension to compile the CUDA code at runtime
custom_conv_transpose3d_module = load_inline(
    name="custom_conv_transpose3d_module",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=False, # Set to True for detailed compilation output
)


class ModelNew(nn.Module):
    """
    A custom implementation of 3D transposed convolution using a hand-written CUDA kernel.
    This module mimics the interface of nn.ConvTranspose3d and manages its own
    learnable weight and bias parameters.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Define learnable parameters: weight and bias
        # The weight shape for ConvTranspose3d is (in_channels, out_channels, kD, kH, kW)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            # Important to register as None so it's handled correctly by .to(device) etc.
            self.register_parameter('bias', None)

        # Initialize parameters similar to PyTorch's default for proper training behavior
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Replicates the default initialization from nn.Conv3d to ensure the model
        starts in a reasonable state for training.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            # Calculate fan_in from the weight tensor shape
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the custom 3D transposed convolution by calling the compiled CUDA function.
        """
        # The C++ wrapper expects a defined tensor for bias, even if it's empty.
        # If self.bias is None, we pass an empty tensor which the wrapper will handle.
        bias_tensor = self.bias if self.bias is not None else torch.empty(0, device=x.device)

        return custom_conv_transpose3d_module.conv_transpose3d_cuda(
            x, self.weight, bias_tensor, self.stride, self.padding, self.dilation
        )