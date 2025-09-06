import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for ConvTranspose3d
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int kD, int kH, int kW,
    int sD, int sH, int sW,
    int pD, int pH, int pW,
    int groups,
    bool has_bias
) {
    // Linear index for the output tensor
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total_outputs = (long long)N * C_out * D_out * H_out * W_out;

    if (idx >= total_outputs) return;

    // De-linearize index to get (n, c_out, d_out, h_out, w_out)
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int d_out = (idx / (W_out * H_out)) % D_out;
    int c_out = (idx / ((long long)W_out * H_out * D_out)) % C_out;
    int n = idx / ((long long)W_out * H_out * D_out * C_out);

    // Accumulator for the output value
    float acc = 0.0f;

    // Group convolution logic
    int c_in_per_group = C_in / groups;
    int c_out_per_group = C_out / groups;
    int group_idx = c_out / c_out_per_group;
    int c_in_start = group_idx * c_in_per_group;
    int c_in_end = c_in_start + c_in_per_group;
    int c_out_group = c_out % c_out_per_group;

    // Loop over input channels in the group
    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        // Loop over kernel
        for (int kd = 0; kd < kD; ++kd) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    // Check if this kernel position contributes to the output
                    int d_in_nom = d_out + pD - kd;
                    int h_in_nom = h_out + pH - kh;
                    int w_in_nom = w_out + pW - kw;

                    if ((d_in_nom % sD == 0) && (h_in_nom % sH == 0) && (w_in_nom % sW == 0)) {
                        int d_in = d_in_nom / sD;
                        int h_in = h_in_nom / sH;
                        int w_in = w_in_nom / sW;

                        // Check bounds of the input tensor
                        if (d_in >= 0 && d_in < D_in &&
                            h_in >= 0 && h_in < H_in &&
                            w_in >= 0 && w_in < W_in) {

                            // Calculate linear indices for input and weight
                            // Input: (n, c_in, d_in, h_in, w_in)
                            long long input_idx = (long long)n * C_in * D_in * H_in * W_in +
                                                  (long long)c_in * D_in * H_in * W_in +
                                                  (long long)d_in * H_in * W_in +
                                                  (long long)h_in * W_in +
                                                  w_in;

                            // Weight: (c_in, c_out_group, kd, kh, kw)
                            // PyTorch weight shape is (C_in, C_out/groups, kD, kH, kW)
                            long long weight_idx = (long long)c_in * c_out_per_group * kD * kH * kW +
                                                   (long long)c_out_group * kD * kH * kW +
                                                   (long long)kd * kH * kW +
                                                   (long long)kh * kW +
                                                   kw;

                            // Accumulate
                            acc += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }

    // Add bias if it exists
    if (has_bias) {
        acc += bias[c_out];
    }

    // Write to output
    output[idx] = acc;
}

torch::Tensor conv_transpose3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int groups
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");

    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    // Weight shape: (C_in, C_out/groups, kD, kH, kW)
    const int C_out = weight.size(1) * groups;
    const int kD = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);

    // Calculate output dimensions
    const int D_out = (D_in - 1) * stride_d - 2 * pad_d + kD + out_pad_d;
    const int H_out = (H_in - 1) * stride_h - 2 * pad_h + kH + out_pad_h;
    const int W_out = (W_in - 1) * stride_w - 2 * pad_w + kW + out_pad_w;

    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    const bool has_bias = bias.defined() && bias.numel() > 0;
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
        TORCH_CHECK(bias.size(0) == C_out, "Bias size must match out_channels");
    }

    const long long total_outputs = (long long)N * C_out * D_out * H_out * W_out;
    if (total_outputs == 0) {
        return output;
    }
    const int block_size = 256;
    const int num_blocks = (total_outputs + block_size - 1) / block_size;

    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        groups,
        has_bias
    );
    
    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int groups
);
"""

# Compile the inline CUDA code
conv_transpose3d_op = load_inline(
    name="conv_transpose3d_op",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution using a custom CUDA kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_depth, kernel_width, kernel_height).
        stride (tuple, optional): Stride of the convolution. Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input. Defaults to (0, 0, 0).
        output_padding (tuple, optional): Additional size added to one side of the output shape. Defaults to (0, 0, 0).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        if self.in_channels % self.groups != 0 or self.out_channels % self.groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")

        # PyTorch's ConvTranspose weight shape is (in_channels, out_channels / groups, *kernel_size)
        self.weight = nn.Parameter(torch.Tensor(
            in_channels, out_channels // groups, *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using the custom CUDA kernel.
        """
        # The C++ function expects a tensor for bias, even if it's empty.
        bias_tensor = self.bias if self.bias is not None else torch.empty(0, device=x.device)

        return conv_transpose3d_op.conv_transpose3d_forward_cuda(
            x,
            self.weight,
            bias_tensor,
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.output_padding[0], self.output_padding[1], self.output_padding[2],
            self.groups
        )