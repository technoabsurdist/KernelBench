import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_forward_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    // Input dimensions
    int N, int C_in, int D_in, int H_in, int W_in,
    // Output dimensions
    int C_out, int D_out, int H_out, int W_out,
    // Kernel dimensions
    int K_d, int K_h, int K_w,
    // Conv params
    int S_d, int S_h, int S_w,
    int P_d, int P_h, int P_w,
    int groups
) {
    // Using long long for index to prevent overflow with large tensors
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total_outputs = (long long)N * C_out * D_out * H_out * W_out;

    if (idx >= total_outputs) {
        return;
    }

    // Decompose 1D index 'idx' into 5D output coordinates (n, c_out, d_out, h_out, w_out)
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int d_out = (idx / (W_out * H_out)) % D_out;
    int c_out = (idx / (W_out * H_out * D_out)) % C_out;
    int n = idx / (W_out * H_out * D_out * C_out);

    // Grouping calculations
    int C_in_per_group = C_in / groups;
    int C_out_per_group = C_out / groups;
    int group_idx = c_out / C_out_per_group;
    int c_out_in_group = c_out % C_out_per_group;
    int c_in_start = group_idx * C_in_per_group;
    int c_in_end = c_in_start + C_in_per_group;

    float sum = 0.0f;

    // Iterate over all input channels within the current group
    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        // Iterate over the 3D kernel
        for (int kd = 0; kd < K_d; ++kd) {
            for (int kh = 0; kh < K_h; ++kh) {
                for (int kw = 0; kw < K_w; ++kw) {
                    // Map output coordinates back to input coordinates
                    int d_in_unstrided = d_out + P_d - kd;
                    int h_in_unstrided = h_out + P_h - kh;
                    int w_in_unstrided = w_out + P_w - kw;

                    // Check if the current kernel position contributes to the output pixel
                    // This is the core logic of transposed convolution: the stride applies to the input grid
                    if (d_in_unstrided % S_d == 0 &&
                        h_in_unstrided % S_h == 0 &&
                        w_in_unstrided % S_w == 0) {

                        int d_in = d_in_unstrided / S_d;
                        int h_in = h_in_unstrided / S_h;
                        int w_in = w_in_unstrided / S_w;

                        // Check if the calculated input coordinates are within the valid bounds
                        if (d_in >= 0 && d_in < D_in &&
                            h_in >= 0 && h_in < H_in &&
                            w_in >= 0 && w_in < W_in) {

                            // Calculate flat indices for input and weight tensors
                            long long input_offset = (long long)n * C_in * D_in * H_in * W_in +
                                                     (long long)c_in * D_in * H_in * W_in +
                                                     (long long)d_in * H_in * W_in +
                                                     (long long)h_in * W_in + w_in;

                            // PyTorch weight format for ConvTranspose3d: (in_channels, out_channels / groups, kD, kH, kW)
                            long long weight_offset = (long long)c_in * C_out_per_group * K_d * K_h * K_w +
                                                      (long long)c_out_in_group * K_d * K_h * K_w +
                                                      (long long)kd * K_h * K_w +
                                                      (long long)kh * K_w + kw;

                            sum += input[input_offset] * weight[weight_offset];
                        }
                    }
                }
            }
        }
    }

    // Add bias if it exists
    if (bias != nullptr) {
        sum += bias[c_out];
    }

    output[idx] = sum;
}

torch::Tensor conv_transpose3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<long> stride,
    std::vector<long> padding,
    std::vector<long> output_padding,
    long groups
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 5, "Input must be a 5D tensor");
    TORCH_CHECK(weight.dim() == 5, "Weight must be a 5D tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias.dim() == 1, "Bias must be a 1D tensor");
    }

    // Get dimensions
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    const int C_out = weight.size(1) * groups;

    const int K_d = weight.size(2);
    const int K_h = weight.size(3);
    const int K_w = weight.size(4);

    const int S_d = stride[0];
    const int S_h = stride[1];
    const int S_w = stride[2];

    const int P_d = padding[0];
    const int P_h = padding[1];
    const int P_w = padding[2];

    const int OP_d = output_padding[0];
    const int OP_h = output_padding[1];
    const int OP_w = output_padding[2];

    // Calculate output dimensions
    const int D_out = (D_in - 1) * S_d - 2 * P_d + K_d + OP_d;
    const int H_out = (H_in - 1) * S_h - 2 * P_h + K_h + OP_h;
    const int W_out = (W_in - 1) * S_w - 2 * P_w + K_w + OP_w;

    // Create output tensor
    auto output = torch::empty({N, C_out, D_out, H_out, W_out}, input.options());
    long long total_outputs = (long long)N * C_out * D_out * H_out * W_out;
    if (total_outputs == 0) {
        return output;
    }

    // Setup CUDA launch configuration
    const int block_size = 256;
    const long long num_blocks = (total_outputs + block_size - 1) / block_size;

    // Launch the kernel
    conv_transpose3d_forward_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        K_d, K_h, K_w,
        S_d, S_h, S_w,
        P_d, P_h, P_w,
        groups
    );

    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<long> stride,
    std::vector<long> padding,
    std::vector<long> output_padding,
    long groups
);
"""

# Compile the inline CUDA code
custom_conv_transpose3d = load_inline(
    name="custom_conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_forward_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    A custom implementation of 3D transposed convolution using a handwritten CUDA kernel.
    This module mimics the behavior and interface of torch.nn.ConvTranspose3d.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Define learnable parameters (weight and bias)
        # The weight shape for ConvTranspose3d is (in_channels, out_channels / groups, kD, kH, kW)
        self.weight = nn.Parameter(torch.Tensor(
            in_channels, out_channels // groups, *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize weights and bias to match the default behavior of nn.ConvTranspose3d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the custom 3D transposed convolution by calling the CUDA kernel.
        """
        # The custom CUDA function expects lists/vectors for stride, padding, etc.
        return custom_conv_transpose3d.conv_transpose3d_forward_cuda(
            x, self.weight, self.bias,
            list(self.stride), list(self.padding), list(self.output_padding), self.groups
        )