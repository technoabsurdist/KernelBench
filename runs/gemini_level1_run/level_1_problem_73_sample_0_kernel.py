import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    // Input dims
    const int N, const int C_in, const int D_in, const int H_in, const int W_in,
    // Output dims
    const int C_out, const int D_out, const int H_out, const int W_out,
    // Kernel dims
    const int K,
    // Conv params
    const int stride, const int padding, const int groups,
    // Total output elements for grid-stride loop
    const long long total_elements
) {
    // Using a grid-stride loop for robustness
    for (long long linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
         linear_idx < total_elements;
         linear_idx += gridDim.x * blockDim.x) {

        // Decompose linear_idx to 5D output coordinates (n, c_out, d_out, h_out, w_out)
        const int w_out = linear_idx % W_out;
        const int h_out = (linear_idx / W_out) % H_out;
        const int d_out = (linear_idx / (W_out * H_out)) % D_out;
        const int c_out = (linear_idx / (W_out * H_out * D_out)) % C_out;
        const int n = linear_idx / (W_out * H_out * D_out * C_out);

        // Grouping calculations
        const int C_in_per_group = C_in / groups;
        const int C_out_per_group = C_out / groups;
        const int group_idx = c_out / C_out_per_group;
        const int c_out_in_group = c_out % C_out_per_group;
        const int c_in_start = group_idx * C_in_per_group;

        float sum = 0.0f;

        // Loop over input channels for the current group
        for (int c_in_offset = 0; c_in_offset < C_in_per_group; ++c_in_offset) {
            const int c_in = c_in_start + c_in_offset;

            // Loop over kernel
            for (int kd = 0; kd < K; ++kd) {
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        // Map output coord to input coord
                        const int d_in_nom = d_out + padding - kd;
                        const int h_in_nom = h_out + padding - kh;
                        const int w_in_nom = w_out + padding - kw;

                        if (d_in_nom >= 0 && h_in_nom >= 0 && w_in_nom >= 0 &&
                            d_in_nom % stride == 0 && h_in_nom % stride == 0 && w_in_nom % stride == 0) {

                            const int d_in = d_in_nom / stride;
                            const int h_in = h_in_nom / stride;
                            const int w_in = w_in_nom / stride;

                            // Check input bounds
                            if (d_in < D_in && h_in < H_in && w_in < W_in) {
                                // Input index
                                const long long input_idx = (long long)n * C_in * D_in * H_in * W_in +
                                                            (long long)c_in * D_in * H_in * W_in +
                                                            (long long)d_in * H_in * W_in +
                                                            (long long)h_in * W_in +
                                                            w_in;

                                // Weight index (shape: C_in, C_out/groups, K, K, K)
                                const long long weight_idx = (long long)c_in * C_out_per_group * K * K * K +
                                                             (long long)c_out_in_group * K * K * K +
                                                             (long long)kd * K * K +
                                                             (long long)kh * K +
                                                             kw;

                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[c_out];
        }

        output[linear_idx] = sum;
    }
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int groups
) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias tensor must be a CUDA tensor");
    }

    const auto x_sizes = x.sizes();
    const int N = x_sizes[0];
    const int C_in = x_sizes[1];
    const int D_in = x_sizes[2];
    const int H_in = x_sizes[3];
    const int W_in = x_sizes[4];

    const auto w_sizes = weight.sizes();
    // weight shape is (C_in, C_out/groups, K, K, K)
    TORCH_CHECK(w_sizes[0] == C_in, "Weight in_channels mismatch");
    const int C_out_per_group = w_sizes[1];
    const int K = w_sizes[2];
    const int C_out = C_out_per_group * groups;

    // Calculate output dimensions
    const int D_out = (D_in - 1) * stride - 2 * padding + K + output_padding;
    const int H_out = (H_in - 1) * stride - 2 * padding + K + output_padding;
    const int W_out = (W_in - 1) * stride - 2 * padding + K + output_padding;

    TORCH_CHECK(D_out > 0 && H_out > 0 && W_out > 0, "Calculated output size is non-positive");

    auto out = torch::zeros({N, C_out, D_out, H_out, W_out}, x.options());

    const long long total_elements = out.numel();
    if (total_elements == 0) {
        return out;
    }

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        (bias.defined() ? bias.data_ptr<float>() : nullptr),
        out.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        K,
        stride, padding, groups,
        total_elements
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int groups
);
"""

# Compile the inline CUDA code
custom_conv_transpose3d_op = load_inline(
    name="custom_conv_transpose3d_op",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Performs a 3D transposed convolution operation with a custom CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Define trainable parameters
        self.weight = nn.Parameter(torch.Tensor(
            in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size
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
        Performs the 3D transposed convolution using the custom CUDA kernel.
        """
        return custom_conv_transpose3d_op.conv_transpose3d_cuda(
            x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups
        )