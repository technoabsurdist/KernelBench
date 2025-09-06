import torch
import torch.nn as nn
import math
from torch.nn import init
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for 2D transposed convolution
conv_transpose_2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA Kernel for 2D Transposed Convolution
__global__ void conv_transpose_2d_kernel(
    const float* input, const float* weight, const float* bias, float* output,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int H_out, const int W_out,
    const int kH, const int kW,
    const int sH, const int sW,
    const int pH, const int pW,
    const int dH, const int dW,
    const int G, const bool has_bias) {

    // Each thread computes one output element
    const long long output_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long num_output_elements = (long long)N * C_out * H_out * W_out;

    if (output_idx >= num_output_elements) {
        return;
    }

    // Decompose output_idx into (n, c_out, h_out, w_out)
    const int w_out = output_idx % W_out;
    const int h_out = (output_idx / W_out) % H_out;
    const int c_out = (output_idx / ((long long)W_out * H_out)) % C_out;
    const int n = output_idx / ((long long)W_out * H_out * C_out);

    // Group calculation
    const int C_out_per_group = C_out / G;
    const int C_in_per_group = C_in / G;
    const int g = c_out / C_out_per_group; // group index
    const int c_out_g = c_out % C_out_per_group; // output channel index within the group

    float sum = 0.0f;

    // Iterate over input channels for this group
    for (int c_in_g = 0; c_in_g < C_in_per_group; ++c_in_g) {
        const int c_in = g * C_in_per_group + c_in_g; // absolute input channel

        // Iterate over kernel
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                // Core logic of transposed convolution: check if an input pixel (h_in, w_in)
                // could have contributed to the current output pixel (h_out, w_out)
                // via this kernel tap (kh, kw).
                // The relationship is: h_out = h_in * sH - pH + dH * kh
                // So, we solve for h_in: h_in = (h_out + pH - dH * kh) / sH
                
                int h_in_nom = h_out + pH - dH * kh;
                int w_in_nom = w_out + pW - dW * kw;

                // Check if the division is exact (i.e., there is a valid integer h_in, w_in)
                if ((h_in_nom % sH == 0) && (w_in_nom % sW == 0)) {
                    int h_in = h_in_nom / sH;
                    int w_in = w_in_nom / sW;

                    // Check if the calculated input coordinates are within bounds
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        // Calculate flat indices
                        // Input: (N, C_in, H_in, W_in)
                        long long input_offset = (long long)n * C_in * H_in * W_in + (long long)c_in * H_in * W_in + (long long)h_in * W_in + w_in;
                        
                        // Weight: (C_in, C_out/G, kH, kW)
                        long long weight_offset = (long long)c_in * C_out_per_group * kH * kW + (long long)c_out_g * kH * kW + (long long)kh * kW + kw;
                        
                        sum += input[input_offset] * weight[weight_offset];
                    }
                }
            }
        }
    }

    if (has_bias) {
        sum += bias[c_out];
    }

    output[output_idx] = sum;
}

// C++ Wrapper and Kernel Launcher
torch::Tensor conv_transpose_2d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::optional<torch::Tensor> bias,
    std::vector<long> stride, std::vector<long> padding, std::vector<long> dilation, long groups) {

    // Input checks
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "Weight must be a float32 tensor");
    
    input = input.contiguous();
    weight = weight.contiguous();

    // Get dimensions
    const long N = input.size(0);
    const long C_in = input.size(1);
    const long H_in = input.size(2);
    const long W_in = input.size(3);

    // Weight shape is (C_in, C_out/G, kH, kW)
    TORCH_CHECK(weight.size(0) == C_in, "Weight in_channels doesn't match input channels");
    const long C_out = weight.size(1) * groups;
    const long kH = weight.size(2);
    const long kW = weight.size(3);

    const long sH = stride[0];
    const long sW = stride[1];
    const long pH = padding[0];
    const long pW = padding[1];
    const long dH = dilation[0];
    const long dW = dilation[1];

    // Calculate output dimensions
    const long H_out = (H_in - 1) * sH - 2 * pH + dH * (kH - 1) + 1;
    const long W_out = (W_in - 1) * sW - 2 * pW + dW * (kW - 1) + 1;

    TORCH_CHECK(H_out > 0 && W_out > 0, "Output size is non-positive. H_out=", H_out, ", W_out=", W_out);

    // Create output tensor
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    // Handle bias
    bool has_bias = bias.has_value();
    torch::Tensor bias_c;
    if (has_bias) {
        bias_c = bias.value().contiguous();
        TORCH_CHECK(bias_c.is_cuda(), "Bias tensor must be a CUDA tensor");
        TORCH_CHECK(bias_c.scalar_type() == torch::kFloat32, "Bias must be a float32 tensor");
        TORCH_CHECK(bias_c.dim() == 1 && bias_c.size(0) == C_out, "Bias has incorrect shape");
    }

    const long long num_output_elements = N * C_out * H_out * W_out;
    const int block_size = 256;
    const int grid_size = (num_output_elements + block_size - 1) / block_size;

    conv_transpose_2d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias_c.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kH, kW,
        sH, sW,
        pH, pW,
        dH, dW,
        groups,
        has_bias
    );
    
    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

conv_transpose_2d_cpp_source = """
#include <torch/extension.h>
#include <vector>

// Forward declaration of the CUDA function
torch::Tensor conv_transpose_2d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::optional<torch::Tensor> bias,
    std::vector<long> stride, std::vector<long> padding, std::vector<long> dilation, long groups);
"""

# Compile the inline CUDA code
custom_conv_transpose_2d = load_inline(
    name="custom_conv_transpose_2d",
    cpp_sources=conv_transpose_2d_cpp_source,
    cuda_sources=conv_transpose_2d_source,
    functions=["conv_transpose_2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Performs a 2D transposed convolution operation using a custom CUDA kernel.
    The arguments and behavior are designed to mimic torch.nn.ConvTranspose2d.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (tuple, optional): Stride of the convolution (height, width). Defaults to (1, 1).
        padding (tuple, optional): Padding applied to the input (height, width). Defaults to (0, 0).
        dilation (tuple, optional): Spacing between kernel elements (height, width). Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Check for valid parameters
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        # PyTorch's ConvTranspose2d weight shape is (in_channels, out_channels / groups, kH, kW)
        self.weight = nn.Parameter(torch.empty(
            in_channels, out_channels // groups, *kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Mimic the initialization of nn.ConvTranspose2d
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # The C++ function expects std::vector<long>, so we convert tuples to lists
        return custom_conv_transpose_2d.conv_transpose_2d_cuda(
            x, self.weight, self.bias, list(self.stride), list(self.padding), list(self.dilation), self.groups
        )