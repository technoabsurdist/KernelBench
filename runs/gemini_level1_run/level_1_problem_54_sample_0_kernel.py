import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for 3D convolution forward pass
__global__ void conv3d_forward_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const long N, const long C_in, const long D, const long H, const long W,
    const long C_out, const long K, const int S, const int P,
    const long D_out, const long H_out, const long W_out,
    const bool has_bias,
    const long total_output_elements) {

    // Using a 1D grid-stride loop to cover all output elements
    for (long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_output_elements;
         idx += blockDim.x * gridDim.x) {

        // Map linear output index to 5D coordinates
        const long w_out = idx % W_out;
        const long h_out = (idx / W_out) % H_out;
        const long d_out = (idx / (W_out * H_out)) % D_out;
        const long c_out = (idx / (W_out * H_out * D_out)) % C_out;
        const long n = idx / (W_out * H_out * D_out * C_out);

        float acc = 0.0f;

        // Iterate over input channels and kernel dimensions
        for (long c_in = 0; c_in < C_in; ++c_in) {
            for (long kd = 0; kd < K; ++kd) {
                for (long kh = 0; kh < K; ++kh) {
                    for (long kw = 0; kw < K; ++kw) {
                        // Calculate input coordinates based on output coordinates, stride, and padding
                        const long d_in = d_out * S - P + kd;
                        const long h_in = h_out * S - P + kh;
                        const long w_in = w_out * S - P + kw;

                        // Check if the input coordinates are within bounds (not in the padded area)
                        if (d_in >= 0 && d_in < D && h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                            // Calculate linear indices for input and weight tensors
                            long input_idx = n * C_in * D * H * W +
                                             c_in * D * H * W +
                                             d_in * H * W +
                                             h_in * W +
                                             w_in;
                            long weight_idx = c_out * C_in * K * K * K +
                                              c_in * K * K * K +
                                              kd * K * K +
                                              kh * K +
                                              kw;
                            
                            acc += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }

        // Add bias if it exists
        if (has_bias) {
            acc += bias[c_out];
        }

        // Store the final result
        output[idx] = acc;
    }
}

// C++ wrapper function to be called from Python
torch::Tensor conv3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias, // Can be an undefined tensor
    int stride,
    int padding) {

    // Ensure tensors are on the correct device and are contiguous
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
    const long N = input.size(0);
    const long C_in = input.size(1);
    const long D = input.size(2);
    const long H = input.size(3);
    const long W = input.size(4);

    const long C_out = weight.size(0);
    const long K = weight.size(2); // Assuming K_d = K_h = K_w

    // Calculate output dimensions
    const long D_out = (D + 2 * padding - K) / stride + 1;
    const long H_out = (H + 2 * padding - K) / stride + 1;
    const long W_out = (W + 2 * padding - K) / stride + 1;

    // Create the output tensor
    auto output = torch::empty({N, C_out, D_out, H_out, W_out}, input.options());

    const long total_output_elements = N * C_out * D_out * H_out * W_out;
    if (total_output_elements == 0) {
        return output;
    }

    // Set up grid and block dimensions for the CUDA kernel
    const int block_size = 256;
    const int num_blocks = std::min((int)((total_output_elements + block_size - 1) / block_size), 4096);

    // Launch the kernel
    conv3d_forward_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, D, H, W,
        C_out, K, stride, padding,
        D_out, H_out, W_out,
        bias.defined(),
        total_output_elements
    );
    
    // Check for any CUDA errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

conv3d_cpp_source = (
    "torch::Tensor conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding);"
)

# Compile the inline CUDA code
custom_conv3d_op = load_inline(
    name="custom_conv3d_op",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with a custom CUDA kernel.
    This implementation replaces nn.Conv3d with a direct convolution kernel.
    NOTE: This custom kernel only supports dilation=1 and groups=1.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Must be 1.
        groups (int, optional): Number of blocked connections. Must be 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        if dilation != 1:
            raise NotImplementedError("Custom Conv3D kernel only supports dilation=1")
        if groups != 1:
            raise NotImplementedError("Custom Conv3D kernel only supports groups=1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # Create learnable parameters, same as in nn.Conv3d
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Replicate PyTorch's default initialization for Conv3d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in > 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        return custom_conv3d_op.conv3d_forward_cuda(
            x, self.weight, self.bias, self.stride, self.padding
        )