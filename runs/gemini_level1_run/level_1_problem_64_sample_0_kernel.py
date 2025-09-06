import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for 1D transposed convolution
conv_transpose1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for the forward pass of 1D transposed convolution
// This kernel is designed from the output's perspective (a "gather" operation)
// to avoid atomic operations. Each thread computes a single output element.
__global__ void conv_transpose1d_forward_kernel(
    const float* input,
    const float* weight,
    float* output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int K, int stride, int padding) {

    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N * C_out * L_out;

    if (idx >= total_outputs) {
        return;
    }

    // Decode the 3D index (n, c_out, l_out) from the 1D global index
    int l_out_idx = idx % L_out;
    int c_out_idx = (idx / L_out) % C_out;
    int n_idx = idx / (L_out * C_out);

    float sum = 0.0f;

    // This operation is equivalent to the backward pass of a standard convolution
    // with respect to its input. We iterate over the input channels and the kernel
    // to find which input elements contribute to the current output element.
    for (int c_in_idx = 0; c_in_idx < C_in; ++c_in_idx) {
        for (int k_idx = 0; k_idx < K; ++k_idx) {
            // The relationship between input and output indices in a standard convolution is:
            // input_pos = output_pos * stride + kernel_pos - padding
            // For the transpose, we are calculating the value at `input_pos` (here, l_out_idx)
            // by summing contributions from `output_pos` (here, l_in_idx).
            // So, l_out_idx = l_in_idx * stride + k_idx - padding
            // We need to find l_in_idx: l_in_idx = (l_out_idx + padding - k_idx) / stride
            int numerator = l_out_idx + padding - k_idx;

            // Check if the contribution is valid:
            // 1. The numerator must be non-negative.
            // 2. The numerator must be perfectly divisible by the stride.
            if (numerator >= 0 && numerator % stride == 0) {
                int l_in_idx = numerator / stride;

                // 3. The calculated input index must be within the bounds of the input length.
                if (l_in_idx < L_in) {
                    // Calculate flat indices for input and weight tensors
                    int input_flat_idx = n_idx * C_in * L_in + c_in_idx * L_in + l_in_idx;
                    // Weight tensor layout: (in_channels, out_channels, kernel_size)
                    int weight_flat_idx = c_in_idx * C_out * K + c_out_idx * K + k_idx;

                    sum += input[input_flat_idx] * weight[weight_flat_idx];
                }
            }
        }
    }

    output[idx] = sum;
}

// C++ wrapper function to be called from Python
torch::Tensor conv_transpose1d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding) {

    // Ensure tensors are on the correct device and are contiguous
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be a CUDA tensor");
    input = input.contiguous();
    weight = weight.contiguous();

    // Get tensor dimensions
    const auto N = input.size(0);
    const auto C_in = input.size(1);
    const auto L_in = input.size(2);

    // Weight shape is (in_channels, out_channels, kernel_size)
    TORCH_CHECK(weight.size(0) == C_in, "Weight tensor in_channels mismatch");
    const auto C_out = weight.size(1);
    const auto K = weight.size(2);

    // Calculate output length using the formula from PyTorch docs
    // L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
    // Assuming dilation = 1
    const int L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;

    // Create the output tensor, initialized to zeros
    auto output = torch::zeros({N, C_out, L_out}, input.options());

    // Set up grid and block dimensions for the CUDA kernel
    const int total_outputs = N * C_out * L_out;
    if (total_outputs == 0) {
        return output;
    }
    const int block_size = 256;
    const int num_blocks = (total_outputs + block_size - 1) / block_size;

    // Launch the kernel
    conv_transpose1d_forward_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out, L_in, L_out, K, stride, padding
    );
    
    // Check for any CUDA errors after kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

# C++ source for defining the function signature for the PyTorch binding
conv_transpose1d_cpp_source = """
torch::Tensor conv_transpose1d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding
);
"""

# Use torch.utils.cpp_extension.load_inline to compile the CUDA code
# This creates a Python module on the fly that can be called.
custom_conv_transpose1d_op = load_inline(
    name="custom_conv_transpose1d_op",
    cpp_sources=conv_transpose1d_cpp_source,
    cuda_sources=conv_transpose1d_source,
    functions=["conv_transpose1d_forward_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation using a custom CUDA kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # The custom kernel has limitations, so we assert them here.
        if groups != 1:
            raise NotImplementedError("Custom CUDA kernel only supports groups=1")
        if bias:
            raise NotImplementedError("Custom CUDA kernel does not support bias")

        # Store convolution parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        
        # Manually define the weight as a learnable parameter, similar to how nn.ConvTranspose1d would.
        # The weight shape for ConvTranspose1d is (in_channels, out_channels, kernel_size).
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        # A proper initialization is important for training. We mimic the default kaiming_uniform_.
        nn.init.kaiming_uniform_(self.weight, a=pow(5, 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return custom_conv_transpose1d_op.conv_transpose1d_forward_cuda(
            x, self.weight, self.stride, self.padding, self.output_padding
        )