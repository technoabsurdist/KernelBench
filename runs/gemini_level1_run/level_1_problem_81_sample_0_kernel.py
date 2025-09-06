import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ConvTranspose2d
conv_transpose2d_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for the main transposed convolution computation using atomicAdd
__global__ void conv_transpose2d_kernel(
    const float* input, const float* weight, float* output,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int K,
    const int H_out, const int W_out,
    const int stride, const int padding, const int dilation) {

    const int total_input_elements = N * C_in * H_in * W_in;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_input_elements) {
        return;
    }

    // Decompose 1D index back to 4D tensor coordinates (n, c_in, h_in, w_in)
    const int w_in = idx % W_in;
    const int h_in = (idx / W_in) % H_in;
    const int c_in = (idx / (W_in * H_in)) % C_in;
    const int n = idx / (W_in * H_in * C_in);

    const float input_val = input[idx];

    // This single input element contributes to a KxK region in the output for each output channel.
    // We "scatter" the weighted input value to the output tensor.
    for (int c_out = 0; c_out < C_out; ++c_out) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                const int h_out = h_in * stride + kh * dilation - padding;
                const int w_out = w_in * stride + kw * dilation - padding;

                // Boundary check to ensure we are writing within the output tensor
                if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out) {
                    // Weight tensor layout: (C_in, C_out, K, K)
                    const int weight_idx = c_in * (C_out * K * K) + c_out * (K * K) + kh * K + kw;
                    // Output tensor layout: (N, C_out, H_out, W_out)
                    const int output_idx = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out;

                    // Use atomicAdd to prevent race conditions from multiple threads writing to the same output pixel
                    atomicAdd(&output[output_idx], input_val * weight[weight_idx]);
                }
            }
        }
    }
}

// CUDA kernel to add the bias term
__global__ void add_bias_kernel(
    float* output, const float* bias,
    const int N, const int C_out, const int H_out, const int W_out) {

    const int total_output_elements = N * C_out * H_out * W_out;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_output_elements) {
        return;
    }

    // Find the corresponding channel for the current output element
    const int c_out = (idx / (H_out * W_out)) % C_out;
    output[idx] += bias[c_out];
}


// Launcher function for the main kernel
void conv_transpose2d_cuda_launcher(
    torch::Tensor input, torch::Tensor weight, torch::Tensor output,
    int stride, int padding, int dilation) {

    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    const int C_out = weight.size(1);
    const int K = weight.size(2);
    const int H_out = output.size(2);
    const int W_out = output.size(3);

    const int total_input_elements = N * C_in * H_in * W_in;
    const int block_size = 256;
    const int num_blocks = (total_input_elements + block_size - 1) / block_size;

    conv_transpose2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        N, C_in, H_in, W_in, C_out, K, H_out, W_out,
        stride, padding, dilation
    );
}

// Launcher function for the bias kernel
void add_bias_cuda_launcher(
    torch::Tensor output, torch::Tensor bias) {

    const int N = output.size(0);
    const int C_out = output.size(1);
    const int H_out = output.size(2);
    const int W_out = output.size(3);

    const int total_output_elements = N * C_out * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_output_elements + block_size - 1) / block_size;

    add_bias_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), bias.data_ptr<float>(),
        N, C_out, H_out, W_out
    );
}

// C++ interface function that will be bound to Python
torch::Tensor conv_transpose2d_custom(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation) {

    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor");

    // Get dimensions
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    // PyTorch weight format for ConvTranspose2d is (C_in, C_out, K_h, K_w)
    TORCH_CHECK(weight.size(0) == C_in, "Weight C_in mismatch");
    const int C_out = weight.size(1);
    const int K_h = weight.size(2);
    const int K_w = weight.size(3);
    TORCH_CHECK(K_h == K_w, "Only square kernels are supported by this custom op");

    // Calculate output dimensions (assuming output_padding = 0)
    const int H_out = (H_in - 1) * stride - 2 * padding + dilation * (K_h - 1) + 1;
    const int W_out = (W_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;

    // Create output tensor, initialized to zero for atomicAdd
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    // Call the CUDA kernel launcher
    conv_transpose2d_cuda_launcher(input, weight, output, stride, padding, dilation);

    // Add bias if it exists
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias.value().dim() == 1 && bias.value().size(0) == C_out, "Bias shape mismatch");
        add_bias_cuda_launcher(output, bias.value());
    }

    return output;
}
"""

conv_transpose2d_cpp_source = """
#include <torch/extension.h>
#include <vector>

// Forward declaration of the function implemented in the CUDA source
torch::Tensor conv_transpose2d_custom(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation);
"""

# Compile the inline CUDA code
# This is done once when the script is loaded.
conv_transpose_op = load_inline(
    name="conv_transpose_op",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_cuda_source,
    functions=["conv_transpose2d_custom"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Performs a 2D transposed convolution operation using a custom CUDA kernel.
    The implementation keeps the original nn.ConvTranspose2d layer to manage
    the weight and bias parameters, but overrides the forward pass to call
    the custom kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel (square, e.g., 3 for a 3x3 kernel).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We instantiate the original layer to handle parameter registration,
        # initialization, and storage.
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

        # Store hyper-parameters to pass them to the CUDA kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Assign the compiled custom operator
        self.custom_op = conv_transpose_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # The custom kernel expects contiguous tensors
        x_contiguous = x.contiguous()
        weight_contiguous = self.conv_transpose2d.weight.contiguous()
        bias_contiguous = self.conv_transpose2d.bias.contiguous() if self.conv_transpose2d.bias is not None else None

        return self.custom_op.conv_transpose2d_custom(
            x_contiguous,
            weight_contiguous,
            bias_contiguous,
            self.stride,
            self.padding,
            self.dilation
        )