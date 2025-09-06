import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for 3D transposed convolution forward pass
conv_transpose_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Helper to calculate the flat index for a 5D tensor (N, C, D, H, W)
#define TENSOR_IDX(n, c, d, h, w, C, D, H, W) \
    ((((n * C + c) * D + d) * H + h) * W + w)

__global__ void conv_transpose3d_forward_kernel(
    const float* input,
    const float* weight,
    float* output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int K_d, int K_h, int K_w,
    int S_d, int S_h, int S_w,
    int P_d, int P_h, int P_w) {

    // Calculate the global thread index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    long long num_outputs = (long long)N * C_out * D_out * H_out * W_out;

    if (index >= num_outputs) {
        return;
    }

    // Decompose the 1D index into 5D output coordinates (n, c_out, d_out, h_out, w_out)
    int w_out = index % W_out;
    int h_out = (index / W_out) % H_out;
    int d_out = (index / (W_out * H_out)) % D_out;
    int c_out = (index / (W_out * H_out * D_out)) % C_out;
    int n = index / (W_out * H_out * D_out * C_out);

    float sum = 0.0f;

    // This loop structure is equivalent to the backward data pass of a standard convolution
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kd = 0; kd < K_d; ++kd) {
            for (int kh = 0; kh < K_h; ++kh) {
                for (int kw = 0; kw < K_w; ++kw) {
                    int d_in_unpadded = d_out + P_d - kd;
                    int h_in_unpadded = h_out + P_h - kh;
                    int w_in_unpadded = w_out + P_w - kw;

                    // Check if the current output position corresponds to a strided input position
                    if ((d_in_unpadded % S_d == 0) && (h_in_unpadded % S_h == 0) && (w_in_unpadded % S_w == 0)) {
                        int d_in = d_in_unpadded / S_d;
                        int h_in = h_in_unpadded / S_h;
                        int w_in = w_in_unpadded / S_w;

                        // Check if the input coordinates are within bounds
                        if ((d_in >= 0 && d_in < D_in) && (h_in >= 0 && h_in < H_in) && (w_in >= 0 && w_in < W_in)) {
                            long long input_idx = TENSOR_IDX(n, c_in, d_in, h_in, w_in, C_in, D_in, H_in, W_in);
                            // Weight shape is (C_in, C_out, K_d, K_h, K_w)
                            long long weight_idx = TENSOR_IDX(c_in, c_out, kd, kh, kw, C_out, K_d, K_h, K_w);
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    output[index] = sum;
}

torch::Tensor conv_transpose3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    std::vector<long> stride,
    std::vector<long> padding,
    std::vector<long> output_padding) {

    // Input dimensions
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    // Weight dimensions (C_in, C_out, K_d, K_h, K_w)
    const int C_out = weight.size(1);
    const int K_d = weight.size(2);
    const int K_h = weight.size(3);
    const int K_w = weight.size(4);

    // Stride, padding
    const int S_d = stride[0], S_h = stride[1], S_w = stride[2];
    const int P_d = padding[0], P_h = padding[1], P_w = padding[2];
    const int OP_d = output_padding[0], OP_h = output_padding[1], OP_w = output_padding[2];

    // Calculate output dimensions
    const int D_out = (D_in - 1) * S_d - 2 * P_d + K_d + OP_d;
    const int H_out = (H_in - 1) * S_h - 2 * P_h + K_h + OP_h;
    const int W_out = (W_in - 1) * S_w - 2 * P_w + K_w + OP_w;

    // Create output tensor
    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    // Launch kernel
    long long num_outputs = (long long)N * C_out * D_out * H_out * W_out;
    if (num_outputs == 0) {
        return output;
    }
    const int block_size = 256;
    const int num_blocks = (num_outputs + block_size - 1) / block_size;

    conv_transpose3d_forward_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        K_d, K_h, K_w,
        S_d, S_h, S_w,
        P_d, P_h, P_w
    );

    return output;
}
"""

conv_transpose_3d_cpp_source = """
torch::Tensor conv_transpose3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    std::vector<long> stride,
    std::vector<long> padding,
    std::vector<long> output_padding);
"""

# Compile the inline CUDA code
# This is a global operation, so it's done once when the script is loaded.
conv_transpose_3d_op = load_inline(
    name="conv_transpose_3d_op",
    cpp_sources=conv_transpose_3d_cpp_source,
    cuda_sources=conv_transpose_3d_source,
    functions=["conv_transpose3d_forward_cuda"],
    verbose=False,
)

class ConvTranspose3dCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding, output_padding):
        # Save context for backward pass
        ctx.stride = stride
        ctx.padding = padding
        ctx.save_for_backward(input, weight)

        # Call the custom CUDA kernel for the forward pass
        output = conv_transpose_3d_op.conv_transpose3d_forward_cuda(
            input, weight, stride, padding, output_padding
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors and parameters
        input, weight = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding

        grad_input = grad_weight = None

        # Gradient w.r.t. input is a standard 3D convolution
        if ctx.needs_input_grad[0]:
            # For conv3d, weight shape is (C_out, C_in, K...).
            # Our weight shape is (C_in, C_out, K...). So we transpose.
            grad_input = F.conv3d(grad_output, weight.transpose(0, 1), stride=stride, padding=padding)

        # Gradient w.r.t. weight is another 3D convolution
        if ctx.needs_input_grad[1]:
            # This is equivalent to convolving the input with the grad_output,
            # but requires permuting dimensions to match the conv3d API.
            # input becomes the "data", grad_output becomes the "kernel".
            # Permute (N, C, D, H, W) -> (C, N, D, H, W)
            input_permuted = input.permute(1, 0, 2, 3, 4)
            grad_output_permuted = grad_output.permute(1, 0, 2, 3, 4)
            
            # The output of this conv will have shape (C_in, C_out, K, K, K), which is correct.
            grad_weight = F.conv3d(input_permuted, grad_output_permuted, stride=stride, padding=padding)

        return grad_input, grad_weight, None, None, None


class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution using a custom CUDA kernel for the forward pass.
    The backward pass leverages PyTorch's native `conv3d` for efficiency and correctness.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of 3 integers representing the kernel size.
        stride (tuple, optional): Tuple of 3 integers representing the stride. Defaults to (1, 1, 1).
        padding (tuple, optional): Tuple of 3 integers representing the padding. Defaults to (0, 0, 0).
        output_padding (tuple, optional): Tuple of 3 integers representing the output padding. Defaults to (0, 0, 0).
        groups (int, optional): Not supported by this custom kernel, must be 1.
        bias (bool, optional): Not supported by this custom kernel, must be `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        if groups != 1:
            raise NotImplementedError("Custom CUDA kernel only supports groups=1")
        if bias:
            raise NotImplementedError("Custom CUDA kernel does not support bias")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # The weight parameter for ConvTranspose3d has shape (in_channels, out_channels, K_d, K_h, K_w)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, *kernel_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Standard kaiming uniform initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the custom transposed 3D convolution.
        """
        return ConvTranspose3dCudaFunction.apply(x, self.weight, self.stride, self.padding, self.output_padding)