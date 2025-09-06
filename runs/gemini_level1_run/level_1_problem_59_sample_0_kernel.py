import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel source and the C++ wrapper function
# This is done at the module level to avoid recompilation every time a ModelNew instance is created.
conv_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Naive direct convolution kernel specialized for 3D input with a 2D (K, K, 1) kernel.
// This is equivalent to applying the same 2D convolution to each depth slice.
// This implementation assumes groups = 1.
__global__ void conv2d_depthwise_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    const int N, const int C_in, const int H, const int W, const int D,
    const int C_out, const int K,
    const int H_out, const int W_out,
    const int stride, const int padding, const int dilation) {

    // Grid is configured as (ceil(W_out/TILE), ceil(H_out/TILE), N * C_out * D)
    // Each block computes a tile of the output for a given (n, c_out, d)
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int z_idx = blockIdx.z;

    if (w_out >= W_out || h_out >= H_out) {
        return;
    }

    // Decompose z_idx into n, c_out, d
    const int d = z_idx % D;
    const int c_out = (z_idx / D) % C_out;
    const int n = z_idx / (D * C_out);

    // Calculate input starting position for the top-left corner of the receptive field
    const int h_in_start = h_out * stride - padding;
    const int w_in_start = w_out * stride - padding;

    float acc = 0.0f;

    // Loop over input channels and kernel dimensions
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K; ++kh) {
            const int h_in = h_in_start + kh * dilation;
            if (h_in >= 0 && h_in < H) {
                for (int kw = 0; kw < K; ++kw) {
                    const int w_in = w_in_start + kw * dilation;
                    if (w_in >= 0 && w_in < W) {
                        // Calculate flat indices for x and weight
                        // Using long long to prevent overflow for large tensors
                        const long long x_idx = (long long)n * C_in * H * W * D +
                                                (long long)c_in * H * W * D +
                                                (long long)h_in * W * D +
                                                (long long)w_in * D +
                                                d;
                        // Kernel depth is 1, so last dim is 0
                        const long long w_idx = (long long)c_out * C_in * K * K +
                                                (long long)c_in * K * K +
                                                (long long)kh * K +
                                                kw;
                        acc += x[x_idx] * weight[w_idx];
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        acc += bias[c_out];
    }

    // Calculate flat index for out and write result
    const long long out_idx = (long long)n * C_out * H_out * W_out * D +
                              (long long)c_out * H_out * W_out * D +
                              (long long)h_out * W_out * D +
                              (long long)w_out * D +
                              d;
    out[out_idx] = acc;
}

torch::Tensor conv2d_depthwise_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias_opt,
    int stride, int padding, int dilation, int groups) {

    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be a CUDA tensor");
    
    // For simplicity in this kernel, we ensure contiguous tensors.
    // A more robust implementation might handle non-contiguous inputs directly.
    x = x.contiguous();
    weight = weight.contiguous();

    TORCH_CHECK(groups == 1, "Custom kernel currently only supports groups=1");

    const int N = x.size(0);
    const int C_in = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int D = x.size(4);

    const int C_out = weight.size(0);
    const int K = weight.size(2);

    TORCH_CHECK(weight.size(4) == 1, "Kernel depth must be 1 for this specialized kernel");
    TORCH_CHECK(weight.size(2) == weight.size(3), "Kernel height and width must be equal");

    const int H_out = (H + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    const int W_out = (W + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    auto out = torch::zeros({N, C_out, H_out, W_out, D}, x.options());

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value().contiguous();
        TORCH_CHECK(bias.is_cuda(), "Bias tensor must be a CUDA tensor");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C_out, "Bias has wrong shape");
        bias_ptr = bias.data_ptr<float>();
    }

    const int TILE_DIM = 16;
    dim3 threads(TILE_DIM, TILE_DIM, 1);
    dim3 blocks(
        (W_out + TILE_DIM - 1) / TILE_DIM,
        (H_out + TILE_DIM - 1) / TILE_DIM,
        (long long)N * C_out * D
    );

    // Check for grid size limitations on the Z dimension
    int max_grid_dim_z;
    cudaDeviceGetAttribute(&max_grid_dim_z, cudaDevAttrMaxGridDimZ, 0);
    if (blocks.z > max_grid_dim_z) {
        AT_ERROR("Grid Z dimension (", blocks.z, ") exceeds device limit (", max_grid_dim_z, ")");
    }

    conv2d_depthwise_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        out.data_ptr<float>(),
        N, C_in, H, W, D,
        C_out, K,
        H_out, W_out,
        stride, padding, dilation
    );

    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# Define the C++ source for the function signature
conv_cpp_source = "torch::Tensor conv2d_depthwise_cuda(torch::Tensor x, torch::Tensor weight, torch::optional<torch::Tensor> bias, int stride, int padding, int dilation, int groups);"

# Load the inline CUDA kernel
custom_conv_op = load_inline(
    name="custom_conv_op_specialized", # Use a unique name to avoid conflicts
    cpp_sources=conv_cpp_source,
    cuda_sources=conv_kernel_source,
    functions=["conv2d_depthwise_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with an asymmetric input and a square kernel,
    using a custom CUDA kernel for the convolution.
    The custom kernel is specialized for convolutions where the kernel depth is 1.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel (kernel_size x kernel_size).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        if self.groups != 1:
            raise NotImplementedError("Custom CUDA kernel only supports groups=1")

        # Define learnable parameters, matching the shape of the original nn.Conv3d
        # The kernel depth is hardcoded to 1 as in the original model.
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Replicate the default initialization of nn.Conv3d for fair comparison
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            # Calculate fan_in for the 2D kernel part
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[:,:,:,:,0])
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out, depth_out).
        """
        return custom_conv_op.conv2d_depthwise_cuda(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )