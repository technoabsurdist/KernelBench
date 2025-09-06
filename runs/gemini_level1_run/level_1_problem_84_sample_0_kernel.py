import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for depthwise 2D convolution
depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* input, 
    const float* weight, 
    float* output,
    int N, int C, int H_in, int W_in,
    int K, int H_out, int W_out,
    int stride, int padding) {

    // Calculate the output pixel (n, c, h_out, w_out) this thread will compute
    // Using a 2D grid for the spatial dimensions and a 1D grid for batch*channels
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z; // Index for batch and channel combined

    // Ensure we are within the output bounds
    if (w_out >= W_out || h_out >= H_out) {
        return;
    }

    // Decompose nc into n (batch) and c (channel)
    int n = nc / C;
    int c = nc % C;

    // Accumulator for the convolution sum
    float acc = 0.0f;

    // Loop over the kernel
    for (int kh = 0; kh < K; ++kh) {
        for (int kw = 0; kw < K; ++kw) {
            // Calculate input coordinates
            int h_in = h_out * stride - padding + kh;
            int w_in = w_out * stride - padding + kw;

            // Boundary check for input
            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                // Calculate flat indices for contiguous tensors
                long input_idx = n * C * H_in * W_in + c * H_in * W_in + h_in * W_in + w_in;
                // For depthwise, weight is (C, 1, K, K)
                long weight_idx = c * K * K + kh * K + kw;
                
                acc += input[input_idx] * weight[weight_idx];
            }
        }
    }

    // Write the result to the output tensor
    long output_idx = n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out;
    output[output_idx] = acc;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor x, 
    torch::Tensor weight, 
    int stride, 
    int padding) {
    
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "Weight must be a float32 tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor");
    // For depthwise conv (groups = in_channels), weight shape is (out_channels, 1, K, K)
    TORCH_CHECK(x.size(1) == weight.size(0), "Input channels must match output channels for this depthwise kernel");
    TORCH_CHECK(weight.size(1) == 1, "Weight tensor channel multiplier must be 1 for depthwise conv");

    // Get dimensions
    const int N = x.size(0);
    const int C = x.size(1); // in_channels == out_channels
    const int H_in = x.size(2);
    const int W_in = x.size(3);
    const int K = weight.size(2);

    // Calculate output dimensions
    const int H_out = (H_in + 2 * padding - K) / stride + 1;
    const int W_out = (W_in + 2 * padding - K) / stride + 1;

    // Create output tensor
    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    // Set up grid and block dimensions
    const dim3 block_dim(16, 16, 1);
    const dim3 grid_dim(
        (W_out + block_dim.x - 1) / block_dim.x,
        (H_out + block_dim.y - 1) / block_dim.y,
        N * C
    );

    // Launch the kernel
    depthwise_conv2d_kernel<<<grid_dim, block_dim>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        K, H_out, W_out,
        stride, padding
    );
    
    // Check for any CUDA errors after kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

depthwise_conv_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(torch::Tensor x, torch::Tensor weight, int stride, int padding);
"""

# Compile the inline CUDA code for the depthwise convolution
custom_depthwise_conv = load_inline(
    name="custom_depthwise_conv",
    cpp_sources=depthwise_conv_cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with a custom CUDA kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution. Must be equal to in_channels.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # For a standard depthwise convolution, in_channels must equal out_channels
        # and groups is equal to in_channels. Our custom kernel is built on this assumption.
        if in_channels != out_channels:
            raise ValueError("in_channels must equal out_channels for this custom depthwise convolution implementation.")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # The weight for a depthwise convolution has shape (out_channels, 1, kernel_size, kernel_size)
        # because each output channel depends only on one input channel.
        self.weight = nn.Parameter(torch.randn(out_channels, 1, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Custom kernels require contiguous tensors
        x = x.contiguous()
        
        output = custom_depthwise_conv.depthwise_conv2d_cuda(x, self.weight, self.stride, self.padding)
        
        if self.bias is not None:
            # Reshape bias to (1, C, 1, 1) for broadcasting
            output = output + self.bias.view(1, -1, 1, 1)
            
        return output