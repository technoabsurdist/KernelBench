import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel source for depthwise 2D convolution
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int N, int C, int H, int W,
    int K, int stride, int padding,
    int H_out, int W_out) {

    // Use a grid-stride loop to ensure all output elements are processed
    // regardless of the number of blocks launched.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < N * C * H_out * W_out; 
         i += gridDim.x * blockDim.x) {
        
        // Decompose the linear index 'i' to 4D output coordinates (n, c, h_out, w_out)
        int w_out = i % W_out;
        int h_out = (i / W_out) % H_out;
        int c = (i / (W_out * H_out)) % C;
        int n = i / (W_out * H_out * C);

        float acc = 0.0f;

        // Perform the convolution for the current output element
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                // Calculate the corresponding input coordinates
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;

                // Check if the input coordinates are within the valid bounds of the input tensor
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    // Calculate linear indices for input and weight tensors
                    // Input tensor is in NCHW format
                    int x_idx = n * C * H * W + c * H * W + h_in * W + w_in;
                    // Weight tensor for depthwise conv is (C_out, 1, K, K), where C_out = C
                    int w_idx = c * K * K + kh * K + kw;
                    
                    acc += x[x_idx] * weight[w_idx];
                }
            }
        }

        // Add bias if it is provided
        if (bias != nullptr) {
            acc += bias[c];
        }

        out[i] = acc;
    }
}

// C++ wrapper function that interfaces with PyTorch
torch::Tensor depthwise_conv2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding) {

    // Input validation checks
    TORCH_CHECK(x.is_cuda(), "Input tensor 'x' must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Input tensor 'weight' must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input 'x' must be a float32 tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "Input 'weight' must be a float32 tensor");
    
    bool has_bias = bias.defined() && bias.numel() > 0;
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "Input tensor 'bias' must be a CUDA tensor");
        TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "Input 'bias' must be a float32 tensor");
    }

    // Ensure tensors are contiguous in memory for correct pointer arithmetic
    x = x.contiguous();
    weight = weight.contiguous();
    if (has_bias) {
        bias = bias.contiguous();
    }

    // Extract dimensions from input tensors
    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int K = weight.size(2); // Kernel size from weight tensor (C_out, 1, K, K)

    // Calculate output dimensions
    const int H_out = (H + 2 * padding - K) / stride + 1;
    const int W_out = (W + 2 * padding - K) / stride + 1;

    // Create the output tensor, initialized to zeros
    auto out = torch::zeros({N, C, H_out, W_out}, x.options());

    const long long total_output_elements = (long long)N * C * H_out * W_out;
    if (total_output_elements == 0) {
        return out; // Return empty tensor if output size is zero
    }

    // Configure and launch the CUDA kernel
    const int block_size = 256;
    const int num_blocks = (total_output_elements + block_size - 1) / block_size;

    // Get raw data pointers
    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = has_bias ? bias.data_ptr<float>() : nullptr;
    float* out_ptr = out.data_ptr<float>();

    depthwise_conv2d_kernel<<<num_blocks, block_size>>>(
        x_ptr, weight_ptr, bias_ptr, out_ptr,
        N, C, H, W, K, stride, padding, H_out, W_out
    );
    
    // Check for any errors during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return out;
}
"""

# C++ source for function signature declaration
depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding);
"""

# Use torch.utils.cpp_extension.load_inline to JIT compile the CUDA code
depthwise_conv2d_cuda_module = load_inline(
    name="depthwise_conv2d_cuda_module",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution using a custom CUDA kernel.
    The interface is identical to the original Model.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Define learnable parameters (weight and optional bias)
        # For depthwise convolution, weight shape is (out_channels, 1, kH, kW),
        # where out_channels = in_channels.
        self.weight = nn.Parameter(torch.Tensor(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_channels))
        else:
            # Register bias as None if it's not used
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize parameters using the same method as standard PyTorch layers
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        # If bias is not used, pass an empty tensor to the CUDA function.
        # The C++ wrapper is designed to handle this case.
        bias_tensor = self.bias if self.bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        
        return depthwise_conv2d_cuda_module.depthwise_conv2d_forward(
            x, self.weight, bias_tensor, self.stride, self.padding
        )