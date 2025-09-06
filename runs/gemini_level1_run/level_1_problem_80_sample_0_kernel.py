import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 2D convolution
conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(
    const float* x,          // Input tensor data
    const float* weight,     // Weight tensor data
    const float* bias,       // Bias tensor data (can be nullptr)
    float* out,              // Output tensor data
    // Dimensions
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    // Conv parameters
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dil_h, int dil_w
) {
    // Calculate the output coordinates for this thread
    // Each thread computes one output pixel (h_out, w_out) for one output channel and batch item.
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int n_cout_idx = blockIdx.z; // Combined batch and out_channel index

    // Early exit if the thread is out of bounds for the output tensor
    if (w_out >= W_out || h_out >= H_out) {
        return;
    }

    // De-multiplex the combined batch and output channel index
    int n = n_cout_idx / C_out;
    int c_out = n_cout_idx % C_out;

    // Accumulator for the dot product
    float acc = 0.0f;

    // Iterate over input channels, kernel height, and kernel width
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int k_h = 0; k_h < K_h; ++k_h) {
            for (int k_w = 0; k_w < K_w; ++k_w) {
                // Calculate corresponding input coordinates
                int h_in = h_out * stride_h - pad_h + k_h * dil_h;
                int w_in = w_out * stride_w - pad_w + k_w * dil_w;

                // Check if the input coordinates are within the valid (non-padded) input area
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    // Calculate flat indices for input and weight tensors (row-major layout)
                    // Input tensor shape: (N, C_in, H_in, W_in)
                    long long x_idx = (long long)n * C_in * H_in * W_in +
                                      (long long)c_in * H_in * W_in +
                                      (long long)h_in * W_in +
                                      w_in;
                    // Weight tensor shape: (C_out, C_in, K_h, K_w)
                    long long w_idx = (long long)c_out * C_in * K_h * K_w +
                                      (long long)c_in * K_h * K_w +
                                      (long long)k_h * K_w +
                                      k_w;
                    
                    acc += x[x_idx] * weight[w_idx];
                }
            }
        }
    }

    // Add bias if it exists
    if (bias != nullptr) {
        acc += bias[c_out];
    }

    // Calculate flat index for the output tensor and write the result
    // Output tensor shape: (N, C_out, H_out, W_out)
    long long out_idx = (long long)n * C_out * H_out * W_out +
                        (long long)c_out * H_out * W_out +
                        (long long)h_out * W_out +
                        w_out;
    out[out_idx] = acc;
}

// C++ wrapper function to be called from Python
torch::Tensor conv2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias_opt,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef dilation
) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Input weight must be a CUDA tensor");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt->is_cuda(), "Input bias must be a CUDA tensor");
    }
    TORCH_CHECK(x.dim() == 4, "Input x must be a 4D tensor");
    TORCH_CHECK(weight.dim() == 4, "Input weight must be a 4D tensor");

    // Ensure tensors are contiguous in memory for direct pointer access
    x = x.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias;
    if (bias_opt.has_value()) {
        bias = bias_opt->contiguous();
    }

    // Extract dimensions from input tensors
    const int N = x.size(0);
    const int C_in = x.size(1);
    const int H_in = x.size(2);
    const int W_in = x.size(3);

    const int C_out = weight.size(0);
    const int K_h = weight.size(2);
    const int K_w = weight.size(3);

    // Extract convolution parameters
    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int pad_h = padding[0];
    const int pad_w = padding[1];
    const int dil_h = dilation[0];
    const int dil_w = dilation[1];

    // Calculate output dimensions
    const int H_out = (H_in + 2 * pad_h - dil_h * (K_h - 1) - 1) / stride_h + 1;
    const int W_out = (W_in + 2 * pad_w - dil_w * (K_w - 1) - 1) / stride_w + 1;

    // Create the output tensor, initialized to zeros
    auto out = torch::zeros({N, C_out, H_out, W_out}, x.options());

    // Configure CUDA grid and block dimensions
    const dim3 threads(16, 16, 1); // 256 threads per block, organized in 2D
    const dim3 blocks(
        (W_out + threads.x - 1) / threads.x,
        (H_out + threads.y - 1) / threads.y,
        (long long)N * C_out // Each z-dimension block handles one (batch, out_channel) pair
    );

    // Launch the CUDA kernel
    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_opt.has_value() ? bias.data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w
    );
    
    // Check for any CUDA errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return out;
}
"""

# C++ source for the function signature
conv2d_cpp_source = """
torch::Tensor conv2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias_opt,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef dilation
);
"""

# Compile the inline CUDA code
custom_conv2d_impl = load_inline(
    name="custom_conv2d_impl",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution using a custom CUDA kernel.
    The interface is identical to the original Model.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store convolution parameters, ensuring they are tuples
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = padding
        self.dilation = dilation

        # Define learnable parameters (weight and optional bias)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            # Register 'bias' as None if not used, which is standard PyTorch practice
            self.register_parameter('bias', None)
        
        # Initialize parameters to match PyTorch's default Conv2d initialization
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes weight and bias using PyTorch's default methods."""
        # Kaiming uniform initialization for the weight tensor
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            # Calculate fan_in and set bias bounds accordingly
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using the custom CUDA kernel.
        """
        # Ensure input is on the same device as the layer's weights
        if x.device != self.weight.device:
            x = x.to(self.weight.device)
            
        return custom_conv2d_impl.conv2d_cuda(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation
        )