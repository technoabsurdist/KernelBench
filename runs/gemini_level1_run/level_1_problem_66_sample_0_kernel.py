import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Helper function to get a pointer from an optional tensor (for bias)
template <typename T>
T* get_optional_ptr(torch::Tensor tensor) {
    return tensor.defined() ? tensor.data_ptr<T>() : nullptr;
}

__global__ void conv3d_forward_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int N, const int C_in, const int D_in, const int H_in, const int W_in,
    const int C_out, const int D_out, const int H_out, const int W_out,
    const int KD, const int KH, const int KW,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int dilation_d, const int dilation_h, const int dilation_w,
    const int groups) {

    // Using a grid-stride loop to handle any number of output elements
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < N * C_out * D_out * H_out * W_out; 
         i += blockDim.x * gridDim.x) {
        
        // Map the linear output index to 5D coordinates
        const int w_out = i % W_out;
        const int h_out = (i / W_out) % H_out;
        const int d_out = (i / (W_out * H_out)) % D_out;
        const int c_out = (i / (W_out * H_out * D_out)) % C_out;
        const int n = i / (W_out * H_out * D_out * C_out);

        // Determine the current group for this output channel
        const int group_idx = c_out / (C_out / groups);
        const int C_in_per_group = C_in / groups;

        // Initialize accumulator with bias if it exists
        float acc = (bias != nullptr) ? bias[c_out] : 0.0f;

        // Iterate over the kernel
        for (int c_in_g = 0; c_in_g < C_in_per_group; ++c_in_g) {
            const int c_in = group_idx * C_in_per_group + c_in_g;
            for (int kd = 0; kd < KD; ++kd) {
                for (int kh = 0; kh < KH; ++kh) {
                    for (int kw = 0; kw < KW; ++kw) {
                        // Calculate corresponding input coordinates
                        const int d_in = d_out * stride_d - pad_d + kd * dilation_d;
                        const int h_in = h_out * stride_h - pad_h + kh * dilation_h;
                        const int w_in = w_out * stride_w - pad_w + kw * dilation_w;

                        // Check if the input coordinates are within the padded bounds
                        if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            // Calculate linear indices for input and weight tensors
                            const long input_idx = n * C_in * D_in * H_in * W_in +
                                                   c_in * D_in * H_in * W_in +
                                                   d_in * H_in * W_in +
                                                   h_in * W_in +
                                                   w_in;
                            const long weight_idx = c_out * C_in_per_group * KD * KH * KW +
                                                    c_in_g * KD * KH * KW +
                                                    kd * KH * KW +
                                                    kh * KW +
                                                    kw;
                            
                            acc += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        output[i] = acc;
    }
}

torch::Tensor conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {

    // Ensure tensors are on CUDA and contiguous for pointer access
    input = input.contiguous();
    weight = weight.contiguous();
    if (bias.defined()) {
        bias = bias.contiguous();
    }

    // Extract dimensions from input tensors
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    const int C_out = weight.size(0);
    const int KD = weight.size(2);
    const int KH = weight.size(3);
    const int KW = weight.size(4);

    // Extract parameters
    const int stride_d = stride[0], stride_h = stride[1], stride_w = stride[2];
    const int pad_d = padding[0], pad_h = padding[1], pad_w = padding[2];
    const int dilation_d = dilation[0], dilation_h = dilation[1], dilation_w = dilation[2];

    // Calculate output dimensions
    const int D_out = (D_in + 2 * pad_d - dilation_d * (KD - 1) - 1) / stride_d + 1;
    const int H_out = (H_in + 2 * pad_h - dilation_h * (KH - 1) - 1) / stride_h + 1;
    const int W_out = (W_in + 2 * pad_w - dilation_w * (KW - 1) - 1) / stride_w + 1;

    // Create the output tensor
    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    const long total_output_elements = N * C_out * D_out * H_out * W_out;
    if (total_output_elements == 0) {
        return output;
    }
    
    // Configure and launch the kernel
    const int block_size = 256;
    const int num_blocks = (total_output_elements + block_size - 1) / block_size;

    conv3d_forward_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        get_optional_ptr<float>(bias),
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        KD, KH, KW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w,
        groups
    );
    
    // Check for any CUDA errors during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA error in conv3d_forward_kernel: ", cudaGetErrorString(err));
    }

    return output;
}
"""

# C++ header for the CUDA function
conv3d_cpp_source = """
torch::Tensor conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups);
"""

# Compile the inline CUDA code using PyTorch's JIT compiler
custom_conv3d_impl = load_inline(
    name="custom_conv3d_impl",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    A 3D convolution module implemented with a custom CUDA kernel.
    This module mimics the behavior of `torch.nn.Conv3d`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store convolution parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Define learnable parameters (weight and bias)
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize weights and bias with standard PyTorch initializations.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using the custom CUDA kernel.
        """
        return custom_conv3d_impl.conv3d_cuda(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )