import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for 3D transposed convolution
conv_transpose3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    const int N, const int C_in, const int D_in, const int H_in, const int W_in,
    const int C_out, const int D_out, const int H_out, const int W_out,
    const int KD, const int KH, const int KW,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int groups) {

    // Calculate total number of output elements
    const long long total_outputs = (long long)N * C_out * D_out * H_out * W_out;
    // Get the global thread index
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= total_outputs) {
        return;
    }

    // Decompose index into 5D coordinates for the output tensor
    const int w_out = index % W_out;
    const int h_out = (index / W_out) % H_out;
    const int d_out = (index / (W_out * H_out)) % D_out;
    const int c_out = (index / (W_out * H_out * D_out)) % C_out;
    const int n = index / (W_out * H_out * D_out * C_out);

    // Determine the group for this output channel
    const int C_out_per_group = C_out / groups;
    const int group_idx = c_out / C_out_per_group;
    const int c_out_in_group = c_out % C_out_per_group;

    const int C_in_per_group = C_in / groups;
    const int c_in_start = group_idx * C_in_per_group;
    const int c_in_end = c_in_start + C_in_per_group;

    float sum = 0.0f;

    // Iterate over input channels within the group
    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        // Iterate over the kernel
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    // This is the "gathering" part. Find which input pixel corresponds 
                    // to this output pixel and kernel tap.
                    const int d_in_nom = d_out + pad_d - kd;
                    const int h_in_nom = h_out + pad_h - kh;
                    const int w_in_nom = w_out + pad_w - kw;

                    // Check if the mapping is valid (divisible by stride)
                    if ((d_in_nom % stride_d == 0) && (h_in_nom % stride_h == 0) && (w_in_nom % stride_w == 0)) {
                        const int d_in = d_in_nom / stride_d;
                        const int h_in = h_in_nom / stride_h;
                        const int w_in = w_in_nom / stride_w;

                        // Check if the calculated input coordinates are within bounds
                        if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            // Calculate flat indices for input and weight tensors
                            // Input: (N, C_in, D_in, H_in, W_in)
                            const long long input_idx = (long long)n * C_in * D_in * H_in * W_in +
                                                        (long long)c_in * D_in * H_in * W_in +
                                                        (long long)d_in * H_in * W_in +
                                                        (long long)h_in * W_in +
                                                        w_in;
                            
                            // Weight: (C_in, C_out/groups, KD, KH, KW)
                            const long long weight_idx = (long long)c_in * C_out_per_group * KD * KH * KW +
                                                         (long long)c_out_in_group * KD * KH * KW +
                                                         (long long)kd * KH * KW +
                                                         (long long)kh * KW +
                                                         kw;

                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }

    // Add bias if it exists
    if (bias != nullptr) {
        sum += bias[c_out];
    }

    output[index] = sum;
}

torch::Tensor conv_transpose3d_forward_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride, int padding, int output_padding, int groups) {

    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias tensor must be a CUDA tensor");
    }
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5D");
    TORCH_CHECK(weight.dim() == 5, "Weight tensor must be 5D");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "Weight must be a float32 tensor");

    // Ensure contiguous memory layout
    input = input.contiguous();
    weight = weight.contiguous();
    if (bias.defined()) {
        bias = bias.contiguous();
    }

    // Get dimensions
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    // Weight shape is (C_in, C_out/groups, KD, KH, KW)
    TORCH_CHECK(C_in == weight.size(0), "Input channels must match weight's in_channels");
    const int C_out = weight.size(1) * groups;
    const int KD = weight.size(2);
    const int KH = weight.size(3);
    const int KW = weight.size(4);
    
    TORCH_CHECK(C_in % groups == 0, "in_channels must be divisible by groups");
    TORCH_CHECK(C_out % groups == 0, "out_channels must be divisible by groups");

    // Calculate output dimensions
    const int D_out = (D_in - 1) * stride - 2 * padding + KD + output_padding;
    const int H_out = (H_in - 1) * stride - 2 * padding + KH + output_padding;
    const int W_out = (W_in - 1) * stride - 2 * padding + KW + output_padding;

    // Create output tensor
    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    // Get data pointers
    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    // CUDA launch configuration
    const long long total_outputs = (long long)N * C_out * D_out * H_out * W_out;
    if (total_outputs == 0) {
        return output;
    }
    const int block_size = 256;
    const int num_blocks = (total_outputs + block_size - 1) / block_size;

    // Launch the kernel
    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        KD, KH, KW,
        stride, stride, stride, // Assuming square stride
        padding, padding, padding, // Assuming square padding
        groups
    );
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_forward_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride, int padding, int output_padding, int groups);
"""

# Compile the inline CUDA code
# This is done once when the module is imported.
conv_transpose3d_op = load_inline(
    name="conv_transpose3d_op",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_cuda_source,
    functions=["conv_transpose3d_forward_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with square input and square kernel using a custom CUDA kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        # Define weight and bias as learnable parameters
        # PyTorch's ConvTransposeNd weights are stored as (in_channels, out_channels / groups, k_dims...)
        self.weight = nn.Parameter(torch.Tensor(
            in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Mimic PyTorch's default initialization for Conv layers
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return conv_transpose3d_op.conv_transpose3d_forward_cuda(
            x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups
        )