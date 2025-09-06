import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_forward_kernel(
    const float* input,
    const float* weight,
    const float* bias, // Can be nullptr
    float* output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int K, int S, int P, int D) {

    // Using a grid-stride loop to handle arbitrary number of elements
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N * C_out * D_out * H_out * W_out;
         idx += blockDim.x * gridDim.x) {

        // Decode 1D index to 5D tensor coordinates for the output tensor
        int w_out = idx % W_out;
        int h_out = (idx / W_out) % H_out;
        int d_out = (idx / (W_out * H_out)) % D_out;
        int c_out = (idx / (W_out * H_out * D_out)) % C_out;
        int n = idx / (W_out * H_out * D_out * C_out);

        float sum = 0.0f;

        // Iterate over input channels
        for (int c_in = 0; c_in < C_in; ++c_in) {
            // Iterate over the 3D kernel
            for (int kd = 0; kd < K; ++kd) {
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        // This is the core logic of transposed convolution (gather operation)
                        // We find which input pixel maps to the current output pixel
                        // The relation is: out_coord = in_coord * stride - padding + dilation * kernel_coord
                        // We need the inverse: in_coord = (out_coord + padding - dilation * kernel_coord) / stride
                        
                        int d_in_nom = d_out + P - kd * D;
                        int h_in_nom = h_out + P - kh * D;
                        int w_in_nom = w_out + P - kw * D;

                        // An input pixel (d_in, h_in, w_in) contributes to an output pixel (d_out, h_out, w_out)
                        // only if the numerator is perfectly divisible by the stride.
                        if (d_in_nom >= 0 && h_in_nom >= 0 && w_in_nom >= 0 &&
                            d_in_nom % S == 0 && h_in_nom % S == 0 && w_in_nom % S == 0) {
                            
                            int d_in = d_in_nom / S;
                            int h_in = h_in_nom / S;
                            int w_in = w_in_nom / S;

                            // Check if the calculated input coordinates are within the bounds of the input tensor
                            if (d_in < D_in && h_in < H_in && w_in < W_in) {
                                // Calculate flat indices for input and weight tensors
                                // Input tensor shape: (N, C_in, D_in, H_in, W_in)
                                int input_idx = n * C_in * D_in * H_in * W_in +
                                                c_in * D_in * H_in * W_in +
                                                d_in * H_in * W_in +
                                                h_in * W_in +
                                                w_in;
                                
                                // Weight tensor shape: (C_in, C_out, K, K, K)
                                int weight_idx = c_in * C_out * K * K * K +
                                                 c_out * K * K * K +
                                                 kd * K * K +
                                                 kh * K +
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
        output[idx] = sum;
    }
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride, int padding, int output_padding, int dilation) {

    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.dim() == 5, "Input must be a 5D tensor");
    TORCH_CHECK(weight.dim() == 5, "Weight must be a 5D tensor");

    input = input.contiguous();
    weight = weight.contiguous();
    
    // Get input dimensions
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    // Get weight dimensions
    TORCH_CHECK(weight.size(0) == C_in, "Weight in_channels mismatch");
    const int C_out = weight.size(1);
    const int K = weight.size(2);
    TORCH_CHECK(weight.size(3) == K && weight.size(4) == K, "Kernel must be square");

    // Calculate output dimensions using the formula from PyTorch docs
    const int D_out = (D_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1;
    const int H_out = (H_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1;
    const int W_out = (W_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1;

    // Create the output tensor, initialized to zeros
    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    const long long total_outputs = N * C_out * D_out * H_out * W_out;
    if (total_outputs == 0) {
        return output;
    }

    // Configure and launch the kernel
    const int block_size = 256;
    const int num_blocks = (total_outputs + block_size - 1) / block_size;

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().contiguous().data_ptr<float>() : nullptr;

    conv_transpose3d_forward_kernel<<<num_blocks, block_size>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        K, stride, padding, dilation
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride, int padding, int output_padding, int dilation);
"""

# Compile the inline CUDA code
conv_transpose3d_op = load_inline(
    name="conv_transpose3d_op",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    A replacement for nn.ConvTranspose3d using a custom CUDA kernel.
    This implementation supports square kernels, uniform stride/padding/dilation, and groups=1.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()

        if groups != 1:
            raise NotImplementedError("Custom CUDA kernel only supports groups=1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        
        # Define learnable parameters, matching the shape of nn.ConvTranspose3d
        self.weight = nn.Parameter(torch.empty(
            in_channels, out_channels, kernel_size, kernel_size, kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize weights and bias to match PyTorch's default for nn.Conv
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using the custom CUDA kernel.
        """
        return conv_transpose3d_op.conv_transpose3d_cuda(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding, self.dilation
        )