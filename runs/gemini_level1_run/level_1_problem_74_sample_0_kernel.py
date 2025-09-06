import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for 1D transposed convolution
conv_transpose1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose1d_kernel(
    const float* input,      // (N, C_in, L_in)
    const float* weight,     // (C_in, C_out, K)
    float* output,           // (N, C_out, L_out)
    const float* bias,       // (C_out) or nullptr
    int N, int C_in, int L_in,
    int C_out, int L_out, int K,
    int stride, int padding, int dilation
) {
    // This kernel uses a "gathering" approach, where each output element
    // computes its value by summing up contributions from the relevant
    // input and weight elements. This avoids atomic operations.
    //
    // Grid: dim3( (L_out + BS-1)/BS, N * C_out )
    // Block: dim3( BS )
    // Each thread computes one element in the output tensor.
    int l_out = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cout_idx = blockIdx.y;

    // Boundary check for the length dimension
    if (l_out >= L_out) {
        return;
    }

    // Decompose the y-dimension block index into batch and output channel indices
    int n = n_cout_idx / C_out;
    int c_out = n_cout_idx % C_out;

    float sum = 0.0f;
    
    // The relationship between ConvTranspose1d and Conv1d allows us to calculate
    // the required padding for an equivalent standard convolution on a strided input.
    const int effective_padding = dilation * (K - 1) - padding;

    // Loop over all input channels
    for (int c_in = 0; c_in < C_in; ++c_in) {
        // Loop over the kernel dimension
        for (int k = 0; k < K; ++k) {
            // Calculate the corresponding position in an "expanded" input tensor
            // (one with zeros inserted for the stride).
            int l_in_expanded = l_out - k * dilation + effective_padding;

            // Check if this expanded position corresponds to an original input element
            if (l_in_expanded >= 0 && l_in_expanded % stride == 0) {
                int l_in = l_in_expanded / stride;
                
                // Check if the calculated input index is within bounds
                if (l_in < L_in) {
                    // Calculate flat indices for the input and weight tensors
                    int input_idx = n * C_in * L_in + c_in * L_in + l_in;
                    // PyTorch weight for ConvTranspose1d is (C_in, C_out, K)
                    int weight_idx = c_in * C_out * K + c_out * K + k;
                    
                    // Accumulate the product
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias if it exists
    if (bias != nullptr) {
        sum += bias[c_out];
    }

    // Calculate the flat index for the output tensor and write the result
    int output_idx = n * C_out * L_out + c_out * L_out + l_out;
    output[output_idx] = sum;
}

// C++ wrapper function that will be called from Python
torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias, // Can be an undefined tensor if bias=False
    int stride,
    int padding,
    int dilation
) {
    // Ensure tensors are on the correct device (CUDA) and are contiguous in memory
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
        bias = bias.contiguous();
    }
    input = input.contiguous();
    weight = weight.contiguous();

    // Get tensor dimensions
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int L_in = input.size(2);

    // PyTorch's nn.ConvTranspose1d weight is shaped (in_channels, out_channels, kernel_size)
    TORCH_CHECK(weight.size(0) == C_in, "Weight in_channels mismatch");
    const int C_out = weight.size(1);
    const int K = weight.size(2);

    // Calculate the output length based on the formula for transposed convolution
    const int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length must be positive, but got ", L_out);

    // Create the output tensor, initialized to zeros
    auto output = torch::zeros({N, C_out, L_out}, input.options());

    // Setup CUDA launch configuration
    const int block_size = 256;
    const dim3 threads(block_size);
    const dim3 blocks((L_out + block_size - 1) / block_size, N * C_out);

    // Get raw data pointers
    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    // Launch the CUDA kernel
    conv_transpose1d_kernel<<<blocks, threads>>>(
        input_ptr, weight_ptr, output_ptr, bias_ptr,
        N, C_in, L_in, C_out, L_out, K,
        stride, padding, dilation
    );

    // Check for any errors during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

# C++ source for the function signature
conv_transpose1d_cpp_source = """
torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation
);
"""

# Compile the inline CUDA code
custom_conv_transpose1d = load_inline(
    name="custom_conv_transpose1d",
    cpp_sources=conv_transpose1d_cpp_source,
    cuda_sources=conv_transpose1d_source,
    functions=["conv_transpose1d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution using a custom CUDA kernel.

    This module holds the parameters (weight and bias) in a standard
    nn.ConvTranspose1d layer, but its forward pass calls the custom,
    high-performance CUDA implementation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store convolution parameters to pass to the CUDA kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # We use a standard PyTorch layer to hold the learnable parameters.
        # This is a clean way to ensure they are registered with the module,
        # appear in .parameters(), are moved to the correct device with .to(device),
        # and are handled correctly by optimizers and serialization.
        self.params_container = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # The forward pass bypasses the torch implementation and calls our custom kernel directly,
        # passing the input tensor and the learnable parameters from our container.
        return custom_conv_transpose1d.conv_transpose1d_cuda(
            x,
            self.params_container.weight,
            self.params_container.bias, # Pass the bias tensor; it will be None if bias=False
            self.stride,
            self.padding,
            self.dilation
        )