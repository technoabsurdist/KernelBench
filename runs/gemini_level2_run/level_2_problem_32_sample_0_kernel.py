import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused scaling and channel-wise minimum
fused_scale_min_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat> // For FLT_MAX

// CUDA kernel to perform scaling and then find the minimum value across the channel dimension.
// The grid is 1D and is mapped to the (batch, height, width) dimensions.
// Each thread computes the minimum for one output pixel.
__global__ void scale_and_min_channel_kernel(
    const float* input,
    float* output,
    const float scale_factor,
    const int B,
    const int C,
    const int H,
    const int W) {

    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of output elements (pixels)
    const int N = B * H * W;

    if (idx < N) {
        // De-linearize the index to get batch, height, and width coordinates
        const int w = idx % W;
        const int h = (idx / W) % H;
        const int b = idx / (W * H);

        // Calculate the base index for the current output pixel in the input tensor (at channel 0)
        const int base_input_idx = b * C * H * W + h * W + w;

        // Initialize min_val with the scaled value from the first channel
        float min_val = input[base_input_idx] * scale_factor;

        // Iterate over the remaining channels to find the minimum
        for (int c = 1; c < C; ++c) {
            // Calculate the index for the current channel
            const int current_input_idx = base_input_idx + c * H * W;
            // Scale the value and update the minimum
            min_val = fminf(min_val, input[current_input_idx] * scale_factor);
        }

        // Write the final minimum value to the output tensor
        // The output index is the same as the thread index 'idx'
        output[idx] = min_val;
    }
}

// C++ wrapper function that launches the CUDA kernel
torch::Tensor scale_and_min_channel_cuda(torch::Tensor input, float scale_factor) {
    // Ensure input is a contiguous 4D CUDA tensor
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    // Get input dimensions
    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    // Create the output tensor with shape (B, 1, H, W)
    auto output = torch::empty({B, 1, H, W}, input.options());

    // Total number of output elements to compute
    const int N = B * H * W;
    if (N == 0) {
        return output;
    }

    // CUDA launch configuration
    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;

    // Launch the kernel
    scale_and_min_channel_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scale_factor,
        B, C, H, W);

    return output;
}
"""

fused_scale_min_cpp_source = """
torch::Tensor scale_and_min_channel_cuda(torch::Tensor input, float scale_factor);
"""

# Compile the inline CUDA code
fused_scale_min = load_inline(
    name="fused_scale_min",
    cpp_sources=fused_scale_min_cpp_source,
    cuda_sources=fused_scale_min_source,
    functions=["scale_and_min_channel_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, then uses a fused CUDA kernel for scaling and minimum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        # The convolution layer remains a standard PyTorch operator
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        # Store the custom fused operator
        self.fused_op = fused_scale_min

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, height, width).
        """
        # 1. Perform convolution using the standard PyTorch layer
        x = self.conv(x)
        # 2. Apply the fused scale and channel-wise minimum operation using our custom CUDA kernel
        x = self.fused_op.scale_and_min_channel_cuda(x, self.scale_factor)
        return x