import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 1D Average Pooling
avg_pool1d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel for 1D Average Pooling forward pass
__global__ void avg_pool1d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N,
    const int C,
    const int L_in,
    const int L_out,
    const int kernel_size,
    const int stride,
    const int padding) {

    // Calculate the global thread index for the flattened output tensor
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_elements = N * C * L_out;

    if (index < num_elements) {
        // De-flatten the index to get (n, c, l_out) coordinates
        const int l_out = index % L_out;
        const int c = (index / L_out) % C;
        const int n = index / (L_out * C);

        // Calculate the start of the pooling window in the input tensor
        const int start_in = l_out * stride - padding;
        
        float sum = 0.0f;
        
        // Iterate over the pooling window
        for (int k = 0; k < kernel_size; ++k) {
            const int current_in_idx = start_in + k;
            
            // Check if the current index is within the valid input range (not in padding)
            if (current_in_idx >= 0 && current_in_idx < L_in) {
                // Calculate the flattened index for the input tensor
                const int input_flat_idx = n * C * L_in + c * L_in + current_in_idx;
                sum += input[input_flat_idx];
            }
        }
        
        // PyTorch's default AvgPool1d (with count_include_pad=True) always divides by kernel_size.
        output[index] = sum / static_cast<float>(kernel_size);
    }
}

// C++ wrapper function to be called from Python
torch::Tensor avg_pool1d_forward(
    torch::Tensor input,
    const int kernel_size,
    const int stride,
    const int padding) {
    
    // Input validation
    TORCH_CHECK(input.dim() == 3, "Input must be a 3D tensor");
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    // Get input dimensions
    const int N = input.size(0);
    const int C = input.size(1);
    const int L_in = input.size(2);

    // Calculate output dimension
    const int L_out = static_cast<int>(floor(static_cast<float>(L_in + 2 * padding - kernel_size) / stride)) + 1;

    // Create the output tensor
    auto output = torch::zeros({N, C, L_out}, input.options());

    // If output has no elements, return it immediately
    if (output.numel() == 0) {
        return output;
    }

    // Set up grid and block dimensions for the kernel launch
    const int num_threads = 256;
    const int num_blocks = (output.numel() + num_threads - 1) / num_threads;

    // Launch the CUDA kernel
    avg_pool1d_forward_kernel<<<num_blocks, num_threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, L_in, L_out,
        kernel_size, stride, padding
    );
    
    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

# C++ source for the function signature, enabling PyTorch to bind it
avg_pool1d_cpp_source = """
torch::Tensor avg_pool1d_forward(
    torch::Tensor input,
    const int kernel_size,
    const int stride,
    const int padding);
"""

# Compile the inline CUDA code using torch.utils.cpp_extension.load_inline
custom_avg_pool1d = load_inline(
    name="custom_avg_pool1d",
    cpp_sources=avg_pool1d_cpp_source,
    cuda_sources=avg_pool1d_cuda_source,
    functions=["avg_pool1d_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Simple model that performs 1D Average Pooling using a custom CUDA kernel.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the custom 1D Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Store the compiled custom operator
        self.custom_avg_pool = custom_avg_pool1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies custom 1D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied, shape (batch_size, in_channels, output_length).
        """
        # Ensure input is on the correct device (CUDA) and is contiguous
        x = x.cuda().contiguous()
        return self.custom_avg_pool.avg_pool1d_forward(
            x, self.kernel_size, self.stride, self.padding
        )