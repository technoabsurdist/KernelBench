import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for MaxPool1d
maxpool1d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For fmaxf and INFINITY

__global__ void maxpool1d_forward_kernel(
    const float* input,
    float* output,
    int batch_size,
    int num_features,
    int input_seq_len,
    int output_seq_len,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int i = index; i < batch_size * num_features * output_seq_len; i += total_threads) {
        // Decompose the flat output index into 3D coordinates
        int out_l = i % output_seq_len;
        int c = (i / output_seq_len) % num_features;
        int b = i / (output_seq_len * num_features);

        // Calculate the starting position of the window in the input sequence
        int in_l_start = out_l * stride - padding;

        float max_val = -INFINITY;

        // Iterate over the kernel window
        for (int k = 0; k < kernel_size; ++k) {
            int in_l = in_l_start + k * dilation;

            // Check if the current position is within the bounds of the input sequence (and not in padding)
            if (in_l >= 0 && in_l < input_seq_len) {
                // Calculate the flat input index
                int input_index = b * num_features * input_seq_len + c * input_seq_len + in_l;
                max_val = fmaxf(max_val, input[input_index]);
            }
        }
        output[i] = max_val;
    }
}

torch::Tensor maxpool1d_forward_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "Input tensor must be 3D (batch, features, sequence_length)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");

    // Get input dimensions
    int batch_size = input.size(0);
    int num_features = input.size(1);
    int input_seq_len = input.size(2);

    // Calculate output sequence length
    int output_seq_len = floor((float)(input_seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1);
    TORCH_CHECK(output_seq_len > 0, "Output length must be positive. Check kernel_size, stride, padding, and dilation.");

    // Create the output tensor
    auto output = torch::empty({batch_size, num_features, output_seq_len}, input.options());

    // Get data pointers
    const float* input_data = input.data_ptr<float>();
    float* output_data = output.data_ptr<float>();

    // Configure and launch the kernel
    long long total_output_elements = (long long)batch_size * num_features * output_seq_len;
    
    // Use a grid-stride loop for flexibility with large inputs
    const int block_size = 256;
    const int num_blocks = std::min((int)((total_output_elements + block_size - 1) / block_size), 4096);


    maxpool1d_forward_kernel<<<num_blocks, block_size>>>(
        input_data,
        output_data,
        batch_size,
        num_features,
        input_seq_len,
        output_seq_len,
        kernel_size,
        stride,
        padding,
        dilation
    );
    
    // Check for any CUDA errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

maxpool1d_cpp_source = """
torch::Tensor maxpool1d_forward_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation);
"""

# Compile the inline CUDA code
maxpool1d_custom = load_inline(
    name="maxpool1d_custom",
    cpp_sources=maxpool1d_cpp_source,
    cuda_sources=maxpool1d_cuda_source,
    functions=["maxpool1d_forward_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 1D using a custom CUDA kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        """
        Initializes the custom Max Pooling 1D layer.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
            padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): This is ignored in the custom implementation for performance.
                                             Must be False.
        """
        super(ModelNew, self).__init__()
        if return_indices:
            raise NotImplementedError("Custom MaxPool1d kernel does not support return_indices=True.")
        
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies custom Max Pooling 1D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 1D applied, shape (batch_size, num_features, output_sequence_length).
        """
        return maxpool1d_custom.maxpool1d_forward_cuda(x, self.kernel_size, self.stride, self.padding, self.dilation)