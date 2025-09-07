import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D Average Pooling
avg_pool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void avg_pool3d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * channels * output_depth * output_height * output_width;
    
    if (out_idx >= total_output_elements) return;
    
    // Calculate output indices
    int w_out = out_idx % output_width;
    int h_out = (out_idx / output_width) % output_height;
    int d_out = (out_idx / (output_width * output_height)) % output_depth;
    int c = (out_idx / (output_width * output_height * output_depth)) % channels;
    int n = out_idx / (output_width * output_height * output_depth * channels);
    
    // Calculate input region bounds
    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;
    
    int d_end = min(d_start + kernel_size, input_depth + padding);
    int h_end = min(h_start + kernel_size, input_height + padding);
    int w_end = min(w_start + kernel_size, input_width + padding);
    
    d_start = max(d_start, 0);
    h_start = max(h_start, 0);
    w_start = max(w_start, 0);
    
    // Compute average
    float sum = 0.0f;
    int count = 0;
    
    for (int d = d_start; d < d_end; ++d) {
        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                int input_idx = ((n * channels + c) * input_depth + d) * input_height * input_width + h * input_width + w;
                sum += input[input_idx];
                count++;
            }
        }
    }
    
    output[out_idx] = (count > 0) ? sum / count : 0.0f;
}

torch::Tensor avg_pool3d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding
) {
    // Ensure we're on the right device
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int channels = input_sizes[1];
    int input_depth = input_sizes[2];
    int input_height = input_sizes[3];
    int input_width = input_sizes[4];
    
    // Calculate output dimensions
    int output_depth = (input_depth + 2 * padding - kernel_size) / stride + 1;
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, channels, output_depth, output_height, output_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    int total_output_elements = batch_size * channels * output_depth * output_height * output_width;
    
    const int block_size = 256;
    const int num_blocks = (total_output_elements + block_size - 1) / block_size;
    
    avg_pool3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding
    );
    
    return output;
}
"""

avg_pool3d_cpp_source = """
torch::Tensor avg_pool3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);
"""

# Compile the inline CUDA code for 3D Average Pooling
avg_pool3d = load_inline(
    name="avg_pool3d",
    cpp_sources=avg_pool3d_cpp_source,
    cuda_sources=avg_pool3d_source,
    functions=["avg_pool3d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs 3D Average Pooling with custom CUDA kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which uses the kernel size.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.avg_pool3d_cuda = avg_pool3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return self.avg_pool3d_cuda.avg_pool3d_cuda(x, self.kernel_size, self.stride, self.padding)