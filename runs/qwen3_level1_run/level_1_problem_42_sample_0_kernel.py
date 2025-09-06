import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Max Pooling 2D
maxpool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void maxpool2d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * output_height * output_width;
    
    if (idx >= total_elements) return;
    
    int w_out = idx % output_width;
    int h_out = (idx / output_width) % output_height;
    int c = (idx / (output_width * output_height)) % channels;
    int n = idx / (output_width * output_height * channels);
    
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;
    int h_end = h_start + (kernel_size - 1) * dilation + 1;
    int w_end = w_start + (kernel_size - 1) * dilation + 1;
    
    float max_val = -FLT_MAX;
    
    for (int h = h_start; h < h_end; h += dilation) {
        for (int w = w_start; w < w_end; w += dilation) {
            if (h >= 0 && h < input_height && w >= 0 && w < input_width) {
                int input_idx = ((n * channels + c) * input_height + h) * input_width + w;
                float val = input[input_idx];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }
    
    output[idx] = max_val;
}

torch::Tensor maxpool2d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const auto input_sizes = input.sizes();
    const int batch_size = input_sizes[0];
    const int channels = input_sizes[1];
    const int input_height = input_sizes[2];
    const int input_width = input_sizes[3];
    
    const int output_height = (input_height + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1;
    const int output_width = (input_width + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1;
    
    auto output = torch::zeros({batch_size, channels, output_height, output_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    const int total_elements = batch_size * channels * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    at::cuda::CUDAGuard device_guard(input.device());
    
    maxpool2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        dilation
    );
    
    return output;
}
"""

maxpool2d_cpp_source = """
torch::Tensor maxpool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);
"""

# Compile the inline CUDA code for Max Pooling 2D
maxpool2d = load_inline(
    name="maxpool2d",
    cpp_sources=maxpool2d_cpp_source,
    cuda_sources=maxpool2d_source,
    functions=["maxpool2d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 2D with custom CUDA kernel.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer with custom CUDA implementation.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.maxpool2d_cuda = maxpool2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 2D to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D, shape (batch_size, channels, pooled_height, pooled_width).
        """
        return self.maxpool2d_cuda.maxpool2d_cuda(x, self.kernel_size, self.stride, self.padding, self.dilation)

batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

def get_inputs():
    x = torch.rand(batch_size, channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]