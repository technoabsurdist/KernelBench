import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for average pooling
avg_pool_2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void avg_pool2d_kernel(
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
    int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * output_height * output_width;
    
    if (idx >= total_elements) return;
    
    int w_out = idx % output_width;
    idx /= output_width;
    int h_out = idx % output_height;
    idx /= output_height;
    int c = idx % channels;
    int n = idx / channels;
    
    int h_start = h_out * stride - padding;
    int h_end = min(h_start + kernel_size, input_height);
    h_start = max(h_start, 0);
    
    int w_start = w_out * stride - padding;
    int w_end = min(w_start + kernel_size, input_width);
    w_start = max(w_start, 0);
    
    float sum = 0.0f;
    int count = 0;
    
    for (int h = h_start; h < h_end; ++h) {
        for (int w = w_start; w < w_end; ++w) {
            int input_idx = ((n * channels + c) * input_height + h) * input_width + w;
            sum += input[input_idx];
            count++;
        }
    }
    
    output[idx * output_height * output_width + h_out * output_width + w_out] = (count > 0) ? sum / count : 0.0f;
}

torch::Tensor avg_pool2d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, channels, output_height, output_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    int total_elements = batch_size * channels * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    avg_pool2d_kernel<<<num_blocks, block_size>>>(
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
        padding
    );
    
    return output;
}
"""

avg_pool_2d_cpp_source = """
torch::Tensor avg_pool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);
"""

# Compile the inline CUDA code for average pooling
avg_pool_2d = load_inline(
    name="avg_pool_2d",
    cpp_sources=avg_pool_2d_cpp_source,
    cuda_sources=avg_pool_2d_source,
    functions=["avg_pool2d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the custom Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.avg_pool_2d = avg_pool_2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies custom 2D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return self.avg_pool_2d.avg_pool2d_cuda(x, self.kernel_size, self.stride, self.padding)

batch_size = 16
channels = 64
height = 2048
width = 2048
kernel_size = 11

def get_inputs():
    x = torch.rand(batch_size, channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size]