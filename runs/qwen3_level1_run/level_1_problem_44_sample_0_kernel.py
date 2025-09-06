import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 1D Average Pooling
avg_pool1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void avg_pool1d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * output_length;
    
    if (idx >= total_elements) return;
    
    int out_w = idx % output_length;
    int channel = (idx / output_length) % channels;
    int batch = idx / (output_length * channels);
    
    int input_start = out_w * stride - padding;
    int input_end = input_start + kernel_size;
    
    float sum = 0.0f;
    int valid_count = 0;
    
    for (int i = input_start; i < input_end; i++) {
        if (i >= 0 && i < input_length) {
            int input_idx = batch * channels * input_length + channel * input_length + i;
            sum += input[input_idx];
            valid_count++;
        }
    }
    
    if (valid_count > 0) {
        output[idx] = sum / valid_count;
    } else {
        output[idx] = 0.0f;
    }
}

torch::Tensor avg_pool1d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_length = input.size(2);
    
    // Calculate output length
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, channels, output_length}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    int total_elements = batch_size * channels * output_length;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    avg_pool1d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding
    );
    
    return output;
}
"""

avg_pool1d_cpp_source = """
torch::Tensor avg_pool1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);
"""

# Compile the inline CUDA code for 1D Average Pooling
avg_pool1d = load_inline(
    name="avg_pool1d",
    cpp_sources=avg_pool1d_cpp_source,
    cuda_sources=avg_pool1d_source,
    functions=["avg_pool1d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs 1D Average Pooling with custom CUDA kernel.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the 1D Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avg_pool1d_cuda = avg_pool1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 1D Average Pooling to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied, shape (batch_size, in_channels, output_length).
        """
        return self.avg_pool1d_cuda.avg_pool1d_cuda(x, self.kernel_size, self.stride, self.padding)

batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4

def get_inputs():
    x = torch.rand(batch_size, in_channels, input_length).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]