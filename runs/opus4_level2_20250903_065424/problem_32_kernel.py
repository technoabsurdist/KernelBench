import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused scale and min operations
scale_min_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void scale_min_kernel(
    const float* input,
    float* output,
    const float scale_factor,
    const int batch_size,
    const int channels,
    const int height,
    const int width
) {
    const int hw = height * width;
    const int total_output_elements = batch_size * hw;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_output_elements) {
        int b = idx / hw;
        int hw_idx = idx % hw;
        
        float min_val = FLT_MAX;
        
        for (int c = 0; c < channels; c++) {
            int input_idx = b * channels * hw + c * hw + hw_idx;
            float scaled_val = input[input_idx] * scale_factor;
            min_val = fminf(min_val, scaled_val);
        }
        
        output[idx] = min_val;
    }
}

torch::Tensor scale_min_cuda(
    torch::Tensor input,
    float scale_factor
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    auto output = torch::zeros({batch_size, 1, height, width}, input.options());
    
    const int total_output_elements = batch_size * height * width;
    const int threads_per_block = 256;
    const int num_blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;
    
    scale_min_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scale_factor,
        batch_size,
        channels,
        height,
        width
    );
    
    return output;
}
"""

scale_min_cpp_source = """
torch::Tensor scale_min_cuda(torch::Tensor input, float scale_factor);
"""

# Compile the inline CUDA code
scale_min = load_inline(
    name="scale_min",
    cpp_sources=scale_min_cpp_source,
    cuda_sources=scale_min_source,
    functions=["scale_min_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution followed by fused scale and minimum operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.scale_min = scale_min

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, height, width).
        """
        x = self.conv(x)
        x = self.scale_min.scale_min_cuda(x, self.scale_factor)
        return x


batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]