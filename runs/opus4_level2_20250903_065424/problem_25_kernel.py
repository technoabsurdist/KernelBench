import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused min + double tanh
min_double_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void min_double_tanh_kernel(const float* input, float* output, 
                                       int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * height * width;
    
    if (idx < total_elements) {
        int b = idx / (height * width);
        int spatial_idx = idx % (height * width);
        
        // Find minimum across channels
        float min_val = FLT_MAX;
        for (int c = 0; c < channels; c++) {
            int input_idx = b * channels * height * width + c * height * width + spatial_idx;
            min_val = fminf(min_val, input[input_idx]);
        }
        
        // Apply double tanh
        min_val = tanhf(min_val);
        min_val = tanhf(min_val);
        
        // Store result
        output[idx] = min_val;
    }
}

torch::Tensor min_double_tanh_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto output = torch::zeros({batch_size, 1, height, width}, input.options());
    
    int total_elements = batch_size * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    min_double_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(),
        batch_size, channels, height, width
    );
    
    return output;
}
"""

min_double_tanh_cpp_source = (
    "torch::Tensor min_double_tanh_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code
min_double_tanh = load_inline(
    name="min_double_tanh",
    cpp_sources=min_double_tanh_cpp_source,
    cuda_sources=min_double_tanh_source,
    functions=["min_double_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with fused min + double tanh CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.min_double_tanh = min_double_tanh

    def forward(self, x):
        x = self.conv(x)
        x = self.min_double_tanh.min_double_tanh_cuda(x)
        return x

batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]