import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused min + softmax
min_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void min_softmax_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * height * width;
    
    if (idx >= total_output_elements) return;
    
    int b = idx / (height * width);
    int hw = idx % (height * width);
    int h = hw / width;
    int w = hw % width;
    
    extern __shared__ float shared_mem[];
    float* min_vals = shared_mem;
    float* exp_vals = min_vals + channels;
    
    // Step 1: Compute minimum along depth dimension for each channel
    for (int c = 0; c < channels; c++) {
        float min_val = FLT_MAX;
        for (int d = 0; d < depth; d++) {
            int input_idx = b * (channels * depth * height * width) +
                           c * (depth * height * width) +
                           d * (height * width) +
                           h * width + w;
            min_val = fminf(min_val, input[input_idx]);
        }
        min_vals[c] = min_val;
    }
    
    // Step 2: Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int c = 0; c < channels; c++) {
        max_val = fmaxf(max_val, min_vals[c]);
    }
    
    // Step 3: Compute exp and sum
    float sum_exp = 0.0f;
    for (int c = 0; c < channels; c++) {
        exp_vals[c] = expf(min_vals[c] - max_val);
        sum_exp += exp_vals[c];
    }
    
    // Step 4: Normalize and write output
    for (int c = 0; c < channels; c++) {
        int output_idx = b * (channels * height * width) +
                        c * (height * width) +
                        h * width + w;
        output[output_idx] = exp_vals[c] / sum_exp;
    }
}

torch::Tensor min_softmax_cuda(torch::Tensor input, int dim) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    
    auto output = torch::zeros({batch_size, channels, height, width}, 
                              torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    const int block_size = 256;
    const int num_elements = batch_size * height * width;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    const int shared_mem_size = 2 * channels * sizeof(float);
    
    min_softmax_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        depth,
        height,
        width
    );
    
    return output;
}
"""

min_softmax_cpp_source = "torch::Tensor min_softmax_cuda(torch::Tensor input, int dim);"

# Compile the inline CUDA code
min_softmax = load_inline(
    name="min_softmax",
    cpp_sources=min_softmax_cpp_source,
    cuda_sources=min_softmax_source,
    functions=["min_softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for fused min + softmax operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim
        self.min_softmax = min_softmax

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W)
        """
        x = self.conv(x)
        x = self.min_softmax.min_softmax_cuda(x, self.dim)
        return x

batch_size = 128
in_channels = 3
out_channels = 24  # Increased output channels
D, H, W = 24, 32, 32  # Increased depth
kernel_size = 3
dim = 2  # Dimension along which to apply minimum operation (e.g., depth)

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]