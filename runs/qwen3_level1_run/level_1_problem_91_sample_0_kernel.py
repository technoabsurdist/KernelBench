import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for reverse cumulative sum
reverse_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void reverse_cumsum_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        int base_idx = row * cols;
        // Perform reverse cumulative sum from right to left
        float cumsum = 0.0f;
        for (int i = cols - 1; i >= 0; i--) {
            cumsum += input[base_idx + i];
            output[base_idx + i] = cumsum;
        }
    }
}

torch::Tensor reverse_cumsum_cuda(torch::Tensor input) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    auto output = torch::zeros_like(input);
    
    int rows = input.size(0);
    int cols = input.size(1);
    
    const int block_size = 256;
    const int num_blocks = (rows + block_size - 1) / block_size;
    
    reverse_cumsum_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        rows, 
        cols
    );
    
    return output;
}
"""

reverse_cumsum_cpp_source = (
    "torch::Tensor reverse_cumsum_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for reverse cumulative sum
reverse_cumsum = load_inline(
    name="reverse_cumsum",
    cpp_sources=reverse_cumsum_cpp_source,
    cuda_sources=reverse_cumsum_source,
    functions=["reverse_cumsum_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    A model that performs a reverse cumulative sum operation along dimension 1,
    optimized with a custom CUDA kernel.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.reverse_cumsum_func = reverse_cumsum

    def forward(self, x):
        # Ensure input is 2D as expected by our kernel
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            # Reshape to 2D, apply operation, then reshape back
            original_shape = x.shape
            x = x.view(-1, original_shape[-1])
            result = self.reverse_cumsum_func.reverse_cumsum_cuda(x)
            return result.view(original_shape)
        
        return self.reverse_cumsum_func.reverse_cumsum_cuda(x)