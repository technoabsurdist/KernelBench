import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for argmax
argmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void argmax_kernel(const float* input, long* output, int outer_size, int reduce_size, int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = outer_size * inner_size;
    
    if (idx < total_threads) {
        int outer_idx = idx / inner_size;
        int inner_idx = idx % inner_size;
        
        const float* row = input + outer_idx * reduce_size * inner_size + inner_idx;
        float max_val = row[0];
        long max_idx = 0;
        
        for (int i = 1; i < reduce_size; i++) {
            float val = row[i * inner_size];
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        
        output[outer_idx * inner_size + inner_idx] = max_idx;
    }
}

torch::Tensor argmax_cuda(torch::Tensor input, int dim) {
    auto input_sizes = input.sizes();
    auto output_sizes = input_sizes.vec();
    output_sizes.erase(output_sizes.begin() + dim);
    
    auto output = torch::zeros(output_sizes, torch::kLong);
    
    // Set the device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    if (dim == input.dim() - 1) {
        // Last dimension case
        int outer_size = 1;
        for (int i = 0; i < dim; i++) {
            outer_size *= input_sizes[i];
        }
        int reduce_size = input_sizes[dim];
        int inner_size = 1;
        
        const int block_size = 256;
        const int num_blocks = (outer_size + block_size - 1) / block_size;
        
        argmax_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(), 
            output.data_ptr<long>(), 
            outer_size, 
            reduce_size, 
            inner_size
        );
    } else if (dim == 0) {
        // First dimension case
        int outer_size = 1;
        int reduce_size = input_sizes[0];
        int inner_size = 1;
        for (int i = 1; i < input.dim(); i++) {
            inner_size *= input_sizes[i];
        }
        
        const int block_size = 256;
        const int num_blocks = (inner_size + block_size - 1) / block_size;
        
        argmax_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(), 
            output.data_ptr<long>(), 
            1, 
            reduce_size, 
            inner_size
        );
    } else {
        // Middle dimension case
        int outer_size = 1;
        for (int i = 0; i < dim; i++) {
            outer_size *= input_sizes[i];
        }
        int reduce_size = input_sizes[dim];
        int inner_size = 1;
        for (int i = dim + 1; i < input.dim(); i++) {
            inner_size *= input_sizes[i];
        }
        
        const int block_size = 256;
        const int num_blocks = (outer_size * inner_size + block_size - 1) / block_size;
        
        argmax_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(), 
            output.data_ptr<long>(), 
            outer_size, 
            reduce_size, 
            inner_size
        );
    }
    
    return output;
}
"""

argmax_cpp_source = """
torch::Tensor argmax_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code for argmax
argmax = load_inline(
    name="argmax",
    cpp_sources=argmax_cpp_source,
    cuda_sources=argmax_source,
    functions=["argmax_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Argmax over a specified dimension using custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmax_func = argmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmax over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        return self.argmax_func.argmax_cuda(x, self.dim)