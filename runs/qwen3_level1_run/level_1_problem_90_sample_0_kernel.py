import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for cumulative product
cumprod_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

template<typename T>
__global__ void cumprod_kernel(const T* input, T* output, int rows, int cols, int dim) {
    if (dim == 1) {
        int row = blockIdx.x;
        int tid = threadIdx.x;
        
        if (row < rows) {
            extern __shared__ char shared_mem[];
            T* shared_data = reinterpret_cast<T*>(shared_mem);
            
            // Load data into shared memory
            for (int i = tid; i < cols; i += blockDim.x) {
                shared_data[i] = input[row * cols + i];
            }
            __syncthreads();
            
            // Perform inclusive scan with multiplication
            for (int i = 1; i < cols; i <<= 1) {
                T temp = 1;
                if (tid >= i) {
                    temp = shared_data[tid - i] * shared_data[tid];
                }
                __syncthreads();
                if (tid >= i) {
                    shared_data[tid] = temp;
                }
                __syncthreads();
            }
            
            // Write back to global memory
            for (int i = tid; i < cols; i += blockDim.x) {
                output[row * cols + i] = shared_data[i];
            }
        }
    }
}

torch::Tensor cumprod_cuda(torch::Tensor input, int64_t dim) {
    auto output = torch::zeros_like(input);
    auto sizes = input.sizes();
    
    if (dim < 0) dim += input.dim();
    
    if (input.dim() == 2 && dim == 1) {
        int rows = sizes[0];
        int cols = sizes[1];
        
        dim3 block_size(256);
        dim3 grid_size(rows);
        size_t shared_mem_size = cols * sizeof(float);
        
        cumprod_kernel<float><<<grid_size, block_size, shared_mem_size>>>(
            input.data_ptr<float>(), 
            output.data_ptr<float>(), 
            rows, 
            cols, 
            dim
        );
    } else {
        // Fallback to PyTorch's implementation for other cases
        return torch::cumprod(input, dim);
    }
    
    return output;
}
"""

cumprod_cpp_source = """
torch::Tensor cumprod_cuda(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code for cumulative product
cumprod = load_inline(
    name="cumprod",
    cpp_sources=cumprod_cpp_source,
    cuda_sources=cumprod_source,
    functions=["cumprod_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    A model that performs a cumulative product operation along a specified dimension
    using a custom CUDA kernel for improved performance.
    """

    def __init__(self, dim):
        """
        Initialize the optimized CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cumprod_func = cumprod

    def forward(self, x):
        """
        Forward pass, computing the cumulative product along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        return self.cumprod_func.cumprod_cuda(x, self.dim)