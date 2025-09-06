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
        long max_idx = 0;
        float max_val = row[0];
        
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
    
    if (input.dim() == 1) {
        // Special case for 1D tensor
        const int size = input.size(0);
        const int block_size = 256;
        const int num_blocks = (size + block_size - 1) / block_size;
        
        // For 1D, we need a different approach - use thrust or cub
        // For simplicity, we'll use a single block for small tensors
        auto input_accessor = input.accessor<float, 1>();
        auto output_accessor = output.accessor<long, 0>();
        
        // Launch a single thread to find argmax
        auto stream = at::cuda::getCurrentCUDAStream();
        thrust::device_ptr<float> input_ptr(input.data_ptr<float>());
        thrust::device_ptr<float> max_ptr = thrust::max_element(input_ptr, input_ptr + size);
        int max_index = thrust::distance(input_ptr, max_ptr);
        output.fill_(max_index);
    } else {
        // Handle multi-dimensional case
        int outer_size = 1;
        int reduce_size = input.size(dim);
        int inner_size = 1;
        
        for (int i = 0; i < dim; i++) {
            outer_size *= input.size(i);
        }
        for (int i = dim + 1; i < input.dim(); i++) {
            inner_size *= input.size(i);
        }
        
        const int total_threads = outer_size * inner_size;
        const int block_size = 256;
        const int num_blocks = (total_threads + block_size - 1) / block_size;
        
        argmax_kernel<<<num_blocks, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
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
#include <torch/extension.h>
torch::Tensor argmax_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code for argmax
argmax_module = load_inline(
    name="argmax_cuda",
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmax over the specified dimension to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        return argmax_module.argmax_cuda(x, self.dim)

batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [1]