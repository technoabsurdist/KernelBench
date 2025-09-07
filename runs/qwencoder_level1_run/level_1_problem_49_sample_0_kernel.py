import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for max reduction
max_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void max_reduction_kernel(const float* input, float* output, 
                                     const int outer_size, const int reduction_size, const int inner_size) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    const float* input_slice = input + outer_idx * reduction_size * inner_size + inner_idx;
    float max_val = -FLT_MAX;
    
    for (int i = 0; i < reduction_size; ++i) {
        float val = input_slice[i * inner_size];
        max_val = fmaxf(max_val, val);
    }
    
    output[outer_idx * inner_size + inner_idx] = max_val;
}

__global__ void max_reduction_kernel_last_dim(const float* input, float* output, 
                                              const int outer_size, const int reduction_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= outer_size) return;
    
    const float* input_slice = input + idx * reduction_size;
    float max_val = -FLT_MAX;
    
    for (int i = 0; i < reduction_size; ++i) {
        max_val = fmaxf(max_val, input_slice[i]);
    }
    
    output[idx] = max_val;
}

torch::Tensor max_reduction_cuda(torch::Tensor input, int dim) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto input_sizes = input.sizes();
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    
    auto output = torch::zeros(output_sizes, input.options());
    
    if (dim == input.dim() - 1) {
        // Last dimension reduction
        int outer_size = 1;
        for (int i = 0; i < dim; ++i) {
            outer_size *= input_sizes[i];
        }
        int reduction_size = input_sizes[dim];
        
        const int block_size = 256;
        const int num_blocks = (outer_size + block_size - 1) / block_size;
        
        max_reduction_kernel_last_dim<<<num_blocks, block_size>>>(
            input.data_ptr<float>(), 
            output.data_ptr<float>(), 
            outer_size, 
            reduction_size
        );
    } else {
        // General dimension reduction
        int outer_size = 1;
        for (int i = 0; i < dim; ++i) {
            outer_size *= input_sizes[i];
        }
        int reduction_size = input_sizes[dim];
        int inner_size = 1;
        for (int i = dim + 1; i < input.dim(); ++i) {
            inner_size *= input_sizes[i];
        }
        
        dim3 block_size(inner_size > 1024 ? 1024 : inner_size);
        dim3 num_blocks(outer_size);
        
        max_reduction_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(), 
            output.data_ptr<float>(), 
            outer_size, 
            reduction_size, 
            inner_size
        );
    }
    
    return output;
}
"""

max_reduction_cpp_source = """
torch::Tensor max_reduction_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code for max reduction
max_reduction = load_inline(
    name="max_reduction",
    cpp_sources=max_reduction_cpp_source,
    cuda_sources=max_reduction_source,
    functions=["max_reduction_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Max reduction over a specific dimension using custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.max_reduction = max_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max reduction over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        return self.max_reduction.max_reduction_cuda(x, self.dim)