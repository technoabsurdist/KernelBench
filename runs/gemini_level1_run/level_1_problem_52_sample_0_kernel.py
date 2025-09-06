import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for argmin along dim=1
argmin_dim1_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void argmin_dim1_kernel(
    const float* __restrict__ input,
    int64_t* __restrict__ output,
    const int dim0_size,
    const int dim1_size,
    const int dim2_size) {

    // Calculate the global thread index for the output tensor
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of reductions to perform (size of the output tensor)
    const int total_reductions = dim0_size * dim2_size;

    // Boundary check
    if (idx < total_reductions) {
        // Map the 1D output index back to 2D coordinates (dim0, dim2)
        const int b = idx / dim2_size; // Index for dimension 0
        const int k = idx % dim2_size; // Index for dimension 2

        float min_val = FLT_MAX;
        int64_t min_idx = 0; // In case of all NaNs or empty dim, PyTorch returns 0

        // Calculate the starting memory offset for the current reduction slice
        const int start_offset = b * dim1_size * dim2_size + k;
        // The stride to move along dimension 1
        const int dim1_stride = dim2_size;

        // Iterate along dimension 1 to find the minimum value and its index
        for (int j = 0; j < dim1_size; ++j) {
            const float current_val = input[start_offset + j * dim1_stride];
            if (current_val < min_val) {
                min_val = current_val;
                min_idx = j;
            }
        }
        // Write the resulting index to the output tensor
        output[idx] = min_idx;
    }
}

torch::Tensor argmin_dim1_cuda(torch::Tensor x) {
    // Input validation checks
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "Input tensor must be 3-dimensional, but got ", x.dim());
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be of type float32");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    // Get input dimensions
    const int dim0_size = x.size(0);
    const int dim1_size = x.size(1);
    const int dim2_size = x.size(2);

    // Create the output tensor with the correct shape and type (int64)
    auto output_options = torch::TensorOptions().device(x.device()).dtype(torch::kInt64);
    auto output = torch::empty({dim0_size, dim2_size}, output_options);

    // If the dimension to reduce is empty, return a tensor of zeros
    if (dim1_size == 0) {
        output.zero_();
        return output;
    }

    // Kernel launch configuration
    const int total_reductions = dim0_size * dim2_size;
    const int block_size = 256;
    const int num_blocks = (total_reductions + block_size - 1) / block_size;

    // Launch the CUDA kernel
    argmin_dim1_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        dim0_size,
        dim1_size,
        dim2_size
    );
    
    // Check for errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

argmin_dim1_cpp_source = (
    "torch::Tensor argmin_dim1_cuda(torch::Tensor x);"
)

# JIT compile the CUDA code
argmin_dim1 = load_inline(
    name="argmin_dim1",
    cpp_sources=argmin_dim1_cpp_source,
    cuda_sources=argmin_dim1_source,
    functions=["argmin_dim1_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that finds the index of the minimum value along a specified dimension
    using a custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Finds the index of the minimum value along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor containing the indices of the minimum values along the specified dimension.
        """
        # Use the custom CUDA kernel only if the conditions are met:
        # 1. The specified dimension is 1.
        # 2. The input tensor is 3-dimensional.
        # 3. The input tensor is on a CUDA device.
        if self.dim == 1 and x.dim() == 3 and x.is_cuda:
            return argmin_dim1.argmin_dim1_cuda(x)
        else:
            # Fallback to the original PyTorch implementation for all other cases
            return torch.argmin(x, dim=self.dim)