import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused masked cumulative sum
masked_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void masked_cumsum_kernel(const float* x, const bool* mask, float* out, int rows, int cols) {
    // Each thread is responsible for processing one row
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < rows) {
        float current_sum = 0.0f;
        // Calculate the starting index for this row in the flattened 1D array
        int row_start_idx = row_idx * cols;

        for (int col_idx = 0; col_idx < cols; ++col_idx) {
            int current_idx = row_start_idx + col_idx;
            
            // Fused operation: Check the mask and add the value from x if the mask is true.
            // This avoids creating an intermediate tensor for (x * mask).
            if (mask[current_idx]) {
                current_sum += x[current_idx];
            }
            
            // Store the resulting cumulative sum at this position
            out[current_idx] = current_sum;
        }
    }
}

torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int64_t dim) {
    // Input validation to ensure tensors and parameters are compatible with the kernel
    TORCH_CHECK(x.dim() == 2, "Input tensor x must be 2D for this custom kernel.");
    TORCH_CHECK(mask.dim() == 2, "Mask tensor must be 2D for this custom kernel.");
    TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have the same shape.");
    TORCH_CHECK(dim == 1, "Custom kernel only supports cumsum along dim=1.");
    TORCH_CHECK(x.is_cuda() && mask.is_cuda(), "Tensors must be on a CUDA device.");
    TORCH_CHECK(x.is_contiguous() && mask.is_contiguous(), "Tensors must be contiguous.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor x must be of type float32.");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "Mask tensor must be of type bool.");

    auto rows = x.size(0);
    auto cols = x.size(1);

    // Allocate the output tensor. Using empty_like is efficient as all values will be overwritten.
    auto out = torch::empty_like(x);

    // Configure kernel launch parameters. We launch one thread per row.
    const int block_size = 256;
    const int num_blocks = (rows + block_size - 1) / block_size;

    // Launch the CUDA kernel
    masked_cumsum_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        mask.data_ptr<bool>(),
        out.data_ptr<float>(),
        rows,
        cols
    );

    // Check for any errors during kernel execution (important for debugging)
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# C++ source for the function signature, required by load_inline
masked_cumsum_cpp_source = (
    "torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int64_t dim);"
)

# Compile the inline CUDA code
# This creates a Python module with the 'masked_cumsum_cuda' function
masked_cumsum_op = load_inline(
    name="masked_cumsum_op",
    cpp_sources=masked_cumsum_cpp_source,
    cuda_sources=masked_cumsum_source,
    functions=["masked_cumsum_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    An optimized model that performs a masked cumulative sum using a custom fused CUDA kernel.
    The kernel combines the element-wise multiplication by the mask and the cumulative sum
    into a single operation, improving performance by reducing memory bandwidth and kernel launch overhead.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
                   (Note: The custom kernel is optimized for and only supports dim=1).
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        # The custom kernel is specifically written for dim=1 on a 2D tensor.
        # We add a check to ensure the model is initialized with a supported dimension.
        if self.dim != 1:
            raise ValueError("Custom CUDA kernel for ModelNew only supports dim=1.")
        self.masked_cumsum_op = masked_cumsum_op

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.

        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True, computed by the custom kernel.
        """
        # Call the custom CUDA function from the compiled module
        return self.masked_cumsum_op.masked_cumsum_cuda(x, mask, self.dim)