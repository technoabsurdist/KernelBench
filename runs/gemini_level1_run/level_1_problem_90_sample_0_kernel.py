import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and its C++ wrapper
cumprod_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cumprod_dim1_kernel(const float* __restrict__ in_data, float* __restrict__ out_data, int rows, int cols) {
    // Each thread is responsible for computing the cumulative product for one entire row.
    // This approach is efficient when 'rows' is large, as it maximizes parallelism across rows.
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        const float* in_row = in_data + row * cols;
        float* out_row = out_data + row * cols;

        // Sequentially compute the cumulative product along the row.
        float current_prod = 1.0f;
        for (int col = 0; col < cols; ++col) {
            current_prod *= in_row[col];
            out_row[col] = current_prod;
        }
    }
}

torch::Tensor cumprod_cuda(torch::Tensor x, int64_t dim) {
    // Input validation checks
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D for this custom kernel");
    TORCH_CHECK(dim == 1, "Custom kernel only supports cumprod along dim=1");
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");

    auto rows = x.size(0);
    auto cols = x.size(1);

    // Allocate output tensor
    auto out = torch::empty_like(x);

    // Kernel launch configuration
    const int block_size = 256;
    const int num_blocks = (rows + block_size - 1) / block_size;

    // Launch the kernel
    cumprod_dim1_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols
    );
    
    // Check for any CUDA errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return out;
}
"""

# Define the C++ function signature for the JIT compiler
cumprod_cpp_source = (
    "torch::Tensor cumprod_cuda(torch::Tensor x, int64_t dim);"
)

# Compile the inline CUDA code using torch.utils.cpp_extension.load_inline
# This creates a Python module on the fly that we can call.
custom_cumprod = load_inline(
    name="custom_cumprod",
    cpp_sources=cumprod_cpp_source,
    cuda_sources=cumprod_kernel_source,
    functions=["cumprod_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    A model that performs a cumulative product operation along a specified dimension
    using a custom CUDA kernel. The kernel is optimized for 2D tensors and dim=1.
    """

    def __init__(self, dim):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        if self.dim != 1:
            # The custom kernel is specialized for dim=1.
            # Fallback to the original PyTorch implementation for other dimensions
            # could be an option, but for this problem, we'll raise an error.
            raise NotImplementedError("Custom CUDA kernel only supports dim=1 for 2D tensors.")

    def forward(self, x):
        """
        Forward pass, computing the cumulative product along the specified dimension
        using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        return custom_cumprod.cumprod_cuda(x, self.dim)