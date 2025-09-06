import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for argmax along dimension 1
argmax_dim1_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat> // For FLT_MAX

// CUDA kernel to perform argmax along dimension 1 of a 3D tensor
// Input tensor shape: (N, C, H)
// Output tensor shape: (N, H)
__global__ void argmax_dim1_kernel(const float* x, long* out, int N, int C, int H) {
    // Calculate the global thread indices for the output tensor (N, H)
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure the thread is within the bounds of the output tensor
    if (n < N && h < H) {
        // Pointer to the first element for this (n, h) slice: x[n, 0, h]
        const float* x_ptr = x + n * C * H + h;
        
        float max_val = -FLT_MAX;
        long max_idx = -1;

        // Iterate along the C dimension to find the max value and its index
        for (int c = 0; c < C; ++c) {
            // Access x[n, c, h] with a stride of H
            float current_val = x_ptr[c * H];
            if (current_val > max_val) {
                max_val = current_val;
                max_idx = c;
            }
        }

        // Write the resulting index to the output tensor at position (n, h)
        out[n * H + h] = max_idx;
    }
}

// C++ wrapper function that launches the CUDA kernel
torch::Tensor argmax_dim1_cuda(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "Input tensor must be 3-dimensional");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be of type float32");

    // Get tensor dimensions
    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);

    // Create the output tensor for the indices
    // Shape will be (N, H), and dtype will be int64 (long)
    auto out = torch::empty({N, H}, x.options().dtype(torch::kInt64));

    // Define grid and block dimensions for the kernel launch
    const dim3 block_dim(16, 16);
    const dim3 grid_dim(
        (H + block_dim.x - 1) / block_dim.x,
        (N + block_dim.y - 1) / block_dim.y
    );

    // Launch the kernel
    argmax_dim1_kernel<<<grid_dim, block_dim>>>(
        x.data_ptr<float>(),
        out.data_ptr<long>(),
        N, C, H
    );
    
    // Check for any CUDA errors after kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

argmax_dim1_cpp_source = "torch::Tensor argmax_dim1_cuda(torch::Tensor x);"

# Compile the inline CUDA code for argmax
argmax_custom = load_inline(
    name="argmax_custom",
    cpp_sources=argmax_dim1_cpp_source,
    cuda_sources=argmax_dim1_source,
    functions=["argmax_dim1_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that uses a custom CUDA kernel for Argmax over a specified dimension.
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
        Applies argmax over the specified dimension to the input tensor.
        Uses a custom CUDA kernel if dim is 1, otherwise falls back to torch.argmax.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        # The custom kernel is highly specialized for a 3D tensor and dim=1.
        if self.dim == 1 and x.is_cuda and x.dim() == 3:
            return argmax_custom.argmax_dim1_cuda(x)
        else:
            # Fallback to the original PyTorch implementation for other cases
            return torch.argmax(x, dim=self.dim)