import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for a fused, memory-efficient Softmax
# This kernel performs the softmax operation along dim=1 for a 2D tensor.
# It's a single-kernel implementation that:
# 1. Finds the maximum value per row in a single pass.
# 2. Computes exp(x - max) and the sum of these exponentials in a second pass.
# 3. Normalizes the values in a final pass.
# This approach is fused into one kernel launch to reduce overhead and is optimized
# for memory bandwidth by minimizing reads from global memory.
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void softmax_kernel_dim1(const float* x, float* out, int dim) {
    // Shared memory for block-wide reductions
    extern __shared__ float s_data[];
    
    // Each block processes one row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const float* x_row = x + row * dim;
    float* out_row = out + row * dim;

    // ====================================================================
    // Step 1: Find the maximum value in the row using parallel reduction
    // ====================================================================
    float max_val = -FLT_MAX;
    // Each thread finds the max in its assigned portion of the row
    for (int i = tid; i < dim; i += block_size) {
        max_val = fmaxf(max_val, x_row[i]);
    }
    
    // Store thread's max in shared memory
    s_data[tid] = max_val;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]);
        }
        __syncthreads();
    }
    // The final max for the row is in s_data[0]
    max_val = s_data[0];
    __syncthreads();

    // =================================================================================
    // Step 2: Compute exponentials, store them temporarily in 'out', and find their sum
    // =================================================================================
    float sum_val = 0.0f;
    for (int i = tid; i < dim; i += block_size) {
        float val = expf(x_row[i] - max_val);
        out_row[i] = val; // Store intermediate result
        sum_val += val;
    }

    // Store thread's sum in shared memory
    s_data[tid] = sum_val;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    // The final sum for the row is in s_data[0]
    sum_val = s_data[0];
    __syncthreads();

    // ====================================================================
    // Step 3: Normalize the values in 'out'
    // ====================================================================
    // Add a small epsilon for numerical stability
    float inv_sum = 1.0f / (sum_val + 1e-8f);
    for (int i = tid; i < dim; i += block_size) {
        out_row[i] *= inv_sum;
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    const auto batch_size = x.size(0);
    const auto dim = x.size(1);

    auto out = torch::empty_like(x);

    // Kernel launch configuration
    // Use 1024 threads per block, a common choice for high occupancy.
    // This must be a power of 2 for the reduction logic to work correctly.
    const int block_size = 1024;
    // Launch one block per row in the batch
    const int grid_size = batch_size;
    // Allocate shared memory for the reduction
    const int shared_mem_size = block_size * sizeof(float);

    softmax_kernel_dim1<<<grid_size, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        dim
    );
    
    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# C++ source for PyTorch binding
softmax_cpp_source = "torch::Tensor softmax_cuda(torch::Tensor x);"

# JIT compile the custom CUDA kernel
custom_softmax = load_inline(
    name="custom_softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that uses a custom CUDA kernel for the Softmax activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a custom high-performance Softmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        # Call the custom CUDA function
        return custom_softmax.softmax_cuda(x)

batch_size = 4096
dim = 393216

def get_inputs():
    # Ensure input is on CUDA
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed