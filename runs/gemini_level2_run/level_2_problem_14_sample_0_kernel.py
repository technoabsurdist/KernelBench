import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation (divide -> sum -> scale)
fused_divide_sum_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel to perform fused (division -> sum -> scale)
// Each block processes one row of the input tensor.
__global__ void fused_divide_sum_scale_kernel(
    const float* input,      // Input tensor (output of matmul), shape (batch_size, hidden_size)
    float* output,           // Output tensor, shape (batch_size, 1)
    int batch_size,
    int hidden_size,
    float scaling_factor)
{
    // Use dynamically allocated shared memory for reduction within a block
    extern __shared__ float sdata[];

    // Get the row index for this block
    int b_idx = blockIdx.x;

    // Each thread computes a partial sum
    float my_sum = 0.0f;

    // Grid-stride loop to sum up elements in the row
    // Each thread processes elements strided by the block size
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        // Fused division and summation
        my_sum += input[b_idx * hidden_size + i] / 2.0f;
    }

    // Store partial sum in shared memory
    sdata[threadIdx.x] = my_sum;
    __syncthreads();

    // Perform reduction in shared memory. This is a standard parallel reduction algorithm.
    // The number of threads participating in the reduction halves each iteration.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // The first thread in the block writes the final result for the row
    if (threadIdx.x == 0) {
        // Fused scaling
        output[b_idx] = sdata[0] * scaling_factor;
    }
}

// C++ wrapper function to be called from Python
torch::Tensor fused_divide_sum_scale_cuda(
    torch::Tensor input,
    float scaling_factor)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    const auto batch_size = input.size(0);
    const auto hidden_size = input.size(1);

    // Create the output tensor with shape (batch_size, 1)
    auto output = torch::zeros({batch_size, 1}, input.options());

    // Kernel launch configuration
    // Use a common block size, e.g., 1024, which is the max on many architectures
    const int block_size = 1024;
    // Launch one block per row of the input tensor
    const int num_blocks = batch_size;
    
    // Shared memory size: one float per thread in the block
    const int shared_mem_size = block_size * sizeof(float);

    // Launch the kernel
    fused_divide_sum_scale_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hidden_size,
        scaling_factor
    );
    
    // Check for any CUDA errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

# C++ source for the function signature
fused_divide_sum_scale_cpp_source = (
    "torch::Tensor fused_divide_sum_scale_cuda(torch::Tensor input, float scaling_factor);"
)

# Compile the inline CUDA code
# This fuses the division, summation, and scaling operations into a single kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_divide_sum_scale_cpp_source,
    cuda_sources=fused_divide_sum_scale_source,
    functions=["fused_divide_sum_scale_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses the division, summation, and scaling operations
    into a single custom CUDA kernel. The high-performance torch.matmul (cuBLAS)
    is retained.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # 1. Perform matrix multiplication using the highly optimized PyTorch operator
        x = torch.matmul(x, self.weight.T)
        
        # 2. Call the custom CUDA kernel to perform the fused operation:
        #    - Element-wise division by 2
        #    - Summation along dim=1
        #    - Scaling by self.scaling_factor
        x = fused_op.fused_divide_sum_scale_cuda(x, self.scaling_factor)
        
        return x