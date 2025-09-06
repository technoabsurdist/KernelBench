import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused sigmoid and sum
sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For expf

// A simple device function for sigmoid
__device__ inline float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Kernel to compute sigmoid and then sum along a dimension.
// Each block is responsible for reducing one row of the input matrix.
__global__ void sigmoid_sum_kernel(const float* input, float* output, int batch_size, int hidden_size) {
    // Each block processes one row of the input matrix
    int row = blockIdx.x;
    if (row >= batch_size) {
        return;
    }

    // Use dynamically allocated shared memory for reduction within a block.
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    float local_sum = 0.0f;

    const float* row_input = input + row * hidden_size;

    // Each thread computes a partial sum for its assigned row using a grid-stride loop.
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        local_sum += sigmoidf(row_input[i]);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // Perform parallel reduction in shared memory.
    // This implementation assumes blockDim.x is a power of 2.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 of each block writes the final reduced sum to the output tensor.
    if (tid == 0) {
        output[row] = sdata[0];
    }
}

torch::Tensor sigmoid_sum_cuda(torch::Tensor input) {
    // Input validation checks
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D");

    const auto batch_size = input.size(0);
    const auto hidden_size = input.size(1);

    // Allocate the output tensor
    auto output = torch::zeros({batch_size, 1}, input.options());

    // Kernel launch configuration
    const int block_size = 512; // A common choice for reduction kernels
    const int num_blocks = batch_size; // One block per row

    // Shared memory size is determined by the block size
    const size_t shared_mem_size = block_size * sizeof(float);

    // Launch the kernel
    sigmoid_sum_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hidden_size
    );

    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

sigmoid_sum_cpp_source = """
torch::Tensor sigmoid_sum_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code using torch's C++ extension loader
sigmoid_sum_fused = load_inline(
    name="sigmoid_sum_fused",
    cpp_sources=sigmoid_sum_cpp_source,
    cuda_sources=sigmoid_sum_source,
    functions=["sigmoid_sum_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses the sigmoid and sum operations into a single custom CUDA kernel.
    The nn.Linear layer is kept as is, since it is highly optimized (typically using cuBLAS).
    The fusion reduces memory bandwidth by avoiding the materialization of the intermediate
    tensor after the sigmoid operation.
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        # Store the custom fused operator
        self.sigmoid_sum = sigmoid_sum_fused.sigmoid_sum_cuda

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        # Step 1: Perform the linear transformation.
        x = self.linear(x)
        # Step 2: Apply the custom fused sigmoid + sum kernel.
        x = self.sigmoid_sum(x)
        return x