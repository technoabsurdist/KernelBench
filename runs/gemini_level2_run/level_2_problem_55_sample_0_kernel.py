import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused MaxPool1d, Sum, and Scale
fused_pool_sum_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat> // For FLT_MAX

__global__ void fused_pool_sum_scale_kernel(
    const float* __restrict__ input,      // Input tensor data (B, C)
    float* __restrict__ output,           // Output tensor data (B)
    const int batch_size,          // B
    const int channels,            // C
    const int kernel_size,
    const float scale_factor
) {
    // Dynamically allocated shared memory for reduction within a block
    extern __shared__ float sdata[];

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int batch_idx = blockIdx.x; // Each block processes one batch item

    // Early exit for blocks that are out of bounds (if batch_size is not a multiple of gridDim.x)
    if (batch_idx >= batch_size) {
        return;
    }

    // Pointer to the current row in the input tensor for this block
    const float* row_input = input + batch_idx * channels;
    
    float partial_sum = 0.0f;
    const int pooled_len = channels / kernel_size;

    // Each thread computes a partial sum over a strided subset of the pooled elements
    for (int i = tid; i < pooled_len; i += block_size) {
        const int start_idx = i * kernel_size;
        
        // Find max in the window of size kernel_size
        float max_val = -FLT_MAX;
        for (int k = 0; k < kernel_size; ++k) {
            max_val = fmaxf(max_val, row_input[start_idx + k]);
        }
        partial_sum += max_val;
    }

    // Store partial sum in shared memory
    sdata[tid] = partial_sum;
    __syncthreads();

    // Perform reduction in shared memory. This is a standard parallel reduction.
    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 of each block writes the final reduced and scaled result for its batch item
    if (tid == 0) {
        output[batch_idx] = sdata[0] * scale_factor;
    }
}

torch::Tensor fused_pool_sum_scale_cuda(
    torch::Tensor input,
    int kernel_size,
    float scale_factor
) {
    // Input validation checks
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D (batch_size, features)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.size(1) % kernel_size == 0, "Input channels must be divisible by kernel_size");

    const int batch_size = input.size(0);
    const int channels = input.size(1);

    // Prepare the output tensor
    auto output = torch::zeros({batch_size}, input.options());

    // Configure and launch the kernel
    // We launch one block per batch item. Threads within the block cooperate on the reduction.
    const int block_size = 512; // A common choice for reduction kernels
    const int num_blocks = batch_size;
    
    // The size of shared memory required per block for the reduction
    const size_t smem_size = block_size * sizeof(float);

    fused_pool_sum_scale_kernel<<<num_blocks, block_size, smem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        kernel_size,
        scale_factor
    );
    
    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

fused_pool_sum_scale_cpp_source = (
    "torch::Tensor fused_pool_sum_scale_cuda(torch::Tensor input, int kernel_size, float scale_factor);"
)

# JIT compile the inline CUDA code
fused_pool_sum_scale = load_inline(
    name="fused_pool_sum_scale",
    cpp_sources=fused_pool_sum_scale_cpp_source,
    cuda_sources=fused_pool_sum_scale_source,
    functions=["fused_pool_sum_scale_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses MaxPool1d, Sum, and Scale operations into a single custom CUDA kernel.
    The expensive matrix multiplication is left to the highly optimized torch.nn.Linear layer.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        # The matmul is kept as a standard, highly optimized PyTorch layer
        self.matmul = nn.Linear(in_features, out_features)
        
        # Store parameters for the custom kernel
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

        # The custom fused operator is loaded and stored during initialization
        self.fused_op = fused_pool_sum_scale

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        # Step 1: Perform matrix multiplication using the standard nn.Linear
        x = self.matmul(x)
        
        # Step 2: Apply the fused MaxPool1d + Sum + Scale operation via the custom CUDA kernel.
        # The original unsqueeze/squeeze operations are no longer needed as the kernel
        # is designed to work directly on the 2D output of the linear layer.
        x = self.fused_op.fused_pool_sum_scale_cuda(x, self.kernel_size, self.scale_factor)
        
        return x