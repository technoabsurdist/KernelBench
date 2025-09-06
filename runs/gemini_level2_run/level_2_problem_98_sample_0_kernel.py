import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: AvgPool -> GELU -> Scale -> Max
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// GELU approximation using tanh, which is often faster and used in implementations like OpenAI's
__device__ __forceinline__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// Fused kernel that performs:
// 1. 1D Average Pooling
// 2. GELU activation
// 3. Scaling
// 4. Max reduction along the feature dimension
// This is all done for each row in the batch independently.
__global__ void fused_op_kernel(
    const float* input,      // Input tensor of shape (batch_size, out_features)
    float* output,           // Output tensor of shape (batch_size,)
    int out_features,
    int pool_kernel_size,
    float scale_factor
) {
    // Each CUDA block is responsible for processing one row of the batch.
    int row_idx = blockIdx.x;
    const float* row_input = input + row_idx * out_features;

    // Use shared memory for the final block-level reduction (for the Max operation).
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int num_pooled_features = out_features / pool_kernel_size;

    // Each thread initializes its local maximum value.
    float thread_max = -FLT_MAX;

    // This is a grid-stride loop. Each thread processes multiple pooled elements
    // if the number of pooled features is larger than the block size.
    for (int i = tid; i < num_pooled_features; i += blockDim.x) {
        // Step 1: AvgPool
        float sum = 0.0f;
        const float* pool_start = row_input + i * pool_kernel_size;
        // The compiler can often unroll this loop if pool_kernel_size is a reasonable constant.
        for (int k = 0; k < pool_kernel_size; ++k) {
            sum += pool_start[k];
        }
        float avg = sum / (float)pool_kernel_size;

        // Step 2: GELU
        float gelu_val = gelu_approx(avg);

        // Step 3: Scale
        float scaled_val = gelu_val * scale_factor;

        // Update the thread-local maximum value.
        thread_max = fmaxf(thread_max, scaled_val);
    }

    // Store the thread-local maximum in shared memory.
    sdata[tid] = thread_max;
    __syncthreads();

    // Perform a parallel reduction in shared memory to find the maximum value in the block.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // The first thread in the block writes the final reduced maximum value to the output tensor.
    if (tid == 0) {
        output[row_idx] = sdata[0];
    }
}

// C++ wrapper function that will be called from Python.
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    int pool_kernel_size,
    float scale_factor
) {
    // Input validation checks.
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input must be a 2D tensor");

    const auto batch_size = input.size(0);
    const auto out_features = input.size(1);

    TORCH_CHECK(out_features % pool_kernel_size == 0, "out_features must be divisible by pool_kernel_size for this kernel");

    // Allocate the output tensor.
    auto output = torch::empty({batch_size}, input.options());

    // Configure and launch the CUDA kernel.
    const int block_size = 256;
    const int num_blocks = batch_size;
    const size_t shared_mem_size = block_size * sizeof(float);

    fused_op_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        out_features,
        pool_kernel_size,
        scale_factor
    );
    
    // Check for any CUDA errors after kernel launch.
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

# C++ source for defining the function signature for PyTorch's JIT compiler.
fused_op_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    int pool_kernel_size,
    float scale_factor
);
"""

# Use load_inline to JIT compile the C++/CUDA source code.
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    An optimized model that fuses "AvgPool_GELU_Scale_Max" into a single custom CUDA kernel.
    The initial Matmul (nn.Linear) is kept as a standard PyTorch operator because it is
    highly optimized in cuBLAS. The main performance gain comes from fusing the subsequent
    memory-bound operations, which reduces kernel launch overhead and memory bandwidth usage.
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        # The computationally intensive matrix multiplication is handled by PyTorch's nn.Linear,
        # which is a thin wrapper around the highly optimized cuBLAS library.
        self.matmul = nn.Linear(in_features, out_features)
        
        # Store parameters needed for the custom kernel.
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        
        # Assign the compiled custom operator.
        self.fused_op = fused_op

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        # 1. Perform the matrix multiplication using the standard, optimized PyTorch operator.
        x = self.matmul(x)
        
        # 2. Call the single custom CUDA kernel to perform the sequence of
        #    AvgPool -> GELU -> Scale -> Max reduction.
        x = self.fused_op.fused_op_cuda(x, self.pool_kernel_size, self.scale_factor)
        
        return x