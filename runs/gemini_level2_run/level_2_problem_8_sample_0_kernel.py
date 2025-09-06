import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel source for the fused operation
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel to perform Division, Global Average Pooling, and Bias Addition in a fused manner.
// This kernel replaces the sequence: x / divisor -> global_avg_pool -> + bias
// The grid is dimensioned (N, C), where each block computes the result for one channel of one batch item.
__global__ void fused_div_avg_bias_kernel(
    const float* input,      // Input tensor from max_pool (N, C, D, H, W)
    const float* bias,       // Bias tensor (C)
    float* output,           // Output tensor (N, C)
    const float divisor,
    const int N,
    const int C,
    const int D,
    const int H,
    const int W)
{
    // Use dynamically allocated shared memory for the reduction
    extern __shared__ float sdata[];

    // Get batch and channel indices from block indices
    const int n = blockIdx.x;
    const int c = blockIdx.y;

    // Each thread's local sum for the reduction
    float thread_sum = 0.0f;

    // Calculate the number of spatial elements (D*H*W) and the offset to the start of the data
    // for the current (n, c) slice. Use long long for safety with large tensors.
    const long long num_spatial_elements = (long long)D * H * W;
    const long long slice_offset = ((long long)n * C + c) * num_spatial_elements;
    const float* input_slice = input + slice_offset;

    // Each thread sums a portion of the spatial elements in a grid-stride loop
    for (long long i = threadIdx.x; i < num_spatial_elements; i += blockDim.x) {
        thread_sum += input_slice[i];
    }

    // Store the thread's partial sum in shared memory
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // Perform reduction in shared memory. This is a standard parallel reduction algorithm.
    // It requires the block size to be a power of 2 for correctness.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 of each block computes the final value and writes it to the output
    if (threadIdx.x == 0) {
        // The total sum for the (n, c) slice is now in sdata[0]
        float total_sum = sdata[0];

        // Perform the fused operations:
        // 1. Global Average Pooling: (sum / num_elements)
        // 2. Division: (average / divisor)
        // 3. Bias Addition: (+ bias[c])
        float avg = total_sum / (float)num_spatial_elements;
        float result = (avg / divisor) + bias[c];

        // Write the final result to the output tensor
        output[n * C + c] = result;
    }
}

// C++ wrapper function that will be called from Python
torch::Tensor fused_div_avg_bias_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    const float divisor)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias tensor must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input tensor must be a float32 tensor");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "Bias tensor must be a float32 tensor");
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5D");
    TORCH_CHECK(bias.dim() == 1, "Bias tensor must be 1D for this kernel");

    // Get tensor dimensions
    const auto sizes = input.sizes();
    const int N = sizes[0];
    const int C = sizes[1];
    const int D = sizes[2];
    const int H = sizes[3];
    const int W = sizes[4];

    TORCH_CHECK(C == bias.size(0), "Input channel dimension must match bias size");

    // Create the output tensor of shape (N, C)
    auto output = torch::zeros({N, C}, input.options());

    // Configure the kernel launch
    const dim3 grid_dim(N, C);
    // Block size should be a power of 2 for the reduction algorithm. 256 is a common choice.
    const int block_size = 256;
    const dim3 block_dim(block_size);

    // Shared memory size: one float per thread in the block
    const int shared_mem_size = block_size * sizeof(float);

    // Launch the CUDA kernel
    fused_div_avg_bias_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        divisor,
        N, C, D, H, W
    );

    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

# Define the C++ source for the function signature
fused_op_cpp_source = """
torch::Tensor fused_div_avg_bias_cuda(torch::Tensor input, torch::Tensor bias, const float divisor);
"""

# Use torch's JIT compiler to build the custom operator
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_div_avg_bias_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses the Division, Global Average Pooling, and Bias Addition
    operations into a single custom CUDA kernel. The order of Division and MaxPool is
    swapped, which is a valid mathematical transformation for a positive divisor.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        # Standard PyTorch operators for complex operations like Conv3d and MaxPool3d
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.max_pool = nn.MaxPool3d(pool_size)
        
        # Store parameters needed for the fused operation and final sum
        self.divisor = divisor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        
        # The custom fused operator compiled above
        self.fused_op = fused_op

    def forward(self, x):
        # 1. 3D Convolution (using efficient cuDNN backend)
        x = self.conv(x)
        
        # 2. 3D Max Pooling (using efficient cuDNN backend)
        # Note: We swapped the original order of (division -> max_pool) to
        # (max_pool -> division) to enable the fusion. This is mathematically
        # equivalent since max_pool(x / c) == max_pool(x) / c for c > 0.
        x = self.max_pool(x)
        
        # 3. Fused Operation: Division + GlobalAvgPool + BiasAdd
        # The custom kernel expects a 1D bias tensor, so we squeeze the parameter
        # from its original (C, 1, 1, 1) shape to (C,).
        squeezed_bias = self.bias.squeeze()
        x = self.fused_op.fused_div_avg_bias_cuda(x, squeezed_bias, self.divisor)
        
        # 4. Summation
        # The output of our fused kernel is a 2D tensor of shape (N, C).
        # We sum along the channel dimension (dim=1), which is equivalent to the
        # original model's sum operation. The result is a 1D tensor of shape (N,).
        x = torch.sum(x, dim=self.sum_dim)
        
        # 5. Reshape Output
        # The original model's output shape was (N, 1, 1, 1). We reshape our
        # (N,) tensor to match it.
        return x.view(-1, 1, 1, 1)