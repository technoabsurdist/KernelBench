import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: MaxPool2d -> Hardtanh -> Mean -> Tanh
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Device function for Hardtanh activation
__device__ inline float hardtanh_func(float val, float min_val, float max_val) {
    return fminf(fmaxf(val, min_val), max_val);
}

__global__ void maxpool_hardtanh_mean_tanh_fused_kernel(
    const float* input,
    float* output,
    const int B,
    const int C,
    const int H_in,
    const int W_in,
    const int H_out,
    const int W_out,
    const int pool_kernel,
    const int pool_stride,
    const float hardtanh_min,
    const float hardtanh_max) {

    // Each block computes one output value for a given (batch, channel) pair.
    // blockIdx.x corresponds to the batch index 'b'.
    // blockIdx.y corresponds to the channel index 'c'.
    const int b = blockIdx.x;
    const int c = blockIdx.y;

    // Shared memory for the reduction step.
    extern __shared__ float sdata[];

    // Each thread computes a partial sum over the elements it's assigned.
    float my_sum = 0.0f;

    // Total number of elements to reduce per block (i.e., per channel after pooling).
    const int num_elements_to_reduce = H_out * W_out;

    // Grid-stride loop: each thread processes multiple elements if needed.
    for (int i = threadIdx.x; i < num_elements_to_reduce; i += blockDim.x) {
        // Convert 1D index 'i' to 2D pooled coordinates (h_pool, w_pool).
        const int h_pool = i / W_out;
        const int w_pool = i % W_out;

        // Find the top-left corner in the input tensor for the current pooling window.
        const int h_start = h_pool * pool_stride;
        const int w_start = w_pool * pool_stride;

        // --- Start of Max Pooling ---
        float max_val = -1e20f; // Initialize with a very small number.
        for (int ph = 0; ph < pool_kernel; ++ph) {
            for (int pw = 0; pw < pool_kernel; ++pw) {
                const int current_h = h_start + ph;
                const int current_w = w_start + pw;
                // Boundary check for the pooling window.
                if (current_h < H_in && current_w < W_in) {
                    const int input_idx = b * C * H_in * W_in +
                                          c * H_in * W_in +
                                          current_h * W_in +
                                          current_w;
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }
        // --- End of Max Pooling ---

        // --- Start of Hardtanh ---
        float hardtanh_val = hardtanh_func(max_val, hardtanh_min, hardtanh_max);
        // --- End of Hardtanh ---

        // Add the result to this thread's partial sum for the mean calculation.
        my_sum += hardtanh_val;
    }

    // Store the partial sum in shared memory.
    sdata[threadIdx.x] = my_sum;
    __syncthreads();

    // --- Start of Mean (Reduction part 1: Sum) ---
    // Perform parallel reduction in shared memory.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    // --- End of Mean (Sum) ---

    // Thread 0 of each block computes the final value and writes to global memory.
    if (threadIdx.x == 0) {
        float total_sum = sdata[0];
        
        // --- Start of Mean (Division part 2) ---
        float mean_val = total_sum / num_elements_to_reduce;
        // --- End of Mean ---

        // --- Start of Tanh ---
        float final_val = tanhf(mean_val);
        // --- End of Tanh ---

        // The output tensor has shape [B, C, 1, 1], so its memory is contiguous
        // like a [B, C] tensor.
        const int output_idx = b * C + c;
        output[output_idx] = final_val;
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor input,
    int pool_kernel,
    int pool_stride,
    float hardtanh_min,
    float hardtanh_max) {

    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.dim() == 4, "Input tensor must be 4D");

    const int B = input.size(0);
    const int C = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    // Calculate output dimensions after pooling (assuming padding=0, dilation=1).
    const int H_out = (H_in - pool_kernel) / pool_stride + 1;
    const int W_out = (W_in - pool_kernel) / pool_stride + 1;
    TORCH_CHECK(H_out > 0 && W_out > 0, "Pooled dimensions must be positive.");

    // The final output tensor shape after mean reduction.
    auto output = torch::zeros({B, C, 1, 1}, input.options());

    // Kernel launch configuration.
    const int block_size = 256; // A common choice, should be a power of 2 for reduction.
    dim3 threads(block_size);
    dim3 blocks(B, C); // Launch one block per (batch, channel) pair.

    // Shared memory size: one float per thread in the block.
    const int shared_mem_size = block_size * sizeof(float);

    // Launch the fused kernel.
    maxpool_hardtanh_mean_tanh_fused_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H_in, W_in, H_out, W_out,
        pool_kernel, pool_stride,
        hardtanh_min, hardtanh_max
    );

    return output;
}
"""

# C++ source for the binding function signature.
fused_op_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    int pool_kernel,
    int pool_stride,
    float hardtanh_min,
    float hardtanh_max);
"""

# Use JIT compilation to build the custom CUDA operator.
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that replaces the sequence of MaxPool, Hardtanh, Mean, and Tanh
    with a single, fused custom CUDA kernel. The ConvTranspose2d operation is
    retained from PyTorch as it is already highly optimized by cuDNN.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        # 1. Transposed Convolution: Best to use the highly optimized PyTorch/cuDNN implementation.
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

        # Store parameters needed for the custom fused kernel.
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

        # 2. The custom fused operator that combines the remaining operations.
        self.fused_op = fused_op

    def forward(self, x):
        # Apply the standard ConvTranspose2d layer first.
        x = self.conv_transpose(x)

        # Apply the single fused CUDA kernel for MaxPool -> Hardtanh -> Mean -> Tanh.
        x = self.fused_op.fused_op_cuda(
            x,
            self.maxpool_kernel_size,
            self.maxpool_stride,
            self.hardtanh_min,
            self.hardtanh_max
        )
        return x