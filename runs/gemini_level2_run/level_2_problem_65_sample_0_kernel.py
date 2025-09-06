import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused AvgPool + Sigmoid + Sum
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Device function for sigmoid
__device__ inline float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Fused kernel for AvgPool -> Sigmoid -> Sum
// Each block processes one item in the batch.
// Threads within a block cooperate to calculate the sum for that item.
__global__ void fused_pool_sigmoid_sum_kernel(
    const float* input,      // Input tensor (output of conv)
    float* output,           // Output tensor (final sum per batch item)
    int N, int C, int H, int W, // Input dimensions
    int H_out, int W_out,       // Pooled output dimensions
    int pool_kernel_size,
    int pool_stride) {

    // Shared memory for block-level reduction
    extern __shared__ float sdata[];

    // Each block processes one batch item
    int n = blockIdx.x;
    if (n >= N) return;

    // Per-thread partial sum
    float partial_sum = 0.0f;

    // Total number of elements to process per batch item (after pooling)
    long long total_elements_per_batch = (long long)C * H_out * W_out;

    // Grid-stride loop for this block's threads to process all elements for batch 'n'
    for (long long i = threadIdx.x; i < total_elements_per_batch; i += blockDim.x) {
        // Decode linear index 'i' into (c, h_out, w_out)
        int w_out = i % W_out;
        int h_out = (i / W_out) % H_out;
        int c = i / (W_out * H_out);

        // --- Start Average Pooling ---
        float current_sum = 0.0f;
        int h_start = h_out * pool_stride;
        int w_start = w_out * pool_stride;

        for (int ph = 0; ph < pool_kernel_size; ++ph) {
            for (int pw = 0; pw < pool_kernel_size; ++pw) {
                int h_in = h_start + ph;
                int w_in = w_start + pw;
                // The input tensor is laid out as NCHW
                long long in_idx = (long long)n * C * H * W +
                                   (long long)c * H * W +
                                   (long long)h_in * W +
                                   w_in;
                current_sum += input[in_idx];
            }
        }
        float average = current_sum / (pool_kernel_size * pool_kernel_size);
        // --- End Average Pooling ---

        // --- Sigmoid ---
        float sigmoid_val = sigmoidf(average);

        // Accumulate to the thread's partial sum
        partial_sum += sigmoid_val;
    }

    // --- Block-level Reduction ---
    sdata[threadIdx.x] = partial_sum;
    __syncthreads();

    // Reduce in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final result for this batch item
    if (threadIdx.x == 0) {
        output[n] = sdata[0];
    }
}

// C++ wrapper function that PyTorch will call
torch::Tensor fused_pool_sigmoid_sum_cuda(torch::Tensor input, int pool_kernel_size) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input tensor must be 4D (NCHW)");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);

    // AvgPool2d default stride is kernel_size
    const int pool_stride = pool_kernel_size;
    const int H_out = (H - pool_kernel_size) / pool_stride + 1;
    const int W_out = (W - pool_kernel_size) / pool_stride + 1;

    TORCH_CHECK(H_out > 0 && W_out > 0, "Pooling results in 0-sized dimension. Check input size and pool_kernel_size.");

    // The output is a 1D tensor of size N (batch_size)
    auto output = torch::zeros({N}, input.options());

    const int block_size = 256;
    const int num_blocks = N;
    // Allocate shared memory for the reduction
    const int shared_mem_size = block_size * sizeof(float);

    fused_pool_sigmoid_sum_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        H_out, W_out,
        pool_kernel_size,
        pool_stride
    );
    
    // Check for any CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_pool_sigmoid_sum_cuda(torch::Tensor input, int pool_kernel_size);
"""

# JIT compile the CUDA kernel
fused_op = load_inline(
    name="fused_pool_sigmoid_sum",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_pool_sigmoid_sum_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    This model performs a convolution, then a custom fused operation for 
    (average pooling + sigmoid + sum) to improve performance by reducing
    kernel launch overhead and memory bandwidth usage.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        # The convolution layer remains a standard PyTorch module, as it's 
        # highly optimized by the underlying cuDNN library.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
        # Store the pooling kernel size to pass to our custom op
        self.pool_kernel_size = pool_kernel_size
        
        # Store the compiled custom CUDA function
        self.fused_op = fused_op

    def forward(self, x):
        # 1. Apply the standard, highly-optimized convolution
        x = self.conv(x)
        
        # 2. Apply the custom fused AvgPool + Sigmoid + Sum CUDA kernel
        # This replaces three separate operations with a single kernel launch.
        x = self.fused_op.fused_pool_sigmoid_sum_cuda(x, self.pool_kernel_size)
        
        return x