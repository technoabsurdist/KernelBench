import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Softmax + MaxPool3d
softmax_pool3d_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Helper for warp-level reduction to find the maximum value
__device__ __forceinline__ float warp_reduce_max(float val) {
    // Assumes warp size is 32
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Helper for warp-level reduction to sum values
__device__ __forceinline__ float warp_reduce_sum(float val) {
    // Assumes warp size is 32
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void softmax_pool3d_fused_kernel(
    const float* input,
    float* output,
    int N, int C, int D, int H, int W,
    int D_out, int H_out, int W_out,
    int pool_size
) {
    // This kernel assumes C <= 32 (fits in a warp) and C is the block size.
    // Each block computes one output spatial location (n, d_out, h_out, w_out) across all C channels.

    const int pool_cube = pool_size * pool_size * pool_size;

    // Shared memory to hold the input patch after softmax is applied.
    // Size: C * pool_cube
    extern __shared__ float s_patch[];

    // Block index determines the output spatial location
    int w_out = blockIdx.x;
    int h_out = blockIdx.y;
    int n_d_out = blockIdx.z;
    int n = n_d_out / D_out;
    int d_out = n_d_out % D_out;

    // Thread index determines the channel
    int c = threadIdx.x;

    // Base input coordinates for the pooling window
    int d_in_start = d_out * pool_size;
    int h_in_start = h_out * pool_size;
    int w_in_start = w_out * pool_size;

    // Step 1: Iterate through each spatial location in the pooling window.
    // For each location, compute softmax across channels and store the result in shared memory.
    for (int p_idx = 0; p_idx < pool_cube; ++p_idx) {
        int kd = p_idx / (pool_size * pool_size);
        int kh = (p_idx % (pool_size * pool_size)) / pool_size;
        int kw = p_idx % pool_size;

        int d_in = d_in_start + kd;
        int h_in = h_in_start + kh;
        int w_in = w_in_start + kw;

        // Load value, handling padding by using a large negative number for out-of-bounds accesses.
        float my_val = -1.0e20f;
        if (d_in < D && h_in < H && w_in < W) {
            long long input_idx = (long long)n * C * D * H * W +
                                  (long long)c * D * H * W +
                                  (long long)d_in * H * W +
                                  (long long)h_in * W +
                                  w_in;
            my_val = input[input_idx];
        }

        // --- In-place Softmax Calculation ---
        // 1a. Find max value across channels using warp reduction
        float max_val = warp_reduce_max(my_val);
        max_val = __shfl_sync(0xFFFFFFFF, max_val, 0); // Broadcast max to all threads in warp

        // 1b. Compute exp(x - max)
        my_val = expf(my_val - max_val);

        // 1c. Sum exp values across channels using warp reduction
        float sum_exp = warp_reduce_sum(my_val);
        sum_exp = __shfl_sync(0xFFFFFFFF, sum_exp, 0); // Broadcast sum to all threads in warp

        // 1d. Compute final softmax value and store to shared memory
        float softmax_val = (sum_exp > 1e-6f) ? (my_val / sum_exp) : 0.0f;
        s_patch[c * pool_cube + p_idx] = softmax_val;
    }

    __syncthreads();

    // Step 2: Max pooling over the patch now stored in shared memory.
    // Each thread (representing a channel) finds its max value over the pool_cube locations.
    if (c < C) {
        float max_pooled_val = -1.0e20f;
        for (int p_idx = 0; p_idx < pool_cube; ++p_idx) {
            max_pooled_val = fmaxf(max_pooled_val, s_patch[c * pool_cube + p_idx]);
        }

        // Step 3: Write the final result to global memory
        long long output_idx = (long long)n * C * D_out * H_out * W_out +
                               (long long)c * D_out * H_out * W_out +
                               (long long)d_out * H_out * W_out +
                               (long long)h_out * W_out +
                               w_out;
        output[output_idx] = max_pooled_val;
    }
}

torch::Tensor softmax_pool3d_fused_cuda(torch::Tensor input, int pool_kernel_size) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5D");

    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto D = input.size(2);
    const auto H = input.size(3);
    const auto W = input.size(4);

    TORCH_CHECK(C <= 32, "Fused kernel only supports up to 32 channels (one warp)");

    // Calculate output dimensions, equivalent to PyTorch's MaxPool3d with default stride and ceil_mode=False
    const int D_out = D / pool_kernel_size;
    const int H_out = H / pool_kernel_size;
    const int W_out = W / pool_kernel_size;

    auto output = torch::empty({N, C, D_out, H_out, W_out}, input.options());

    dim3 grid(W_out, H_out, D_out * N);
    dim3 block(C);

    const int pool_cube = pool_kernel_size * pool_kernel_size * pool_kernel_size;
    size_t shared_mem_size = C * pool_cube * sizeof(float);

    softmax_pool3d_fused_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        D_out, H_out, W_out,
        pool_kernel_size
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

softmax_pool3d_fused_cpp_source = """
torch::Tensor softmax_pool3d_fused_cuda(torch::Tensor input, int pool_kernel_size);
"""

# Compile the inline CUDA code
softmax_pool3d_fused = load_inline(
    name="softmax_pool3d_fused",
    cpp_sources=softmax_pool3d_fused_cpp_source,
    cuda_sources=softmax_pool3d_fused_source,
    functions=["softmax_pool3d_fused_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies a fused Softmax + MaxPool operation,
    and performs a second max pooling operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        # The first pooling operation is fused into the custom CUDA kernel
        self.pool2 = nn.MaxPool3d(pool_kernel_size)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor after the full sequence of operations.
        """
        x = self.conv(x)
        # Apply the custom fused kernel for softmax(dim=1) + MaxPool3d
        x = softmax_pool3d_fused.softmax_pool3d_fused_cuda(x, self.pool_kernel_size)
        # Apply the second standard MaxPool3d
        x = self.pool2(x)
        return x