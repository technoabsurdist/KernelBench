import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for a fused RMS Normalization
# This kernel combines square, mean, sqrt, and division operations into a single pass.
# It is specifically designed for 4D tensors where normalization occurs over the second dimension (dim=1).
rms_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// A device function for block-level reduction (sum) using shared memory.
// The shared memory buffer is provided by the caller.
__device__ inline float block_reduce_sum(float val, float* shared_data) {
    int tid = threadIdx.x;
    shared_data[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    return shared_data[0];
}

__global__ void rms_norm_fused_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const int N,          // Total number of vectors to normalize (B * D1 * D2)
    const int D,          // Dimension to normalize over (F)
    const int D1,         // Size of dim1
    const int D2,         // Size of dim2
    const float eps
) {
    // Dynamically allocated shared memory
    extern __shared__ float shared_mem[];

    // Each block processes one vector of size D.
    // A "vector" here corresponds to all features for a single (b, d1, d2) coordinate.
    const int i = blockIdx.x;
    if (i >= N) return;

    // Decode the linear block index `i` into (b, d1, d2) coordinates
    const int d2_idx = i % D2;
    const int d1_idx = (i / D2) % D1;
    const int b_idx = i / (D1 * D2);

    // Calculate the offset to the first element of the vector x[b, 0, d1, d2].
    // This assumes a standard contiguous tensor layout (B, F, D1, D2).
    const long long stride_f = (long long)D1 * D2;
    const long long offset = (long long)b_idx * D * stride_f + (long long)d1_idx * D2 + d2_idx;

    const float* x_ptr = x + offset;
    float* out_ptr = out + offset;

    // --- Pass 1: Calculate sum of squares ---
    // Each thread computes a partial sum of squares for its portion of the vector.
    float sum_sq = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        // Access elements along the feature dimension, which are not contiguous in memory.
        // The stride between them is `stride_f`.
        float val = x_ptr[(long long)j * stride_f];
        sum_sq += val * val;
    }

    // Reduce the partial sums from all threads in the block to get the total sum.
    sum_sq = block_reduce_sum(sum_sq, shared_mem);

    // --- Pass 2: Normalize ---
    // Thread 0 computes the reciprocal of the RMS and stores it in shared memory.
    // rsqrtf(x) = 1/sqrt(x) is used for performance.
    if (threadIdx.x == 0) {
        shared_mem[0] = rsqrtf(sum_sq / D + eps);
    }
    __syncthreads();

    // All threads load the reciprocal RMS value (rrms).
    const float rrms = shared_mem[0];

    // Each thread normalizes its portion of the vector.
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        const long long current_idx = (long long)j * stride_f;
        out_ptr[current_idx] = x_ptr[current_idx] * rrms;
    }
}

// C++ wrapper function that launches the CUDA kernel.
torch::Tensor rms_norm_cuda(torch::Tensor x, float eps) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor");

    const auto S = x.sizes();
    const int B = S[0];
    const int F = S[1];
    const int D1 = S[2];
    const int D2 = S[3];

    auto out = torch::empty_like(x);

    const int N = B * D1 * D2; // Total number of vectors to normalize
    const int D = F;           // Dimension to normalize over

    // Choose a block size that is a power of 2 for efficient reduction.
    // If the reduction dimension is small, adapt the block size.
    int block_size = 256;
    if (D < 256) {
        block_size = 1;
        while (block_size < D && block_size < 1024) block_size *= 2;
    }

    const int grid_size = N;
    const int shared_mem_size = block_size * sizeof(float);

    // Launch the kernel
    rms_norm_fused_kernel<<<grid_size, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D,
        D1,
        D2,
        eps
    );
    
    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# C++ source for the function signature, required by load_inline
rms_norm_cpp_source = "torch::Tensor rms_norm_cuda(torch::Tensor x, float eps);"

# JIT compile the custom CUDA kernel
rms_norm_fused = load_inline(
    name="rms_norm_fused",
    cpp_sources=rms_norm_cpp_source,
    cuda_sources=rms_norm_source,
    functions=["rms_norm_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that performs RMS Normalization using a custom fused CUDA kernel.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        # The custom kernel is fastest on a 4D contiguous tensor.
        # Ensure input is contiguous before passing to the kernel.
        return rms_norm_fused.rms_norm_cuda(x.contiguous(), self.eps)