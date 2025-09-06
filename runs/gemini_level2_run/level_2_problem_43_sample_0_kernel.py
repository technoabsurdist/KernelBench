import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused logsumexp and relu
fused_lse_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

// Functors for reduction operations
struct MaxOp {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

struct SumOp {
    __device__ float operator()(float a, float b) const { return a + b; }
};

// Generic block-wide reduction using shared memory
template <typename T, typename Op>
__device__ T block_reduce(T val, Op op) {
    // Use a dynamic-sized shared memory array if compiling with a newer CUDA toolkit,
    // but for simplicity and compatibility, we use a static array.
    // This assumes blockDim.x <= 512.
    __shared__ T s_data[512];
    int tid = threadIdx.x;
    s_data[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = op(s_data[tid], s_data[tid + s]);
        }
        __syncthreads();
    }
    return s_data[0];
}

__global__ void logsumexp_relu_kernel(const float* input, float* output, int B, int C, int D, int H, int W) {
    // Each block computes one output element for a given (b, d, h, w)
    // The grid is (B, D, H*W)
    int b = blockIdx.x;
    int d = blockIdx.y;
    int spatial_idx = blockIdx.z;

    if (b >= B || d >= D || spatial_idx >= H * W) {
        return;
    }

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Stride for the channel dimension
    long long int channel_stride = (long long int)D * H * W;
    // Pointer to the first channel element for this spatial location
    const float* input_ptr = input + b * C * channel_stride + d * H * W + spatial_idx;

    // --- Phase 1: Find max value across channels for numerical stability ---
    float thread_max = -std::numeric_limits<float>::infinity();
    for (int c = tid; c < C; c += block_size) {
        thread_max = fmaxf(thread_max, input_ptr[c * channel_stride]);
    }
    float block_max = block_reduce(thread_max, MaxOp());

    // --- Phase 2: Sum of exponentials (x - max) ---
    float thread_sum = 0.0f;
    for (int c = tid; c < C; c += block_size) {
        thread_sum += expf(input_ptr[c * channel_stride] - block_max);
    }
    float block_sum = block_reduce(thread_sum, SumOp());

    // --- Phase 3: Final calculation (log(sum) + max) and ReLU, then write ---
    if (tid == 0) {
        float lse_val = block_max + logf(block_sum);
        // Apply ReLU
        float result = fmaxf(0.0f, lse_val);
        
        long long int output_idx = (long long int)b * D * H * W + (long long int)d * H * W + spatial_idx;
        output[output_idx] = result;
    }
}

torch::Tensor logsumexp_relu_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(torch::MemoryFormat::Contiguous), "Input tensor must be contiguous");
    TORCH_CHECK(input.dim() == 5, "Input must be a 5D tensor");

    const auto B = input.size(0);
    const auto C = input.size(1);
    const auto D = input.size(2);
    const auto H = input.size(3);
    const auto W = input.size(4);

    auto output = torch::empty({B, 1, D, H, W}, input.options());

    // A block size of 256 is a reasonable choice for reduction kernels
    const int block_size = 256;
    // The grid is sized to match the output spatial dimensions
    dim3 grid(B, D, H * W);

    logsumexp_relu_kernel<<<grid, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

fused_lse_relu_cpp_source = (
    "torch::Tensor logsumexp_relu_cuda(torch::Tensor input);"
)

# JIT compile the CUDA kernel
fused_lse_relu = load_inline(
    name="fused_lse_relu",
    cpp_sources=fused_lse_relu_cpp_source,
    cuda_sources=fused_lse_relu_source,
    functions=["logsumexp_relu_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, max pooling, and a fused log_sum_exp + ReLU.
    The log_sum_exp and ReLU operations are replaced by a single custom CUDA kernel
    to reduce kernel launch overhead and memory bandwidth.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        # Standard, highly optimized PyTorch operators
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Our custom fused operator
        self.fused_op = fused_lse_relu.logsumexp_relu_cuda

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, 1, depth', height', width')
        """
        x = self.conv(x)
        x = self.max_pool(x)
        
        # Apply the fused logsumexp(dim=1) and relu operation
        x = self.fused_op(x)
        
        return x