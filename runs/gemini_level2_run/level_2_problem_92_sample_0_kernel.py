import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for the fused operations
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

// --- Kernel 1: Fused Tanh -> HardSwish -> Residual Add ---
// This kernel combines three element-wise operations into a single pass,
// reducing kernel launch overhead and memory bandwidth usage.

__global__ void fused_residual_activations_kernel(const float* x_norm, const float* x_conv, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float norm_val = x_norm[idx];
        
        // 1. Tanh activation
        float tanh_val = tanhf(norm_val);
        
        // 2. HardSwish activation: x * relu6(x + 3) / 6
        float temp = tanh_val + 3.0f;
        float hswish_val;
        if (temp <= 0.0f) {
            hswish_val = 0.0f;
        } else if (temp >= 6.0f) {
            hswish_val = tanh_val;
        } else {
            hswish_val = tanh_val * temp / 6.0f;
        }
        
        // 3. Residual Addition
        out[idx] = x_conv[idx] + hswish_val;
    }
}

torch::Tensor fused_residual_activations(torch::Tensor x_norm, torch::Tensor x_conv) {
    TORCH_CHECK(x_norm.is_cuda(), "x_norm must be a CUDA tensor");
    TORCH_CHECK(x_conv.is_cuda(), "x_conv must be a CUDA tensor");
    TORCH_CHECK(x_norm.sizes() == x_conv.sizes(), "Input tensors must have the same shape");
    
    auto out = torch::empty_like(x_norm);
    auto size = x_norm.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_residual_activations_kernel<<<num_blocks, block_size>>>(
        x_norm.data_ptr<float>(), 
        x_conv.data_ptr<float>(), 
        out.data_ptr<float>(), 
        size
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA error in fused_residual_activations: ", cudaGetErrorString(err));
    }
    
    return out;
}


// --- Kernel 2: Optimized LogSumExp Reduction (over channel dimension) ---
// This kernel implements a numerically stable LogSumExp reduction in a single pass.
// It uses shared memory for efficient parallel reduction, which is faster than
// PyTorch's multi-pass approach (max, sub, exp, sum, log, add).

__device__ void block_reduce_sum(volatile float* sdata) {
    int tid = threadIdx.x;
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
    }
}

__device__ void block_reduce_max(volatile float* sdata) {
    int tid = threadIdx.x;
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
    }
}

__global__ void logsumexp_channel_kernel(const float* in, float* out, int N, int C, int H, int W) {
    // Each block processes one pixel (n, h, w) across all C channels.
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    int n = blockIdx.x;
    int hw = blockIdx.y;
    int h = hw / W;
    int w = hw % W;

    // Step 1: Find max value in parallel
    float thread_max = -FLT_MAX;
    for (int c = tid; c < C; c += blockDim.x) {
        int in_idx = n * C * H * W + c * H * W + h * W + w;
        thread_max = fmaxf(thread_max, in[in_idx]);
    }
    sdata[tid] = thread_max;
    __syncthreads();
    if (blockDim.x > 1) block_reduce_max(sdata);
    __syncthreads();
    float max_val = sdata[0];

    // Step 2: Calculate sum of exps in parallel
    float thread_sum = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        int in_idx = n * C * H * W + c * H * W + h * W + w;
        thread_sum += expf(in[in_idx] - max_val);
    }
    sdata[tid] = thread_sum;
    __syncthreads();
    if (blockDim.x > 1) block_reduce_sum(sdata);
    __syncthreads();
    float sum_val = sdata[0];

    // Step 3: Final calculation and write result
    if (tid == 0) {
        int out_idx = n * H * W + h * W + w;
        if (sum_val > 0) {
            out[out_idx] = logf(sum_val) + max_val;
        } else {
            out[out_idx] = -INFINITY;
        }
    }
}

torch::Tensor logsumexp_channel(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor (NCHW)");

    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);

    auto out = torch::empty({N, 1, H, W}, input.options());
    auto out_view = out.view({N, H, W});

    const int block_size = 256;
    dim3 grid(N, H * W);
    dim3 block(block_size);
    size_t shared_mem_size = block_size * sizeof(float);

    logsumexp_channel_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        out_view.data_ptr<float>(),
        N, C, H, W
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA error in logsumexp_channel: ", cudaGetErrorString(err));
    }

    return out;
}
"""

# C++ source for function declarations
cpp_source = """
torch::Tensor fused_residual_activations(torch::Tensor x_norm, torch::Tensor x_conv);
torch::Tensor logsumexp_channel(torch::Tensor input);
"""

# JIT compile the inline CUDA code
custom_ops = load_inline(
    name="custom_fused_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_residual_activations", "logsumexp_channel"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Model that performs a convolution, applies Group Normalization, and then uses
    custom fused CUDA kernels for Tanh+HardSwish+ResidualAdd and LogSumExp.
    We replace the sequence of Tanh, HardSwish, residual add, and LogSumExp with
    two optimized and fused custom CUDA kernels. The Conv2d and GroupNorm layers
    are kept as they are highly optimized in cuDNN.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        # Standard PyTorch layers for operations that are already highly optimized
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)

    def forward(self, x):
        # 1. Convolution (using PyTorch's cuDNN-backed implementation)
        x_conv = self.conv(x)
        
        # 2. Group Normalization (using PyTorch's implementation)
        x_norm = self.group_norm(x_conv)
        
        # 3. Fused Tanh -> HardSwish -> Residual Addition (Custom CUDA Kernel)
        # This replaces three separate operations with a single kernel launch.
        x_res = custom_ops.fused_residual_activations(x_norm, x_conv)
        
        # 4. Optimized LogSumExp Reduction (Custom CUDA Kernel)
        # This replaces a multi-pass operation with an efficient single-pass reduction.
        x_logsumexp = custom_ops.logsumexp_channel(x_res)
        
        return x_logsumexp