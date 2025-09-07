import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations: mean pooling + bias add + softmax + tanh + scale
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cmath>

__global__ void fused_kernel(
    const float* input,
    const float* bias,
    float* output,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const float scaling_factor
) {
    // Each block handles one spatial location across all channels
    int spatial_idx = blockIdx.x; // H*W locations
    int total_spatial = height * width;
    
    if (spatial_idx >= total_spatial) return;
    
    int h = spatial_idx / width;
    int w = spatial_idx % width;
    
    // Shared memory for reduction and softmax
    extern __shared__ float shared_mem[];
    float* shared_data = shared_mem;           // For storing channel data
    float* shared_sum = &shared_mem[channels]; // For reduction sum
    
    int tid = threadIdx.x;
    
    // Load data into shared memory and compute mean over depth
    // Input is [B, C, 1, H, W] after mean pooling
    float val = 0.0f;
    if (tid < channels) {
        int idx = blockIdx.y * (channels * total_spatial) + 
                  tid * total_spatial + 
                  h * width + w;
        val = input[idx] + bias[tid]; // Add bias
        shared_data[tid] = val;
    } else {
        shared_data[tid] = 0.0f;
    }
    
    __syncthreads();
    
    // Compute max for numerical stability in softmax
    float max_val = -INFINITY;
    for (int i = tid; i < channels; i += blockDim.x) {
        max_val = fmaxf(max_val, shared_data[i]);
    }
    
    // Reduction to find maximum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (tid + stride) < channels) {
            float other = shared_data[tid + stride];
            max_val = fmaxf(max_val, other);
        }
        __syncthreads();
    }
    
    if (tid == 0) shared_sum[0] = max_val;
    __syncthreads();
    max_val = shared_sum[0];
    
    // Compute exponentials and sum
    float sum_exp = 0.0f;
    if (tid < channels) {
        float exp_val = expf(shared_data[tid] - max_val);
        shared_data[tid] = exp_val;
        sum_exp = exp_val;
    } else {
        sum_exp = 0.0f;
    }
    
    __syncthreads();
    
    // Reduction to compute sum of exponentials
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = (tid + stride < channels) ? shared_data[tid + stride] : 0.0f;
            sum_exp += other;
        }
        __syncthreads();
    }
    
    if (tid == 0) shared_sum[0] = sum_exp;
    __syncthreads();
    sum_exp = shared_sum[0];
    
    // Apply softmax, tanh, and scaling
    if (tid < channels) {
        float softmax_val = shared_data[tid] / sum_exp;
        float tanh_val = tanhf(softmax_val);
        shared_data[tid] = tanh_val * scaling_factor;
    }
    
    __syncthreads();
    
    // Write result back to global memory
    if (tid < channels) {
        int out_idx = blockIdx.y * (channels * total_spatial) + 
                      tid * total_spatial + 
                      h * width + w;
        output[out_idx] = shared_data[tid];
    }
}

torch::Tensor fused_operation_cuda(
    torch::Tensor input, 
    torch::Tensor bias, 
    float scaling_factor
) {
    // Input shape: [B, C, 1, H, W] after mean pooling
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(3);
    auto width = input.size(4);
    
    auto output = torch::zeros_like(input);
    
    // Grid: (H*W, B)
    dim3 grid(width * height, batch_size);
    
    // Block: up to 1024 threads
    int block_size = (channels < 1024) ? ((channels + 31) / 32) * 32 : 1024;
    block_size = (block_size > 1024) ? 1024 : block_size;
    
    // Shared memory: channels for data + 1 for sum
    size_t shared_mem_size = (channels + 1) * sizeof(float);
    
    fused_kernel<<<grid, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        scaling_factor
    );
    
    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_operation_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_operation_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for fused operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(out_channels))  # Simplified bias
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)                            # (B, C, D, H, W)
        x = x.mean(dim=2, keepdim=True)                       # Mean pool over depth dim (D) -> (B, C, 1, H, W)
        x = self.fused_op.fused_operation_cuda(x, self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), self.scaling_factor)
        return x