import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations: softmax + subtract + swish + max
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cmath>

__global__ void fused_softmax_subtract_swish_max_kernel(
    const float* input,
    const float* subtract,
    float* output,
    int batch_size,
    int channels,
    int spatial_size
) {
    int batch_idx = blockIdx.x;
    int spatial_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || spatial_idx >= spatial_size) return;
    
    const float* input_batch = input + batch_idx * channels * spatial_size;
    float* output_batch = output + batch_idx * spatial_size;
    
    // Shared memory for softmax computation
    extern __shared__ float shared_mem[];
    float* shared_data = shared_mem;
    
    // Load data into shared memory
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        int idx = c * spatial_size + spatial_idx;
        shared_data[c] = (idx < channels * spatial_size) ? input_batch[idx] : 0.0f;
    }
    __syncthreads();
    
    // Subtract operation
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        shared_data[c] -= subtract[c];
    }
    __syncthreads();
    
    // Softmax computation
    float max_val = -INFINITY;
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        max_val = fmaxf(max_val, shared_data[c]);
    }
    
    // Reduction to find maximum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        float temp = 0.0f;
        if (threadIdx.x < stride && threadIdx.x + stride < channels) {
            temp = shared_data[threadIdx.x + stride];
        }
        __syncthreads();
        if (threadIdx.x < stride) {
            max_val = fmaxf(max_val, temp);
        }
        __syncthreads();
    }
    
    max_val = shared_data[0];
    for (int i = 1; i < min(blockDim.x, channels); i++) {
        max_val = fmaxf(max_val, shared_data[i]);
    }
    __syncthreads();
    
    // Compute exponentials and sum
    float sum_exp = 0.0f;
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        shared_data[c] = expf(shared_data[c] - max_val);
        sum_exp += shared_data[c];
    }
    __syncthreads();
    
    // Reduction to find sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        float temp = 0.0f;
        if (threadIdx.x < stride && threadIdx.x + stride < channels) {
            temp = shared_data[threadIdx.x + stride];
        }
        __syncthreads();
        if (threadIdx.x < stride) {
            sum_exp += temp;
        }
        __syncthreads();
    }
    
    sum_exp = 0.0f;
    for (int i = 0; i < min(blockDim.x, channels); i++) {
        sum_exp += shared_data[i];
    }
    __syncthreads();
    
    // Normalize and apply Swish, then find max across channels
    if (threadIdx.x == 0) {
        float max_result = -INFINITY;
        for (int c = 0; c < channels; c++) {
            shared_data[c] = (shared_data[c] / sum_exp) * (1.0f / (1.0f + expf(-(shared_data[c] / sum_exp)))) * (shared_data[c] / sum_exp);
            max_result = fmaxf(max_result, shared_data[c]);
        }
        output_batch[spatial_idx] = max_result;
    }
}

// Simplified version for better performance
__global__ void fused_operations_kernel(
    const float* input,
    const float* subtract,
    float* output,
    int batch_size,
    int channels,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * spatial_size;
    
    if (idx >= total_elements) return;
    
    int batch_idx = idx / spatial_size;
    int spatial_idx = idx % spatial_size;
    
    // Find max for softmax
    float max_val = -INFINITY;
    for (int c = 0; c < channels; c++) {
        int input_idx = ((batch_idx * channels) + c) * spatial_size + spatial_idx;
        float val = input[input_idx] - subtract[c];
        max_val = fmaxf(max_val, val);
    }
    
    // Compute softmax and apply swish
    float sum_exp = 0.0f;
    extern __shared__ float temp_storage[];
    for (int c = 0; c < channels; c++) {
        int input_idx = ((batch_idx * channels) + c) * spatial_size + spatial_idx;
        float val = input[input_idx] - subtract[c];
        float softmax_val = expf(val - max_val);
        sum_exp += softmax_val;
        temp_storage[c] = softmax_val;
    }
    
    // Find max across channels after swish
    float max_result = -INFINITY;
    for (int c = 0; c < channels; c++) {
        float softmax_normalized = temp_storage[c] / sum_exp;
        float swish_val = softmax_normalized / (1.0f + expf(-softmax_normalized));
        max_result = fmaxf(max_result, swish_val);
    }
    
    output[idx] = max_result;
}

torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor subtract) {
    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int channels = input_sizes[1];
    int depth = input_sizes[2];
    int height = input_sizes[3];
    int width = input_sizes[4];
    int spatial_size = depth * height * width;
    
    auto output = torch::zeros({batch_size, depth, height, width}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_spatial = batch_size * spatial_size;
    const int block_size = 256;
    const int grid_size = (total_spatial + block_size - 1) / block_size;
    
    size_t shared_mem_size = channels * sizeof(float);
    
    fused_operations_kernel<<<grid_size, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        subtract.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        spatial_size
    );
    
    return output;
}
"""

fused_cpp_source = (
    "torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor subtract);"
)

# Compile the inline CUDA code for fused operations
fused_operations = load_inline(
    name="fused_operations",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_operations_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernels for performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))
        self.fused_ops = fused_operations

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        # Use custom CUDA kernel for fused operations
        x = self.fused_ops.fused_operations_cuda(x, self.subtract)
        return x