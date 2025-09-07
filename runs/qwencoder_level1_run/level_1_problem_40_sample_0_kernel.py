import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Layer Normalization
layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void layernorm_kernel(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    const int64_t N,
    const int64_t H,
    const float eps
) {
    const int batch_idx = blockIdx.x;
    const int feature_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int H_size = H;
    const int stride = batch_idx * H_size + feature_idx * H;

    // Shared memory for reduction
    extern __shared__ float shared_mem[];
    float* shared_data = shared_mem;

    // Load data into shared memory
    float val = 0.0f;
    if (tid < H) {
        val = input[stride + tid];
    }
    shared_data[tid] = val;
    __syncthreads();

    // Compute mean
    for (int stride_len = H/2; stride_len > 0; stride_len >>= 1) {
        if (tid < stride_len && tid + stride_len < H) {
            shared_data[tid] += shared_data[tid + stride_len];
        }
        __syncthreads();
    }
    float mean = shared_data[0] / H;
    __syncthreads();

    // Compute variance
    float diff = 0.0f;
    if (tid < H) {
        diff = input[stride + tid] - mean;
        shared_data[tid] = diff * diff;
    } else {
        shared_data[tid] = 0.0f;
    }
    __syncthreads();

    for (int stride_len = H/2; stride_len > 0; stride_len >>= 1) {
        if (tid < stride_len && tid + stride_len < H) {
            shared_data[tid] += shared_data[tid + stride_len];
        }
        __syncthreads();
    }
    float var = shared_data[0] / H;
    float inv_std = rsqrtf(var + eps);
    __syncthreads();

    // Normalize and apply affine transformation
    if (tid < H) {
        float normalized = (input[stride + tid] - mean) * inv_std;
        output[stride + tid] = normalized * weight[tid] + bias[tid];
    }
}

__global__ void layernorm_simple_kernel(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    const int64_t N,
    const int64_t H,
    const float eps
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * H) return;

    // Compute batch and feature indices
    const int batch_idx = idx / H;
    const int hid_idx = idx % H;

    // Calculate mean
    float sum = 0.0f;
    for (int i = 0; i < H; ++i) {
        sum += input[batch_idx * H + i];
    }
    float mean = sum / H;

    // Calculate variance
    float sum_sq = 0.0f;
    for (int i = 0; i < H; ++i) {
        float diff = input[batch_idx * H + i] - mean;
        sum_sq += diff * diff;
    }
    float var = sum_sq / H;
    float inv_std = rsqrtf(var + eps);

    // Normalize and apply affine transformation
    float normalized = (input[idx] - mean) * inv_std;
    output[idx] = normalized * weight[hid_idx] + bias[hid_idx];
}

torch::Tensor layernorm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t normalized_shape_size,
    double eps
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    auto output = torch::zeros_like(input);
    const int64_t N = input.numel() / normalized_shape_size;
    const int64_t H = normalized_shape_size;

    if (H <= 1024 && H % 32 == 0) {
        // Use optimized kernel with shared memory
        const dim3 block_size(1024);
        const dim3 grid_size(N, 1, 1);
        const int shared_mem_size = 1024 * sizeof(float);
        
        layernorm_kernel<<<grid_size, block_size, shared_mem_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            N,
            H,
            static_cast<float>(eps)
        );
    } else {
        // Use simple kernel
        const int block_size = 256;
        const int grid_size = (N * H + block_size - 1) / block_size;
        
        layernorm_simple_kernel<<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            N,
            H,
            static_cast<float>(eps)
        );
    }

    return output;
}
"""

layernorm_cpp_source = """
torch::Tensor layernorm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t normalized_shape_size,
    double eps
);
"""

# Compile the inline CUDA code for Layer Normalization
layernorm = load_inline(
    name="layernorm",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_source,
    functions=["layernorm_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization with custom CUDA kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer with custom CUDA implementation.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Extract weight and bias from the internal LayerNorm module
        weight = self.ln.weight
        bias = self.ln.bias
        eps = self.ln.eps
        
        # Calculate the size of the normalized dimensions
        normalized_shape_size = 1
        for dim in self.normalized_shape:
            normalized_shape_size *= dim
            
        # Call custom CUDA kernel
        return layernorm.layernorm_cuda(x, weight, bias, normalized_shape_size, eps)

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [(features, dim1, dim2)]