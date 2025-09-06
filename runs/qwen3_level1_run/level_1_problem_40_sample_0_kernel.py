import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Layer Normalization
layernorm_cuda_source = """
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
    const int64_t h = blockIdx.x;
    const int64_t tid = threadIdx.x;
    const int64_t T = H;
    
    // Use shared memory for reduction
    __shared__ float sum_shared[256];
    __shared__ float sum_sq_shared[256];
    __shared__ float mean, invstd;
    
    const float* x = input + h * H;
    float* y = output + h * H;
    
    // Compute sum and sum of squares
    float sum = 0.0f, sum_sq = 0.0f;
    for (int64_t i = tid; i < H; i += blockDim.x) {
        float val = x[i];
        sum += val;
        sum_sq += val * val;
    }
    sum_shared[tid] = sum;
    sum_sq_shared[tid] = sum_sq;
    __syncthreads();
    
    // Reduction in shared memory
    for (int64_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_shared[tid] += sum_shared[tid + stride];
            sum_sq_shared[tid] += sum_sq_shared[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        mean = sum_shared[0] / H;
        float variance = sum_sq_shared[0] / H - mean * mean;
        invstd = rsqrtf(variance + eps);
    }
    __syncthreads();
    
    // Normalize and apply affine transformation
    for (int64_t i = tid; i < H; i += blockDim.x) {
        float val = (x[i] - mean) * invstd;
        y[i] = val * weight[i] + bias[i];
    }
}

torch::Tensor layernorm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    auto output = torch::empty_like(input);
    
    const int64_t N = input.size(0);  // batch size
    const int64_t C = input.size(1);  // features
    const int64_t H = input.numel() / (N * C);  // elements per channel
    
    const int threads = 256;
    const int blocks = N * C;
    
    layernorm_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        N,
        H,
        eps
    );
    
    return output;
}
"""

layernorm_cpp_source = """
torch::Tensor layernorm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
);
"""

# Compile the inline CUDA code for Layer Normalization
layernorm = load_inline(
    name="layernorm",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_cuda_source,
    functions=["layernorm_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization with custom CUDA kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5
        self.layernorm = layernorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Reshape input to 3D for our kernel (N, C, L)
        original_shape = x.shape
        if x.dim() > 3:
            x = x.view(-1, original_shape[-3], original_shape[-2] * original_shape[-1])
        elif x.dim() == 3:
            x = x.view(x.size(0), x.size(1), -1)
        else:
            x = x.view(-1, 1, x.numel() // x.size(0))
            
        # Apply custom layernorm
        weight = self.weight.view(-1)
        bias = self.bias.view(-1)
        output = self.layernorm.layernorm_cuda(x, weight, bias, self.eps)
        
        # Reshape back to original shape
        return output.view(original_shape)