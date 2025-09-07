import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void instance_norm_kernel(
    const float* input,
    float* output,
    const float* mean,
    const float* invstd,
    const float* weight,
    const float* bias,
    int batch_size,
    int features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * features) {
        int batch_idx = idx / features;
        int feature_idx = idx % features;
        float normalized = (input[idx] - mean[batch_idx]) * invstd[batch_idx];
        output[idx] = weight[feature_idx] * normalized + bias[feature_idx];
    }
}

__global__ void compute_stats_kernel(
    const float* input,
    float* mean,
    float* var,
    int batch_size,
    int features
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float val = input[batch_idx * features + i];
        sum += val;
        sum_sq += val * val;
    }
    
    __shared__ float shared_sum[256];
    __shared__ float shared_sum_sq[256];
    
    shared_sum[threadIdx.x] = sum;
    shared_sum_sq[threadIdx.x] = sum_sq;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            shared_sum_sq[threadIdx.x] += shared_sum_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        mean[batch_idx] = shared_sum[0] / features;
        var[batch_idx] = shared_sum_sq[0] / features - mean[batch_idx] * mean[batch_idx];
    }
}

__global__ void fused_add_mul_kernel(
    const float* x,
    const float* y,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum_val = x[idx] + y[idx];
        output[idx] = sum_val * y[idx];
    }
}

torch::Tensor fused_operations_cuda(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    auto batch_size = x.size(0);
    auto features = x.size(1);
    
    // Instance norm stats
    auto mean = torch::zeros({batch_size}, x.options());
    auto var = torch::zeros({batch_size}, x.options());
    
    // Compute mean and variance
    dim3 stats_grid(batch_size);
    dim3 stats_block(256);
    compute_stats_kernel<<<stats_grid, stats_block>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        batch_size,
        features
    );
    
    // Compute inverse standard deviation
    auto invstd = torch::rsqrt(var + eps);
    
    // Apply instance normalization
    auto normalized = torch::zeros_like(x);
    const int norm_block_size = 256;
    const int norm_num_blocks = (batch_size * features + norm_block_size - 1) / norm_block_size;
    instance_norm_kernel<<<norm_num_blocks, norm_block_size>>>(
        x.data_ptr<float>(),
        normalized.data_ptr<float>(),
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size,
        features
    );
    
    // Fused add and multiply
    auto result = torch::zeros_like(x);
    const int fuse_block_size = 256;
    const int fuse_num_blocks = (batch_size * features + fuse_block_size - 1) / fuse_block_size;
    fused_add_mul_kernel<<<fuse_num_blocks, fuse_block_size>>>(
        normalized.data_ptr<float>(),
        y.data_ptr<float>(),
        result.data_ptr<float>(),
        batch_size * features
    );
    
    return result;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_operations_cuda(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
);
"""

# Compile the inline CUDA code
fused_operations = load_inline(
    name="fused_operations",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_operations_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA operations.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.momentum = momentum
        
        # Linear weight and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # Instance norm parameters
        self.instance_norm_weight = nn.Parameter(torch.ones(out_features))
        self.instance_norm_bias = nn.Parameter(torch.zeros(out_features))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Input tensor of shape (batch_size, out_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Perform linear transformation
        x = torch.mm(x, self.weight.t()) + self.bias
        
        # Apply fused instance normalization, addition, and multiplication
        return fused_operations.fused_operations_cuda(
            x, y, 
            self.instance_norm_weight, 
            self.instance_norm_bias, 
            self.eps
        )