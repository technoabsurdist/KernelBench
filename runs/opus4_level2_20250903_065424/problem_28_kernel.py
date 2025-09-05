import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Linear + Instance Norm
linear_instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void linear_instance_norm_kernel(
    const float* input, const float* weight, const float* bias,
    const float* gamma, const float* beta,
    float* output, 
    int batch_size, int in_features, int out_features, float eps) {
    
    extern __shared__ float sdata[];
    
    int batch_idx = blockIdx.x;
    int feat_idx = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Compute linear transformation for this batch element
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    // First pass: compute output and statistics
    for (int i = feat_idx; i < out_features; i += blockDim.x) {
        float val = 0.0f;
        for (int j = 0; j < in_features; j++) {
            val += input[batch_idx * in_features + j] * weight[i * in_features + j];
        }
        if (bias != nullptr) {
            val += bias[i];
        }
        sdata[i] = val;
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    // Reduce to get mean and variance
    __shared__ float shared_sum[32];
    __shared__ float shared_sum_sq[32];
    
    if (threadIdx.x < 32) {
        shared_sum[threadIdx.x] = 0.0f;
        shared_sum_sq[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    atomicAdd(&shared_sum[threadIdx.x % 32], local_sum);
    atomicAdd(&shared_sum_sq[threadIdx.x % 32], local_sum_sq);
    __syncthreads();
    
    float mean = 0.0f;
    float var = 0.0f;
    if (threadIdx.x == 0) {
        for (int i = 0; i < 32 && i < blockDim.x; i++) {
            mean += shared_sum[i];
            var += shared_sum_sq[i];
        }
        mean /= out_features;
        var = var / out_features - mean * mean;
        shared_sum[0] = mean;
        shared_sum_sq[0] = rsqrtf(var + eps);
    }
    __syncthreads();
    
    mean = shared_sum[0];
    float inv_std = shared_sum_sq[0];
    
    // Apply normalization and write output
    for (int i = feat_idx; i < out_features; i += blockDim.x) {
        float normalized = (sdata[i] - mean) * inv_std;
        output[batch_idx * out_features + i] = gamma[i] * normalized + beta[i];
    }
}

torch::Tensor linear_instance_norm_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta, float eps) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    const int threads = 256;
    const int blocks = batch_size;
    size_t shared_mem = out_features * sizeof(float);
    
    linear_instance_norm_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_features, out_features, eps
    );
    
    return output;
}
"""

# Custom CUDA kernel for fused add and multiply
fused_add_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_add_mul_kernel(
    const float* x, const float* y, float* output, int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = x[idx] + y[idx];
        output[idx] = sum * y[idx];
    }
}

torch::Tensor fused_add_mul_cuda(torch::Tensor x, torch::Tensor y) {
    auto size = x.numel();
    auto output = torch::zeros_like(x);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_add_mul_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), 
        output.data_ptr<float>(), size
    );
    
    return output;
}
"""

linear_instance_norm_cpp_source = """
torch::Tensor linear_instance_norm_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta, float eps);
"""

fused_add_mul_cpp_source = """
torch::Tensor fused_add_mul_cuda(torch::Tensor x, torch::Tensor y);
"""

# Compile the inline CUDA code
linear_instance_norm = load_inline(
    name="linear_instance_norm",
    cpp_sources=linear_instance_norm_cpp_source,
    cuda_sources=linear_instance_norm_source,
    functions=["linear_instance_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

fused_add_mul = load_inline(
    name="fused_add_mul",
    cpp_sources=fused_add_mul_cpp_source,
    cuda_sources=fused_add_mul_source,
    functions=["fused_add_mul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernels for linear transformation, instance normalization, and fused operations.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        
        # Initialize weights and biases for linear layer
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (2.0 / in_features)**0.5)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Initialize parameters for instance norm
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        
        self.linear_instance_norm = linear_instance_norm
        self.fused_add_mul = fused_add_mul

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Input tensor of shape (batch_size, out_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Fused linear transformation and instance normalization
        x = self.linear_instance_norm.linear_instance_norm_cuda(
            x, self.weight, self.bias, self.gamma, self.beta, self.eps
        )
        
        # Fused addition and multiplication
        x = self.fused_add_mul.fused_add_mul_cuda(x, y)
        
        return x

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda(), torch.rand(batch_size, out_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]