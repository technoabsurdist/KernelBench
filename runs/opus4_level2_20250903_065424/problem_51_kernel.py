import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused kernel for Linear + Subtract
fused_linear_subtract_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

torch::Tensor fused_linear_subtract_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor subtract_param) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    // Perform GEMM using cuBLAS
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, batch_size, in_features,
                &alpha,
                weight.data_ptr<float>(), in_features,
                input.data_ptr<float>(), in_features,
                &beta,
                output.data_ptr<float>(), out_features);
    
    // Add bias and subtract in one pass
    const int threads = 256;
    const int blocks = (batch_size * out_features + threads - 1) / threads;
    
    auto kernel = [=] __device__ (int idx) {
        if (idx < batch_size * out_features) {
            int feature_idx = idx % out_features;
            output.data_ptr<float>()[idx] = output.data_ptr<float>()[idx] + 
                                           bias.data_ptr<float>()[feature_idx] - 
                                           subtract_param.data_ptr<float>()[feature_idx];
        }
    };
    
    at::cuda::parallel_for(batch_size * out_features, kernel);
    
    return output;
}
"""

fused_linear_subtract_cpp_source = """
torch::Tensor fused_linear_subtract_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor subtract_param);
"""

# Fused kernel for GlobalAvgPool + LogSumExp + GELU
fused_pool_logsumexp_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_pool_logsumexp_gelu_kernel(const float* input, float* output, int batch_size, int features) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    extern __shared__ float shared_mem[];
    float* shared_sum = shared_mem;
    float* shared_max = &shared_mem[blockDim.x];
    
    int tid = threadIdx.x;
    
    // Compute mean (global average pool)
    float local_sum = 0.0f;
    for (int i = tid; i < features; i += blockDim.x) {
        local_sum += input[batch_idx * features + i];
    }
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduce sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    float mean_val = shared_sum[0] / features;
    
    // LogSumExp computation (on the single mean value)
    // Since we have keepdim=True and dim=1, and input is already reduced to single value per batch
    // LogSumExp of a single value is just the value itself
    float logsumexp_val = mean_val;
    
    // GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    float x = logsumexp_val;
    float x_cubed = x * x * x;
    float tanh_arg = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
    float tanh_val = tanhf(tanh_arg);
    float gelu_val = 0.5f * x * (1.0f + tanh_val);
    
    if (tid == 0) {
        output[batch_idx] = gelu_val;
    }
}

torch::Tensor fused_pool_logsumexp_gelu_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto features = input.size(1);
    auto output = torch::empty({batch_size, 1}, input.options());
    
    int threads = 256;
    int blocks = batch_size;
    size_t shared_mem_size = threads * 2 * sizeof(float);
    
    fused_pool_logsumexp_gelu_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        features
    );
    
    return output;
}
"""

fused_pool_logsumexp_gelu_cpp_source = """
torch::Tensor fused_pool_logsumexp_gelu_cuda(torch::Tensor input);
"""

# Broadcast residual add kernel
broadcast_residual_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_residual_add_kernel(const float* reduced, const float* original, float* output, 
                                               int batch_size, int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * features) {
        int batch_idx = idx / features;
        output[idx] = reduced[batch_idx] + original[idx];
    }
}

torch::Tensor broadcast_residual_add_cuda(torch::Tensor reduced, torch::Tensor original) {
    auto batch_size = original.size(0);
    auto features = original.size(1);
    auto output = torch::empty_like(original);
    
    const int threads = 256;
    const int blocks = (batch_size * features + threads - 1) / threads;
    
    broadcast_residual_add_kernel<<<blocks, threads>>>(
        reduced.data_ptr<float>(),
        original.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        features
    );
    
    return output;
}
"""

broadcast_residual_add_cpp_source = """
torch::Tensor broadcast_residual_add_cuda(torch::Tensor reduced, torch::Tensor original);
"""

# Load all custom CUDA kernels
fused_linear_subtract = load_inline(
    name="fused_linear_subtract",
    cpp_sources=fused_linear_subtract_cpp_source,
    cuda_sources=fused_linear_subtract_source,
    functions=["fused_linear_subtract_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

fused_pool_logsumexp_gelu = load_inline(
    name="fused_pool_logsumexp_gelu",
    cpp_sources=fused_pool_logsumexp_gelu_cpp_source,
    cuda_sources=fused_pool_logsumexp_gelu_source,
    functions=["fused_pool_logsumexp_gelu_cuda"],
    verbose=False,
)

broadcast_residual_add = load_inline(
    name="broadcast_residual_add",
    cpp_sources=broadcast_residual_add_cpp_source,
    cuda_sources=broadcast_residual_add_source,
    functions=["broadcast_residual_add_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else nn.Parameter(torch.zeros(out_features))
        self.subtract = nn.Parameter(torch.randn(out_features))
        
        self.fused_linear_subtract = fused_linear_subtract
        self.fused_pool_logsumexp_gelu = fused_pool_logsumexp_gelu
        self.broadcast_residual_add = broadcast_residual_add

    def forward(self, x):
        original_x = x
        
        # Fused Linear + Subtract
        x = self.fused_linear_subtract.fused_linear_subtract_cuda(
            x, self.weight, self.bias, self.subtract
        )
        
        # Fused GlobalAvgPool + LogSumExp + GELU
        x = self.fused_pool_logsumexp_gelu.fused_pool_logsumexp_gelu_cuda(x)
        
        # Broadcast Residual Add
        x = self.broadcast_residual_add.broadcast_residual_add_cuda(x, original_x)
        
        return x

batch_size = 2048
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]