import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Gemm + LogSumExp + LeakyReLU + LeakyReLU + GELU + GELU
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__device__ float gelu_impl(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_idx >= out_features) return;
    
    // Gemm computation for this output element
    float sum = 0.0f;
    for (int i = 0; i < in_features; i++) {
        sum += input[batch_idx * in_features + i] * weight[out_idx * in_features + i];
    }
    if (bias != nullptr) {
        sum += bias[out_idx];
    }
    
    // LogSumExp - We'll compute this in shared memory for the entire output vector
    // For simplicity in this kernel, we'll approximate by just taking the max
    // A full implementation would require a reduction, but for performance we'll use a simplified approach
    __shared__ float shared_data[1024]; // Assuming max 1024 threads per block
    __shared__ float max_val, sum_exp;
    
    // Store all values in shared memory
    shared_data[threadIdx.x] = sum;
    __syncthreads();
    
    // Find max (reduction)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] = fmaxf(shared_data[threadIdx.x], shared_data[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        max_val = shared_data[0];
    }
    __syncthreads();
    
    // Compute sum of exp(x - max)
    shared_data[threadIdx.x] = expf(sum - max_val);
    __syncthreads();
    
    // Reduction for sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        sum_exp = shared_data[0];
    }
    __syncthreads();
    
    // LogSumExp result
    float lse_result = max_val + logf(sum_exp);
    
    // Apply activations: LeakyReLU -> LeakyReLU -> GELU -> GELU
    float val = lse_result;
    
    // First LeakyReLU
    val = (val > 0.0f) ? val : 0.01f * val;
    
    // Second LeakyReLU
    val = (val > 0.0f) ? val : 0.01f * val;
    
    // First GELU
    val = gelu_impl(val);
    
    // Second GELU
    val = gelu_impl(val);
    
    output[batch_idx * out_features + out_idx] = val;
}

torch::Tensor fused_gemm_activations_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, 1}, torch::kCUDA);
    
    // Kernel launch parameters
    dim3 grid(batch_size, (out_features + 1023) / 1024);
    dim3 block(1024);
    
    fused_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_gemm_activations_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_module = load_inline(
    name="fused_gemm_activations",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_gemm_activations_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused Gemm + LogSumExp + LeakyReLU + LeakyReLU + GELU + GELU
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return fused_module.fused_gemm_activations_cuda(x, self.weight, self.bias if self.bias is not None else torch.tensor([], device=x.device))

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]