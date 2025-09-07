import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + GELU + Softmax
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void gelu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        data[idx] = x * cdf;
    }
}

__global__ void softmax_kernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    int idx = row * cols + threadIdx.x;
    float max_val = -1e20f;
    
    // Find max value for numerical stability
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = input[row * cols + i];
        max_val = fmaxf(max_val, val);
    }
    
    __syncthreads();
    
    // Reduction to find maximum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float val = (threadIdx.x + stride < cols) ? input[row * cols + threadIdx.x + stride] : -1e20f;
            max_val = fmaxf(max_val, val);
        }
        __syncthreads();
    }
    
    float max_shared = max_val;
    __syncthreads();
    
    // Compute exponentials and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float exp_val = expf(input[row * cols + i] - max_shared);
        output[row * cols + i] = exp_val;
        sum += exp_val;
    }
    
    __syncthreads();
    
    // Reduction to find sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum += (threadIdx.x + stride < cols) ? output[row * cols + threadIdx.x + stride] : 0.0f;
        }
        __syncthreads();
    }
    
    float sum_shared = sum;
    __syncthreads();
    
    // Normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        output[row * cols + i] /= sum_shared;
    }
}

torch::Tensor fused_matmul_gelu_softmax_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    // Perform matrix multiplication
    auto output = torch::matmul(input, weight.t()) + bias;
    
    // Apply GELU activation
    auto total_elements = output.numel();
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    gelu_kernel<<<num_blocks, block_size>>>(output.data_ptr<float>(), total_elements);
    
    // Apply Softmax
    auto softmax_output = torch::zeros_like(output);
    dim3 softmax_grid(batch_size);
    dim3 softmax_block(min(out_features, 1024));
    softmax_kernel<<<softmax_grid, softmax_block>>>(
        output.data_ptr<float>(), 
        softmax_output.data_ptr<float>(), 
        batch_size, 
        out_features
    );
    
    return softmax_output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_matmul_gelu_softmax_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_kernel = load_inline(
    name="fused_matmul_gelu_softmax",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_matmul_gelu_softmax_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for matmul + GELU + Softmax
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.fused_op = fused_kernel

    def forward(self, x):
        return self.fused_op.fused_matmul_gelu_softmax_cuda(x, self.weight, self.bias)

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]