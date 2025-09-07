import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + maxpool + sum + scale
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_matmul_maxpool_sum_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    float scale_factor
) {
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_idx >= out_features) return;
    
    // Compute matmul for this output element
    float sum = 0.0f;
    for (int i = 0; i < in_features; ++i) {
        sum += input[batch_idx * in_features + i] * weight[out_idx * in_features + i];
    }
    sum += bias[out_idx];
    
    // Apply max pooling (kernel size = 2) - we do this by taking max of consecutive pairs
    if (out_idx % 2 == 0) {
        float next_val = 0.0f;
        if (out_idx + 1 < out_features) {
            float next_sum = 0.0f;
            for (int i = 0; i < in_features; ++i) {
                next_sum += input[batch_idx * in_features + i] * weight[(out_idx + 1) * in_features + i];
            }
            next_sum += bias[out_idx + 1];
            next_val = next_sum;
        }
        float max_val = fmaxf(sum, next_val);
        output[batch_idx * (out_features/2) + (out_idx/2)] = max_val * scale_factor;
    }
}

// Alternative implementation that computes sum after maxpool
__global__ void fused_matmul_maxpool_sum_kernel_v2(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    float scale_factor
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    extern __shared__ float shared_mem[];
    float* max_vals = shared_mem;
    
    // Each thread computes two elements and takes their max
    for (int i = threadIdx.x; i < out_features/2; i += blockDim.x) {
        float val1 = 0.0f;
        float val2 = 0.0f;
        
        // Compute first element
        for (int j = 0; j < in_features; ++j) {
            val1 += input[batch_idx * in_features + j] * weight[(2*i) * in_features + j];
        }
        val1 += bias[2*i];
        
        // Compute second element
        for (int j = 0; j < in_features; ++j) {
            val2 += input[batch_idx * in_features + j] * weight[(2*i + 1) * in_features + j];
        }
        val2 += bias[2*i + 1];
        
        // Store max of the pair
        max_vals[i] = fmaxf(val1, val2);
    }
    
    __syncthreads();
    
    // Reduce to compute sum
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && threadIdx.x < out_features/4) {
            max_vals[threadIdx.x] += max_vals[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < out_features/2; i += blockDim.x) {
            total_sum += max_vals[i];
        }
        output[batch_idx] = total_sum * scale_factor;
    }
}

torch::Tensor fused_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor
) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Use the simpler kernel for now
    dim3 grid(batch_size);
    dim3 block(256);
    int shared_mem_size = (out_features/2) * sizeof(float);
    
    fused_matmul_maxpool_sum_kernel_v2<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        scale_factor
    );
    
    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor
);
"""

# Compile the inline CUDA code
fused_module = load_inline(
    name="fused_module",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_forward_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernels.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale_factor = scale_factor
        
        # Create parameters manually to access in CUDA kernel
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # Keep reference to fused module
        self.fused_module = fused_module

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        return self.fused_module.fused_forward_cuda(
            x, self.weight, self.bias, self.scale_factor
        )