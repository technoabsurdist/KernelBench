import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + Divide + Sum + Scale
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_gemm_divide_sum_scale_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int input_size,
    int hidden_size,
    float scaling_factor
) {
    int batch_idx = blockIdx.x;
    int hidden_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && hidden_idx < hidden_size) {
        float sum = 0.0f;
        
        // Compute dot product for this output element
        for (int i = 0; i < input_size; ++i) {
            sum += input[batch_idx * input_size + i] * weight[hidden_idx * input_size + i];
        }
        
        // Divide by 2, then accumulate for sum reduction
        sum = sum / 2.0f;
        
        // For sum reduction along hidden dimension, we need atomic add to a temporary buffer
        // But since we want sum over hidden dimension per batch, we do it differently:
        // Each thread computes one output element, then we reduce across hidden dimension
        
        // Since we're computing sum over hidden dimension for each batch:
        // We need a reduction step. For simplicity, we'll compute sum in a two-step process
        // but in this kernel we'll just compute the scaled value directly
        
        // Actually, the original operation is:
        // 1. GEMM: (B, H) = (B, I) @ (I, H)
        // 2. Divide by 2: (B, H) = (B, H) / 2
        // 3. Sum over H dim: (B, 1) = sum((B, H), dim=1)
        // 4. Scale: (B, 1) = (B, 1) * scaling_factor
        
        // So we need to accumulate across the hidden dimension
        __shared__ float shared_data[1024]; // Assuming max block size of 1024
        int tid = threadIdx.x;
        
        shared_data[tid] = sum;
        __syncthreads();
        
        // Reduction in shared memory
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_data[tid] += shared_data[tid + s];
            }
            __syncthreads();
        }
        
        // Write result
        if (tid == 0) {
            output[batch_idx] = shared_data[0] * scaling_factor;
        }
    }
}

// Alternative simpler kernel that computes the entire operation
__global__ void fused_operation_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int input_size,
    int hidden_size,
    float scaling_factor
) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        float sum = 0.0f;
        
        for (int h = 0; h < hidden_size; ++h) {
            float dot_product = 0.0f;
            
            // Vectorized load could be better here
            for (int i = 0; i < input_size; ++i) {
                dot_product += input[batch_idx * input_size + i] * weight[h * input_size + i];
            }
            
            // Divide by 2 and accumulate
            sum += dot_product / 2.0f;
        }
        
        // Scale and store
        output[batch_idx] = sum * scaling_factor;
    }
}

torch::Tensor fused_gemm_divide_sum_scale(torch::Tensor input, torch::Tensor weight, float scaling_factor) {
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto hidden_size = weight.size(0);
    
    auto output = torch::zeros({batch_size, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Launch configuration
    dim3 grid(batch_size);
    dim3 block(1);
    
    fused_operation_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size,
        scaling_factor
    );
    
    return output;
}
"""

fused_cpp_source = (
    "torch::Tensor fused_gemm_divide_sum_scale(torch::Tensor input, torch::Tensor weight, float scaling_factor);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_gemm_divide_sum_scale",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_gemm_divide_sum_scale"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        return self.fused_op.fused_gemm_divide_sum_scale(x, self.weight, self.scaling_factor)


batch_size   = 1024  
input_size   = 8192  
hidden_size  = 8192 
scaling_factor = 1.5

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]