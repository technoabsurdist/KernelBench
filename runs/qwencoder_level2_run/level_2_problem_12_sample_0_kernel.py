import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + Multiply + LeakyReLU
fused_gemm_mul_lrelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_gemm_mul_lrelu_kernel(const float* input, float* output, 
                                           const float* weight, const float* bias,
                                           float multiplier, float negative_slope,
                                           int batch_size, int in_features, int out_features) {
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && out_idx < out_features) {
        float sum = 0.0f;
        
        // Compute GEMM for this output element
        for (int i = 0; i < in_features; ++i) {
            sum += input[batch_idx * in_features + i] * weight[out_idx * in_features + i];
        }
        
        // Add bias
        sum += bias[out_idx];
        
        // Multiply by scalar
        sum *= multiplier;
        
        // Apply LeakyReLU
        if (sum < 0) {
            sum *= negative_slope;
        }
        
        // Store result
        output[batch_idx * out_features + out_idx] = sum;
    }
}

torch::Tensor fused_gemm_mul_lrelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                       float multiplier, float negative_slope) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    dim3 grid_dim(batch_size, (out_features + 255) / 256);
    dim3 block_dim(256);
    
    fused_gemm_mul_lrelu_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        multiplier,
        negative_slope,
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

fused_gemm_mul_lrelu_cpp_source = """
torch::Tensor fused_gemm_mul_lrelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                       float multiplier, float negative_slope);
"""

# Compile the inline CUDA code for fused operation
fused_gemm_mul_lrelu = load_inline(
    name="fused_gemm_mul_lrelu",
    cpp_sources=fused_gemm_mul_lrelu_cpp_source,
    cuda_sources=fused_gemm_mul_lrelu_source,
    functions=["fused_gemm_mul_lrelu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused GEMM + Multiply + LeakyReLU operation.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # Reference to the fused operation
        self.fused_op = fused_gemm_mul_lrelu

    def forward(self, x):
        return self.fused_op.fused_gemm_mul_lrelu_cuda(
            x, self.weight, self.bias, self.multiplier, self.negative_slope
        )

batch_size = 1024
in_features  = 8192  
out_features = 8192
multiplier = 2.0
negative_slope = 0.1

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]