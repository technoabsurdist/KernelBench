import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + GroupNorm + Swish + Multiply + Swish
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

__global__ void group_norm_kernel(const float* input, float* output, 
                                  const float* weight, const float* bias,
                                  int batch_size, int num_channels, int group_size, int elements_per_channel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_channels * elements_per_channel;
    
    if (idx < total_elements) {
        int channel = (idx / elements_per_channel) % num_channels;
        int group = channel / (num_channels / group_size);
        output[idx] = input[idx] * weight[group] + bias[group];
    }
}

__global__ void fused_activation_multiply_kernel(float* data, const float* multiply_weight, 
                                                 int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        // First Swish: x * sigmoid(x)
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        x = x * sigmoid_x;
        
        // Multiply with weight
        x = x * multiply_weight[idx];
        
        // Second Swish: x * sigmoid(x)
        sigmoid_x = 1.0f / (1.0f + expf(-x));
        data[idx] = x * sigmoid_x;
    }
}

torch::Tensor fused_gemm_gn_swish_multiply_swish(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor multiply_weight,
    torch::Tensor gemm_weight,
    torch::Tensor gemm_bias,
    int num_groups) {
    
    // GEMM operation
    auto gemm_output = torch::linear(input, gemm_weight, gemm_bias);
    
    // GroupNorm parameters
    auto batch_size = gemm_output.size(0);
    auto num_channels = gemm_output.size(1);
    if (gemm_output.dim() > 2) {
        // Reshape for GroupNorm if needed
        gemm_output = gemm_output.view({batch_size, num_channels, -1});
    }
    
    // Apply GroupNorm
    auto gn_output = torch::group_norm(gemm_output, num_groups, weight, bias, 1e-5);
    
    // Reshape back if needed
    if (gemm_output.dim() > 2) {
        gn_output = gn_output.view({batch_size, num_channels});
    }
    
    // Fused activation and multiply operations
    auto size = gn_output.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_activation_multiply_kernel<<<num_blocks, block_size>>>(
        gn_output.data_ptr<float>(), 
        multiply_weight.data_ptr<float>(), 
        size
    );
    
    return gn_output;
}
"""

fused_cpp_source = """
torch::Tensor fused_gemm_gn_swish_multiply_swish(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor multiply_weight,
    torch::Tensor gemm_weight,
    torch::Tensor gemm_bias,
    int num_groups);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_gemm_gn_swish_multiply_swish"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA operations.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        # GEMM parameters
        self.gemm_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.gemm_bias = nn.Parameter(torch.randn(out_features))
        
        # GroupNorm parameters
        self.gn_weight = nn.Parameter(torch.ones(num_groups))
        self.gn_bias = nn.Parameter(torch.zeros(num_groups))
        
        # Multiply weight
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        
        # Load fused operations
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_gemm_gn_swish_multiply_swish(
            x, 
            self.gn_weight, 
            self.gn_bias, 
            self.multiply_weight,
            self.gemm_weight,
            self.gemm_bias,
            self.num_groups
        )

batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 256
multiply_weight_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]