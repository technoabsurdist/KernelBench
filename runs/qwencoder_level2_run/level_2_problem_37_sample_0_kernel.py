import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + swish + bias + groupnorm
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

__global__ void swish_bias_gn_kernel(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    const float* gn_weight,
    const float* gn_bias,
    int batch_size,
    int in_features,
    int out_features,
    int num_groups,
    int group_size
) {
    int batch_idx = blockIdx.x;
    int feature_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || feature_idx >= out_features) return;
    
    // Compute linear transformation for this output element
    float sum = 0.0f;
    for (int k = 0; k < in_features; k++) {
        sum += input[batch_idx * in_features + k] * weight[feature_idx * in_features + k];
    }
    
    // Add bias
    sum += bias[feature_idx];
    
    // Apply Swish activation
    float sigmoid_val = 1.0f / (1.0f + expf(-sum));
    float activated = sum * sigmoid_val;
    
    // Store temporarily
    output[batch_idx * out_features + feature_idx] = activated;
    
    __syncthreads();
    
    // GroupNorm normalization (simplified version)
    if (threadIdx.x == 0) {
        int group_idx = blockIdx.y / (out_features / num_groups);
        float sum_val = 0.0f;
        float sum_sq = 0.0f;
        
        for (int i = group_idx * group_size; i < (group_idx + 1) * group_size; i++) {
            float val = output[batch_idx * out_features + i];
            sum_val += val;
            sum_sq += val * val;
        }
        
        float mean = sum_val / group_size;
        float var = sum_sq / group_size - mean * mean;
        float inv_std = rsqrtf(var + 1e-5);
        
        for (int i = group_idx * group_size; i < (group_idx + 1) * group_size; i++) {
            float val = output[batch_idx * out_features + i];
            output[batch_idx * out_features + i] = gn_weight[i] * (val - mean) * inv_std + gn_bias[i];
        }
    }
}

torch::Tensor fused_matmul_swish_bias_gn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int batch_size,
    int in_features,
    int out_features,
    int num_groups
) {
    auto output = torch::zeros({batch_size, out_features}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    dim3 block_size(256);
    dim3 grid_size(batch_size, (out_features + block_size.x - 1) / block_size.x);
    
    int group_size = out_features / num_groups;
    
    swish_bias_gn_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        num_groups,
        group_size
    );
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_matmul_swish_bias_gn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int batch_size,
    int in_features,
    int out_features,
    int num_groups
);
"""

# Compile the inline CUDA code
fused_kernel = load_inline(
    name="fused_kernel",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_matmul_swish_bias_gn_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for matmul + swish + bias + groupnorm
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.gn_weight = nn.Parameter(torch.randn(out_features))
        self.gn_bias = nn.Parameter(torch.randn(out_features))
        
        # Initialize weights properly
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.ones_(self.gn_weight)
        nn.init.zeros_(self.gn_bias)
        
        # Load custom kernel
        self.fused_op = fused_kernel

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return self.fused_op.fused_matmul_swish_bias_gn_cuda(
            x, self.weight, self.bias, self.gn_weight, self.gn_bias,
            x.size(0), self.in_features, self.out_features, self.num_groups
        )

batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]