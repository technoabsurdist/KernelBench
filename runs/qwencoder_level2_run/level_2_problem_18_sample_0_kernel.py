import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void linear_sum_max_mean_logsumexp_logsumexp_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    // Shared memory for reduction operations
    extern __shared__ float sdata[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    // Phase 1: Linear transformation + Sum
    float sum_val = 0.0f;
    for (int i = tid; i < out_features; i += block_size) {
        float val = bias[i];
        for (int j = 0; j < in_features; j++) {
            val += input[batch_idx * in_features + j] * weight[i * in_features + j];
        }
        sum_val += val;
    }
    
    // Reduction for sum
    sdata[tid] = sum_val;
    __syncthreads();
    
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float reduced_sum = sdata[0];
    
    // Phase 2: Max (same value as sum in this case)
    float max_val = reduced_sum;
    
    // Phase 3: Mean (same value as sum in this case since we have only one value)
    float mean_val = reduced_sum;
    
    // Phase 4: First LogSumExp
    // logsumexp of a single value is just that value
    float lse1 = mean_val;
    
    // Phase 5: Second LogSumExp
    // logsumexp of a single value is just that value
    float lse2 = lse1;
    
    if (tid == 0) {
        output[batch_idx] = lse2;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    const int shared_mem_size = block_size * sizeof(float);
    
    linear_sum_max_mean_logsumexp_logsumexp_kernel<<<batch_size, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    // Reshape to (batch_size, 1)
    return output.view({batch_size, 1});
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code for fused operations
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA operations.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.fused_ops = fused_ops

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        return self.fused_ops.fused_ops_cuda(x, self.weight, self.bias)

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]