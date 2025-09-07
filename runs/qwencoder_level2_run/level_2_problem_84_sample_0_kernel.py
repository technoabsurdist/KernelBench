import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Gemm + BatchNorm + Scale + Softmax
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

__global__ void fused_gemm_bn_scale_softmax_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* bn_running_mean,
    const float* bn_running_var,
    const float* scale,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    float eps
) {
    int batch_idx = blockIdx.x;
    int out_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || out_idx >= out_features) return;
    
    // Shared memory for intermediate results
    extern __shared__ float shared_mem[];
    float* softmax_input = shared_mem;
    
    // Compute Gemm output for this thread
    float sum = 0.0f;
    for (int i = 0; i < in_features; ++i) {
        sum += input[batch_idx * in_features + i] * weight[out_idx * in_features + i];
    }
    sum += bias[out_idx];
    
    // Apply BatchNorm
    float bn_mean = bn_running_mean[out_idx];
    float bn_var = bn_running_var[out_idx];
    float normalized = (sum - bn_mean) / sqrtf(bn_var + eps);
    
    // Apply scale
    float scaled = normalized * scale[0];
    
    // Store in shared memory for softmax
    softmax_input[out_idx] = scaled;
    __syncthreads();
    
    // Compute softmax
    float max_val = -INFINITY;
    for (int i = 0; i < out_features; ++i) {
        max_val = fmaxf(max_val, softmax_input[i]);
    }
    __syncthreads();
    
    float sum_exp = 0.0f;
    for (int i = 0; i < out_features; ++i) {
        sum_exp += expf(softmax_input[i] - max_val);
    }
    __syncthreads();
    
    float result = expf(softmax_input[out_idx] - max_val) / sum_exp;
    output[batch_idx * out_features + out_idx] = result;
}

torch::Tensor fused_gemm_bn_scale_softmax_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    torch::Tensor scale,
    int batch_size,
    int in_features,
    int out_features,
    float eps
) {
    auto output = torch::zeros({batch_size, out_features}, torch::kCUDA);
    
    dim3 grid(batch_size);
    dim3 block(min(out_features, 1024));
    size_t shared_mem_size = out_features * sizeof(float);
    
    fused_gemm_bn_scale_softmax_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        bn_running_mean.data_ptr<float>(),
        bn_running_var.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        eps
    );
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_gemm_bn_scale_softmax_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    torch::Tensor scale,
    int batch_size,
    int in_features,
    int out_features,
    float eps
);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_gemm_bn_scale_softmax_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Gemm + BatchNorm + Scale + Softmax
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        
        # Gemm parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # BatchNorm parameters
        self.bn_running_mean = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.bn_running_var = nn.Parameter(torch.ones(out_features), requires_grad=False)
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        
        # Scale parameter
        self.scale = nn.Parameter(torch.ones(scale_shape))
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return fused_ops.fused_gemm_bn_scale_softmax_cuda(
            x,
            self.weight,
            self.bias,
            self.bn_running_mean,
            self.bn_running_var,
            self.scale,
            x.size(0),
            self.in_features,
            self.out_features,
            self.bn_eps
        )