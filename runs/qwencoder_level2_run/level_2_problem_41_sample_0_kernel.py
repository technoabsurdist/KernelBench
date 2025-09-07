import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + BatchNorm + GELU + ReLU
fused_gemm_bn_gelu_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

__global__ void fused_gemm_bn_gelu_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* bn_weight,
    const float* bn_bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    float eps
) {
    int batch_idx = blockIdx.x;
    int out_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (batch_idx >= batch_size || out_idx >= out_features) return;
    
    // Compute GEMM output for this output element
    float sum = 0.0f;
    for (int i = 0; i < in_features; ++i) {
        sum += input[batch_idx * in_features + i] * weight[out_idx * in_features + i];
    }
    sum += bias[out_idx];
    
    // BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    float bn_out = (sum - running_mean[out_idx]) / sqrtf(running_var[out_idx] + eps);
    bn_out = bn_out * bn_weight[out_idx] + bn_bias[out_idx];
    
    // GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608028654f;
    float gelu_out = bn_out * 0.5f * (1.0f + tanhf(sqrt_2_over_pi * (bn_out + 0.044715f * bn_out * bn_out * bn_out)));
    
    // ReLU: max(0, x)
    float relu_out = fmaxf(0.0f, gelu_out);
    
    output[batch_idx * out_features + out_idx] = relu_out;
}

torch::Tensor fused_gemm_bn_gelu_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int threads_per_block = 256;
    const int blocks_per_feature = (out_features + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(batch_size, blocks_per_feature);
    dim3 block(threads_per_block);
    
    fused_gemm_bn_gelu_relu_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        eps
    );
    
    return output;
}
"""

fused_gemm_bn_gelu_relu_cpp_source = """
torch::Tensor fused_gemm_bn_gelu_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
);
"""

# Compile the inline CUDA code for fused operation
fused_gemm_bn_gelu_relu = load_inline(
    name="fused_gemm_bn_gelu_relu",
    cpp_sources=fused_gemm_bn_gelu_relu_cpp_source,
    cuda_sources=fused_gemm_bn_gelu_relu_source,
    functions=["fused_gemm_bn_gelu_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused GEMM + BatchNorm + GELU + ReLU operation.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # GEMM parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # BatchNorm parameters
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        self.eps = 1e-5
        
        self.fused_op = fused_gemm_bn_gelu_relu

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return self.fused_op.fused_gemm_bn_gelu_relu_cuda(
            x,
            self.weight,
            self.bias,
            self.bn_weight,
            self.bn_bias,
            self.running_mean,
            self.running_var,
            self.eps
        )

batch_size = 16384
in_features = 4096
out_features = 4096

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]