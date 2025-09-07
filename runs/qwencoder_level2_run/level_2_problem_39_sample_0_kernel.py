import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + Scale + BatchNorm
fused_gemm_scale_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

__global__ void scale_and_bn_kernel(
    const float* input,
    float* output,
    const float* scale,
    const float* running_mean,
    const float* running_var,
    const float* weight,
    const float* bias,
    float eps,
    int batch_size,
    int features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * features) {
        int feature_idx = idx % features;
        
        // Apply scale
        float scaled_val = input[idx] * scale[feature_idx];
        
        // Apply batch norm: (x - mean) / sqrt(var + eps) * weight + bias
        float normalized = (scaled_val - running_mean[feature_idx]) / sqrtf(running_var[feature_idx] + eps);
        output[idx] = normalized * weight[feature_idx] + bias[feature_idx];
    }
}

torch::Tensor fused_gemm_scale_bn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    float eps
) {
    // Get dimensions
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    // Create output tensor
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::zeros({batch_size, out_features}, options);
    
    // Perform GEMM using cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        out_features, batch_size, in_features,
        &alpha,
        weight.data_ptr<float>(), in_features,
        input.data_ptr<float>(), in_features,
        &beta,
        output.data_ptr<float>(), out_features
    );
    
    cublasDestroy(handle);
    
    // Launch kernel for scale and batch norm fusion
    const int block_size = 256;
    const int num_blocks = (batch_size * out_features + block_size - 1) / block_size;
    
    scale_and_bn_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(),
        output.data_ptr<float>(),
        scale.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        eps,
        batch_size,
        out_features
    );
    
    return output;
}
"""

fused_gemm_scale_bn_cpp_source = """
torch::Tensor fused_gemm_scale_bn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    float eps
);
"""

# Compile the inline CUDA code
fused_gemm_scale_bn = load_inline(
    name="fused_gemm_scale_bn",
    cpp_sources=fused_gemm_scale_bn_cpp_source,
    cuda_sources=fused_gemm_scale_bn_source,
    functions=["fused_gemm_scale_bn_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused GEMM + Scale + BatchNorm operation.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        
        # Linear layer parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Scale parameter
        self.scale = nn.Parameter(torch.randn(scale_shape))
        
        # BatchNorm parameters
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        
        # BatchNorm running statistics
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        self.momentum = momentum
        
        # Reference to the compiled CUDA function
        self.fused_op = fused_gemm_scale_bn

    def forward(self, x):
        return self.fused_op.fused_gemm_scale_bn_cuda(
            x,
            self.weight,
            self.bias,
            self.scale,
            self.running_mean,
            self.running_var,
            self.bn_weight,
            self.bn_bias,
            self.eps
        )

batch_size = 16384
in_features = 4096
out_features = 4096
scale_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, scale_shape]