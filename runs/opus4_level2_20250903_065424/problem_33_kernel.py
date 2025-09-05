import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + scale
fused_gemm_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void scale_kernel(float* output, const float* scale, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx < total_elements) {
        int feature_idx = idx % out_features;
        output[idx] *= scale[feature_idx];
    }
}

torch::Tensor fused_gemm_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    // Perform GEMM using cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Compute output = input @ weight.T
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, batch_size, in_features,
                &alpha,
                weight.data_ptr<float>(), in_features,
                input.data_ptr<float>(), in_features,
                &beta,
                output.data_ptr<float>(), out_features);
    
    // Add bias
    const int threads = 256;
    const int blocks = (batch_size * out_features + threads - 1) / threads;
    
    // Apply scaling in the same kernel call
    scale_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        scale.data_ptr<float>(),
        batch_size,
        out_features);
    
    // Add bias using broadcasting
    output.add_(bias.unsqueeze(0));
    
    cublasDestroy(handle);
    
    return output;
}
"""

fused_gemm_scale_cpp_source = """
torch::Tensor fused_gemm_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale);
"""

# Define custom batch normalization kernel
custom_batchnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void compute_mean_var_kernel(
    const float* input, 
    float* mean, 
    float* var,
    int batch_size,
    int num_features) {
    
    int feature_idx = blockIdx.x;
    if (feature_idx >= num_features) return;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (int b = 0; b < batch_size; b++) {
        float val = input[b * num_features + feature_idx];
        sum += val;
        sum_sq += val * val;
    }
    
    __shared__ float shared_sum;
    __shared__ float shared_sum_sq;
    
    if (threadIdx.x == 0) {
        shared_sum = sum;
        shared_sum_sq = sum_sq;
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float m = shared_sum / batch_size;
        mean[feature_idx] = m;
        var[feature_idx] = shared_sum_sq / batch_size - m * m;
    }
}

__global__ void apply_batchnorm_kernel(
    const float* input,
    const float* mean,
    const float* var,
    const float* weight,
    const float* bias,
    float* output,
    float eps,
    int batch_size,
    int num_features) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_features;
    
    if (idx < total_elements) {
        int feature_idx = idx % num_features;
        float x = input[idx];
        float m = mean[feature_idx];
        float v = var[feature_idx];
        float w = weight[feature_idx];
        float b = bias[feature_idx];
        
        output[idx] = w * (x - m) / sqrtf(v + eps) + b;
    }
}

torch::Tensor custom_batchnorm_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    bool training,
    float momentum,
    float eps) {
    
    auto batch_size = input.size(0);
    auto num_features = input.size(1);
    
    auto output = torch::zeros_like(input);
    auto mean = torch::zeros({num_features}, input.options());
    auto var = torch::zeros({num_features}, input.options());
    
    if (training) {
        // Compute mean and variance
        dim3 blocks(num_features);
        dim3 threads(1);
        compute_mean_var_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            mean.data_ptr<float>(),
            var.data_ptr<float>(),
            batch_size,
            num_features);
        
        // Update running statistics
        running_mean.mul_(1 - momentum).add_(mean.mul(momentum));
        running_var.mul_(1 - momentum).add_(var.mul(momentum));
    } else {
        mean = running_mean;
        var = running_var;
    }
    
    // Apply batch normalization
    const int threads_per_block = 256;
    const int num_blocks = (batch_size * num_features + threads_per_block - 1) / threads_per_block;
    
    apply_batchnorm_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        eps,
        batch_size,
        num_features);
    
    return output;
}
"""

custom_batchnorm_cpp_source = """
torch::Tensor custom_batchnorm_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    bool training,
    float momentum,
    float eps);
"""

# Compile the inline CUDA code
fused_gemm_scale = load_inline(
    name="fused_gemm_scale",
    cpp_sources=fused_gemm_scale_cpp_source,
    cuda_sources=fused_gemm_scale_source,
    functions=["fused_gemm_scale_cuda"],
    extra_cuda_cflags=["-lcublas"],
    verbose=True,
)

custom_batchnorm = load_inline(
    name="custom_batchnorm",
    cpp_sources=custom_batchnorm_cpp_source,
    cuda_sources=custom_batchnorm_source,
    functions=["custom_batchnorm_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.momentum = momentum
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        
        # BatchNorm parameters
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        
        self.fused_gemm_scale = fused_gemm_scale
        self.custom_batchnorm = custom_batchnorm

    def forward(self, x):
        # Fused GEMM + scale
        x = self.fused_gemm_scale.fused_gemm_scale_cuda(
            x.cuda(), 
            self.weight.cuda(), 
            self.bias.cuda(), 
            self.scale.cuda()
        )
        
        # Custom batch normalization
        x = self.custom_batchnorm.custom_batchnorm_cuda(
            x,
            self.running_mean,
            self.running_var,
            self.bn_weight,
            self.bn_bias,
            self.training,
            self.momentum,
            self.eps
        )
        
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
scale_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scale_shape]