import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + scale
fused_gemm_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

torch::Tensor fused_gemm_scale_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor scale) {
    const int batch_size = input.size(0);
    const int out_features = weight.size(0);
    const int in_features = weight.size(1);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    // Perform GEMM using cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, batch_size, in_features,
                &alpha,
                weight.data_ptr<float>(), in_features,
                input.data_ptr<float>(), in_features,
                &beta,
                output.data_ptr<float>(), out_features);
    
    // Add bias and scale in a single kernel
    const int total_elements = batch_size * out_features;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    auto kernel = [=] __device__ (int idx) {
        if (idx < total_elements) {
            int row = idx / out_features;
            int col = idx % out_features;
            float* out_ptr = output.data_ptr<float>();
            out_ptr[idx] = (out_ptr[idx] + bias.data_ptr<float>()[col]) * scale.data_ptr<float>()[col];
        }
    };
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    auto lambda_kernel = [=] __global__ () {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        kernel(idx);
    };
    
    void (*kernel_ptr)() = lambda_kernel;
    kernel_ptr<<<num_blocks, block_size, 0, stream>>>();
    
    cublasDestroy(handle);
    return output;
}

__global__ void bias_scale_kernel(float* output, const float* bias, const float* scale, 
                                  int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_features;
    
    if (idx < total) {
        int col = idx % out_features;
        output[idx] = (output[idx] + bias[col]) * scale[col];
    }
}

torch::Tensor fused_gemm_scale_cuda_v2(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor scale) {
    const int batch_size = input.size(0);
    const int out_features = weight.size(0);
    const int in_features = weight.size(1);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    // Perform GEMM
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, batch_size, in_features,
                &alpha,
                weight.data_ptr<float>(), in_features,
                input.data_ptr<float>(), in_features,
                &beta,
                output.data_ptr<float>(), out_features);
    
    // Apply bias and scale
    const int total_elements = batch_size * out_features;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    bias_scale_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        batch_size,
        out_features
    );
    
    return output;
}
"""

fused_gemm_scale_cpp_source = """
torch::Tensor fused_gemm_scale_cuda_v2(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor scale);
"""

# Define custom CUDA kernel for optimized batch normalization
batch_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void batch_norm_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  const float* __restrict__ running_mean,
                                  const float* __restrict__ running_var,
                                  const float* __restrict__ weight,
                                  const float* __restrict__ bias,
                                  float eps,
                                  int batch_size,
                                  int num_features) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_features;
    
    if (tid < total) {
        int feature_idx = tid % num_features;
        
        float mean = running_mean[feature_idx];
        float var = running_var[feature_idx];
        float gamma = weight[feature_idx];
        float beta = bias[feature_idx];
        
        float inv_std = rsqrtf(var + eps);
        output[tid] = gamma * (input[tid] - mean) * inv_std + beta;
    }
}

torch::Tensor batch_norm_cuda(torch::Tensor input,
                              torch::Tensor running_mean,
                              torch::Tensor running_var,
                              torch::Tensor weight,
                              torch::Tensor bias,
                              float eps) {
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    
    auto output = torch::zeros_like(input);
    
    const int total = batch_size * num_features;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    batch_norm_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        eps,
        batch_size,
        num_features
    );
    
    return output;
}
"""

batch_norm_cpp_source = """
torch::Tensor batch_norm_cuda(torch::Tensor input,
                              torch::Tensor running_mean,
                              torch::Tensor running_var,
                              torch::Tensor weight,
                              torch::Tensor bias,
                              float eps);
"""

# Compile the inline CUDA code
fused_gemm_scale = load_inline(
    name="fused_gemm_scale",
    cpp_sources=fused_gemm_scale_cpp_source,
    cuda_sources=fused_gemm_scale_source,
    functions=["fused_gemm_scale_cuda_v2"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

batch_norm_custom = load_inline(
    name="batch_norm_custom",
    cpp_sources=batch_norm_cpp_source,
    cuda_sources=batch_norm_source,
    functions=["batch_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernels for fused operations.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        
        # BatchNorm parameters
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        
        self.eps = eps
        self.momentum = momentum
        self.training = False
        
        self.fused_gemm_scale = fused_gemm_scale
        self.batch_norm_custom = batch_norm_custom

    def forward(self, x):
        # Fused GEMM + scale operation
        x = self.fused_gemm_scale.fused_gemm_scale_cuda_v2(
            x, self.weight, self.bias, self.scale
        )
        
        # Custom batch normalization (inference mode)
        x = self.batch_norm_custom.batch_norm_cuda(
            x, self.running_mean, self.running_var,
            self.bn_weight, self.bn_bias, self.eps
        )
        
        return x


batch_size = 16384
in_features = 4096
out_features = 4096
scale_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, scale_shape]