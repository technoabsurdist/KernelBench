import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float gelu_impl(float x) {
    // Fast approximation of GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float c1 = 0.7978845608f; // sqrt(2/pi)
    const float c2 = 0.044715f;
    
    float x3 = x * x * x;
    float inner = c1 * (x + c2 * x3);
    float tanh_inner = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_inner);
}

__global__ void fused_scaling_hardtanh_gelu_kernel(
    const float* input,
    float* output,
    const float scaling_factor,
    const float hardtanh_min,
    const float hardtanh_max,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Load once from global memory
        float val = input[idx];
        
        // Apply scaling
        val = val * scaling_factor;
        
        // Apply hardtanh
        val = fminf(fmaxf(val, hardtanh_min), hardtanh_max);
        
        // Apply GELU
        val = gelu_impl(val);
        
        // Store once to global memory
        output[idx] = val;
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max
) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    fused_scaling_hardtanh_gelu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        hardtanh_min,
        hardtanh_max,
        size
    );
    
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max
);
"""

# Define custom GEMM with fused post-processing
gemm_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

__device__ __forceinline__ float gelu_impl(float x) {
    const float c1 = 0.7978845608f;
    const float c2 = 0.044715f;
    
    float x3 = x * x * x;
    float inner = c1 * (x + c2 * x3);
    float tanh_inner = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_inner);
}

__global__ void apply_fused_ops_kernel(
    float* data,
    const float scaling_factor,
    const float hardtanh_min,
    const float hardtanh_max,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float val = data[idx];
        val = val * scaling_factor;
        val = fminf(fmaxf(val, hardtanh_min), hardtanh_max);
        val = gelu_impl(val);
        data[idx] = val;
    }
}

torch::Tensor gemm_fused_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max
) {
    const auto batch_size = input.size(0);
    const auto in_features = input.size(1);
    const auto out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    // Use cuBLAS for GEMM
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Perform GEMM: output = input @ weight.T + bias
    cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        out_features,
        batch_size,
        in_features,
        &alpha,
        weight.data_ptr<float>(),
        in_features,
        input.data_ptr<float>(),
        in_features,
        &beta,
        output.data_ptr<float>(),
        out_features
    );
    
    // Add bias using cublas
    if (bias.numel() > 0) {
        const float one = 1.0f;
        auto ones = torch::ones({batch_size, 1}, input.options());
        cublasSger(
            handle,
            out_features,
            batch_size,
            &one,
            bias.data_ptr<float>(),
            1,
            ones.data_ptr<float>(),
            1,
            output.data_ptr<float>(),
            out_features
        );
    }
    
    cublasDestroy(handle);
    
    // Apply fused operations
    const int size = output.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    apply_fused_ops_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        scaling_factor,
        hardtanh_min,
        hardtanh_max,
        size
    );
    
    return output;
}
"""

gemm_fused_cpp_source = """
torch::Tensor gemm_fused_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max
);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

gemm_fused = load_inline(
    name="gemm_fused",
    cpp_sources=gemm_fused_cpp_source,
    cuda_sources=gemm_fused_source,
    functions=["gemm_fused_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernels for fused operations.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.gemm_fused = gemm_fused
        
    def forward(self, x):
        x = x.contiguous()
        weight = self.gemm.weight.contiguous()
        bias = self.gemm.bias.contiguous() if self.gemm.bias is not None else torch.empty(0, device=x.device)
        
        x = self.gemm_fused.gemm_fused_cuda(
            x,
            weight,
            bias,
            self.scaling_factor,
            self.hardtanh_min,
            self.hardtanh_max
        )
        return x

batch_size = 2048
in_features = 8192
out_features = 8192
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]