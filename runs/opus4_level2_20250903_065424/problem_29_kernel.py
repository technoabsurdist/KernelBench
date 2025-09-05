import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused double Mish activation
double_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float mish_activation(float x) {
    // Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    float sp = logf(1.0f + expf(x));
    return x * tanhf(sp);
}

__global__ void double_mish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        // Apply first mish
        val = mish_activation(val);
        // Apply second mish
        val = mish_activation(val);
        output[idx] = val;
    }
}

torch::Tensor double_mish_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    double_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size
    );
    
    return output;
}
"""

double_mish_cpp_source = "torch::Tensor double_mish_cuda(torch::Tensor input);"

# Define the custom CUDA kernel for fused linear + mish
linear_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__device__ __forceinline__ float mish_activation(float x) {
    float sp = logf(1.0f + expf(x));
    return x * tanhf(sp);
}

__global__ void add_bias_mish_kernel(float* output, const float* bias, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * out_features;
    
    if (idx < total_size) {
        int feature_idx = idx % out_features;
        output[idx] = mish_activation(output[idx] + bias[feature_idx]);
    }
}

torch::Tensor linear_mish_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    // Perform matrix multiplication using cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f, beta = 0.0f;
    
    // Compute output = input @ weight.T
    // cuBLAS uses column-major, so we compute: output^T = weight @ input^T
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, batch_size, in_features,
                &alpha,
                weight.data_ptr<float>(), in_features,
                input.data_ptr<float>(), in_features,
                &beta,
                output.data_ptr<float>(), out_features);
    
    cublasDestroy(handle);
    
    // Add bias and apply mish activation
    const int block_size = 256;
    const int total_size = batch_size * out_features;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    add_bias_mish_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size,
        out_features
    );
    
    return output;
}
"""

linear_mish_cpp_source = "torch::Tensor linear_mish_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Compile the inline CUDA code
double_mish = load_inline(
    name="double_mish",
    cpp_sources=double_mish_cpp_source,
    cuda_sources=double_mish_source,
    functions=["double_mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

linear_mish = load_inline(
    name="linear_mish",
    cpp_sources=linear_mish_cpp_source,
    cuda_sources=linear_mish_source,
    functions=["linear_mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear_mish = linear_mish
        self.double_mish = double_mish

    def forward(self, x):
        # Fused linear + first mish
        x = self.linear_mish.linear_mish_cuda(x, self.linear.weight, self.linear.bias)
        # Apply second mish
        x = torch.nn.functional.mish(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]