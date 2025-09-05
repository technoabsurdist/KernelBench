import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Swish activation and scaling
fused_swish_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_swish_scale_kernel(float* x, const float* bias, float scaling_factor, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx < total_elements) {
        int feat_idx = idx % out_features;
        float val = x[idx];
        if (bias != nullptr) {
            val += bias[feat_idx];
        }
        // Swish activation: x * sigmoid(x)
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        val = val * sigmoid_val;
        // Apply scaling
        x[idx] = val * scaling_factor;
    }
}

torch::Tensor fused_swish_scale_cuda(torch::Tensor x, torch::Tensor bias, float scaling_factor) {
    auto batch_size = x.size(0);
    auto out_features = x.size(1);
    int total_elements = batch_size * out_features;
    
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    
    fused_swish_scale_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        bias_ptr,
        scaling_factor, 
        batch_size, 
        out_features
    );
    
    return x;
}
"""

fused_swish_scale_cpp_source = (
    "torch::Tensor fused_swish_scale_cuda(torch::Tensor x, torch::Tensor bias, float scaling_factor);"
)

# Define custom CUDA kernel for fused matmul + swish + scale
fused_matmul_swish_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void apply_bias_swish_scale_kernel(float* output, const float* bias, float scaling_factor, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx < total_elements) {
        int feat_idx = idx % out_features;
        float val = output[idx];
        if (bias != nullptr) {
            val += bias[feat_idx];
        }
        // Swish activation: x * sigmoid(x)
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        val = val * sigmoid_val;
        // Apply scaling
        output[idx] = val * scaling_factor;
    }
}

torch::Tensor fused_matmul_swish_scale_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scaling_factor) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    // Use cuBLAS for efficient matrix multiplication
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Perform matrix multiplication: output = input @ weight.T
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, batch_size, in_features,
                &alpha,
                weight.data_ptr<float>(), in_features,
                input.data_ptr<float>(), in_features,
                &beta,
                output.data_ptr<float>(), out_features);
    
    cublasDestroy(handle);
    
    // Apply bias, swish activation, and scaling in a single kernel
    int total_elements = batch_size * out_features;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    
    apply_bias_swish_scale_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(),
        bias_ptr,
        scaling_factor,
        batch_size,
        out_features
    );
    
    return output;
}
"""

fused_matmul_swish_scale_cpp_source = (
    "torch::Tensor fused_matmul_swish_scale_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scaling_factor);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=[fused_swish_scale_cpp_source, fused_matmul_swish_scale_cpp_source],
    cuda_sources=[fused_swish_scale_source, fused_matmul_swish_scale_source],
    functions=["fused_swish_scale_cuda", "fused_matmul_swish_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernels for matmul + swish + scaling.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops
        
        # Initialize weights similar to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=torch.nn.init.calculate_gain('linear'))

    def forward(self, x):
        x = x.cuda()
        self.weight = self.weight.cuda()
        self.bias = self.bias.cuda()
        return self.fused_ops.fused_matmul_swish_scale_cuda(x, self.weight, self.bias, self.scaling_factor)

batch_size = 128
in_features = 32768
out_features = 32768
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]