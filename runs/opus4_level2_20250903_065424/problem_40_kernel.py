import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear + scaling + residual
fused_linear_scale_residual_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void add_bias_scale_kernel(float* output, const float* bias, float scale_factor, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * out_features;
    
    if (idx < total_size) {
        int feature_idx = idx % out_features;
        // Apply bias and scaling with residual (1 + scaling_factor)
        output[idx] = (output[idx] + bias[feature_idx]) * (1.0f + scale_factor);
    }
}

torch::Tensor fused_linear_scale_residual_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    // Allocate output tensor
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    // Setup cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Perform matrix multiplication: output = input @ weight.T
    // CUBLAS uses column-major, PyTorch uses row-major
    // We compute: output = input @ weight.T = (weight @ input.T).T
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, batch_size, in_features,
                &alpha,
                weight.data_ptr<float>(), in_features,
                input.data_ptr<float>(), in_features,
                &beta,
                output.data_ptr<float>(), out_features);
    
    // Apply bias and scaling with residual in a fused kernel
    const int block_size = 256;
    const int num_elements = batch_size * out_features;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    
    add_bias_scale_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        scaling_factor,
        batch_size,
        out_features
    );
    
    cublasDestroy(handle);
    
    return output;
}
"""

fused_linear_scale_residual_cpp_source = """
torch::Tensor fused_linear_scale_residual_cuda(
    torch::Tensor input,
    torch::Tensor weight, 
    torch::Tensor bias,
    float scaling_factor);
"""

# Compile the inline CUDA code
fused_linear_scale_residual = load_inline(
    name="fused_linear_scale_residual",
    cpp_sources=fused_linear_scale_residual_cpp_source,
    cuda_sources=fused_linear_scale_residual_source,
    functions=["fused_linear_scale_residual_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for linear + scaling + residual operations.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_linear_scale_residual
        
        # Initialize weights similar to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=torch.nn.init.calculate_gain('linear'))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Forward pass using fused CUDA kernel.
        """
        return self.fused_op.fused_linear_scale_residual_cuda(
            x, self.weight, self.bias, self.scaling_factor
        )

batch_size = 16384
in_features = 4096
out_features = 4096
scaling_factor = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]