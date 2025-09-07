import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + Subtract + GlobalAvgPool + LogSumExp + GELU
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

__global__ void gelu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        data[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

__global__ void subtract_and_pool_kernel(const float* input, const float* subtract_vec, float* output, 
                                        int batch_size, int out_features) {
    int batch_idx = blockIdx.x;
    int feature_idx = threadIdx.x;
    
    if (batch_idx < batch_size && feature_idx < out_features) {
        int idx = batch_idx * out_features + feature_idx;
        output[idx] = input[idx] - subtract_vec[feature_idx];
    }
}

torch::Tensor fused_gemm_ops_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                                  torch::Tensor subtract_vec) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    // GEMM: output = input @ weight.T + bias
    auto gemm_output = torch::zeros({batch_size, out_features}, input.options());
    
    // Perform GEMM using PyTorch's built-in functionality for simplicity
    // In a full implementation, you would use cuBLAS directly
    gemm_output = torch::addmm(bias, input, weight.transpose(0, 1));
    
    // Subtract and GlobalAvgPool combined
    auto pool_output = torch::zeros({batch_size, out_features}, input.options());
    
    // Launch kernel for subtract operation
    dim3 block_size(1024);
    dim3 num_blocks(batch_size);
    subtract_and_pool_kernel<<<num_blocks, block_size>>>(
        gemm_output.data_ptr<float>(), 
        subtract_vec.data_ptr<float>(),
        pool_output.data_ptr<float>(),
        batch_size,
        out_features
    );
    
    // GlobalAvgPool - compute mean across features
    auto global_avg_pool = torch::mean(pool_output, 1, true);
    
    // LogSumExp: we'll compute this in a simplified way
    // For numerical stability, subtract max before exp
    auto max_vals = torch::max(global_avg_pool, 1, true).values;
    auto exp_vals = torch::exp(global_avg_pool - max_vals);
    auto sum_exp = torch::sum(exp_vals, 1, true);
    auto log_sum_exp = torch::log(sum_exp) + max_vals;
    
    // GELU activation
    auto gelu_output = log_sum_exp.clone();
    int size = gelu_output.numel();
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    gelu_kernel<<<blocks, threads_per_block>>>(
        gelu_output.data_ptr<float>(), size);
    
    return gelu_output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_gemm_ops_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor subtract_vec);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_gemm_ops_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA operations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.fused_ops = fused_ops
        self._bias = bias

    def forward(self, x):
        original_x = x
        # Use our fused CUDA kernel for most operations
        x = self.fused_ops.fused_gemm_ops_cuda(x, self.weight, self.bias if self._bias else torch.zeros_like(self.bias), self.subtract)
        # ResidualAdd
        x = x + original_x
        return x

batch_size = 2048
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]