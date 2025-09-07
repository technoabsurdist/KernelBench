import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + swish + scale
fused_matmul_swish_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void swish_scale_kernel(const float* input, float* output, int size, float scaling_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = val * scaling_factor / (1.0f + expf(-val));
    }
}

torch::Tensor fused_matmul_swish_scale_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scaling_factor) {
    // Perform matrix multiplication using cuBLAS
    auto output = torch::mm(input, weight.t());
    if (bias.defined()) {
        output = output + bias;
    }
    
    // Apply Swish activation and scaling in-place
    auto total_elements = output.numel();
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    swish_scale_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), 
        output.data_ptr<float>(), 
        total_elements, 
        scaling_factor
    );
    
    return output;
}
"""

fused_matmul_swish_scale_cpp_source = """
torch::Tensor fused_matmul_swish_scale_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scaling_factor);
"""

# Compile the inline CUDA code for fused operation
fused_matmul_swish_scale = load_inline(
    name="fused_matmul_swish_scale",
    cpp_sources=fused_matmul_swish_scale_cpp_source,
    cuda_sources=fused_matmul_swish_scale_source,
    functions=["fused_matmul_swish_scale_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for matmul + Swish + scaling.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_matmul_swish_scale

    def forward(self, x):
        return self.fused_op.fused_matmul_swish_scale_cuda(
            x, self.weight, self.bias, self.scaling_factor
        )

batch_size = 128
in_features = 32768
out_features = 32768
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]