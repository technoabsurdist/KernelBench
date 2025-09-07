import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + scaling + hardtanh + GELU
fused_gemm_act_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_activation_kernel(float* output, int size, float scaling_factor, float hardtanh_min, float hardtanh_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Scale
        float val = output[idx] * scaling_factor;
        
        // Hardtanh
        val = fmaxf(hardtanh_min, fminf(hardtanh_max, val));
        
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        float gelu_coeff = 0.7978845608028654f; // sqrt(2/pi)
        float x_cube = val * val * val;
        float tanh_arg = gelu_coeff * (val + 0.044715f * x_cube);
        // Fast tanh approximation
        float tanh_val = tanhf(tanh_arg);
        output[idx] = 0.5f * val * (1.0f + tanh_val);
    }
}

torch::Tensor fused_gemm_activation_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max) {
    
    // Perform GEMM using PyTorch's built-in functionality for correctness
    auto output = torch::mm(input, weight.t());
    if (bias.defined()) {
        output = output + bias;
    }
    
    // Launch custom kernel for activation functions
    auto size = output.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_activation_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), 
        size, 
        scaling_factor, 
        hardtanh_min, 
        hardtanh_max
    );
    
    return output;
}
"""

fused_gemm_act_cpp_source = """
torch::Tensor fused_gemm_activation_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max);
"""

# Compile the inline CUDA code
fused_gemm_act = load_inline(
    name="fused_gemm_act",
    cpp_sources=fused_gemm_act_cpp_source,
    cuda_sources=fused_gemm_act_source,
    functions=["fused_gemm_activation_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused GEMM and activation functions.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # Reference to compiled CUDA function
        self.fused_op = fused_gemm_act

    def forward(self, x):
        return self.fused_op.fused_gemm_activation_cuda(
            x, 
            self.weight, 
            self.bias,
            self.scaling_factor,
            self.hardtanh_min,
            self.hardtanh_max
        )

batch_size = 2048
in_features = 8192
out_features = 8192
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]