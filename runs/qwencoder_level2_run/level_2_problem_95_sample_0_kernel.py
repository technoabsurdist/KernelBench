import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + bias + swish + tanh + gelu + hardtanh
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_activation_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        
        // Swish: x * sigmoid(x)
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        val = val * sigmoid_val;
        
        // Tanh
        val = tanhf(val);
        
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float gelu_val = 0.5f * val * (1.0f + tanhf(0.7978845608028654f * (val + 0.044715f * val * val * val)));
        val = gelu_val;
        
        // Hardtanh: clamp to [-1, 1]
        val = fmaxf(-1.0f, fminf(1.0f, val));
        
        x[idx] = val;
    }
}

torch::Tensor fused_matmul_bias_activation_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // Perform matrix multiplication: input @ weight.T + bias
    auto output = torch::matmul(input, weight.transpose(0, 1)) + bias;
    
    // Apply fused activations
    auto size = output.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_activation_kernel<<<num_blocks, block_size>>>(output.data_ptr<float>(), size);
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_matmul_bias_activation_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_matmul_bias_activation_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernels.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_matmul_bias_activation_cuda(x, self.weight, self.bias)

batch_size = 1024
in_features = 8192
out_features = 8192
add_value_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, add_value_shape]