import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + Swish + Divide + Clamp + Tanh + Clamp
fused_gemm_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_gemm_activation_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && out_idx < out_features) {
        float sum = 0.0f;
        
        // Compute GEMM for this output element
        for (int k = 0; k < in_features; ++k) {
            sum += input[batch_idx * in_features + k] * weight[out_idx * in_features + k];
        }
        
        // Add bias if present
        if (bias != nullptr) {
            sum += bias[out_idx];
        }
        
        // Apply Swish: x * sigmoid(x)
        float swish = sum * (1.0f / (1.0f + expf(-sum)));
        
        // Divide by 2.0
        float divided = swish / 2.0f;
        
        // First clamp between -1.0 and 1.0
        float clamped1 = fmaxf(-1.0f, fminf(1.0f, divided));
        
        // Apply tanh
        float tanh_val = tanhf(clamped1);
        
        // Second clamp between -1.0 and 1.0
        float clamped2 = fmaxf(-1.0f, fminf(1.0f, tanh_val));
        
        // Write output
        output[batch_idx * out_features + out_idx] = clamped2;
    }
}

torch::Tensor fused_gemm_activation_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Launch configuration
    dim3 block_size(256);
    dim3 grid_size(batch_size, (out_features + block_size.x - 1) / block_size.x);
    
    fused_gemm_activation_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

fused_gemm_activation_cpp_source = """
torch::Tensor fused_gemm_activation_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code for fused GEMM + activations
fused_gemm_activation = load_inline(
    name="fused_gemm_activation",
    cpp_sources=fused_gemm_activation_cpp_source,
    cuda_sources=fused_gemm_activation_source,
    functions=["fused_gemm_activation_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused GEMM and activation operations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.fused_op = fused_gemm_activation

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return self.fused_op.fused_gemm_activation_cuda(x, self.weight, self.bias if self.bias is not None else torch.tensor([]).cuda())

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]