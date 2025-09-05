import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused BatchNorm + GELU + ReLU
fused_bn_gelu_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float gelu_func(float x) {
    const float sqrt_2_over_pi = 0.79788456080286535587989211986876f;
    const float a = 0.044715f;
    float x_cubed = x * x * x;
    float tanh_arg = sqrt_2_over_pi * (x + a * x_cubed);
    float tanh_val = tanhf(tanh_arg);
    return 0.5f * x * (1.0f + tanh_val);
}

__global__ void fused_bn_gelu_relu_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int num_features,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_features;
    
    if (idx < total_elements) {
        int feature_idx = idx % num_features;
        
        // BatchNorm transformation
        float mean = running_mean[feature_idx];
        float var = running_var[feature_idx];
        float scale = gamma[feature_idx];
        float bias = beta[feature_idx];
        
        float x_normalized = (input[idx] - mean) / sqrtf(var + eps);
        float bn_output = scale * x_normalized + bias;
        
        // GELU activation
        float gelu_output = gelu_func(bn_output);
        
        // ReLU activation
        output[idx] = fmaxf(0.0f, gelu_output);
    }
}

torch::Tensor fused_bn_gelu_relu_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
) {
    auto batch_size = input.size(0);
    auto num_features = input.size(1);
    auto output = torch::zeros_like(input);
    
    int total_elements = batch_size * num_features;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_bn_gelu_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_features,
        eps
    );
    
    return output;
}
"""

fused_bn_gelu_relu_cpp_source = """
torch::Tensor fused_bn_gelu_relu_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
);
"""

# Compile the inline CUDA code
fused_bn_gelu_relu = load_inline(
    name="fused_bn_gelu_relu",
    cpp_sources=fused_bn_gelu_relu_cpp_source,
    cuda_sources=fused_bn_gelu_relu_source,
    functions=["fused_bn_gelu_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused BatchNorm + GELU + ReLU kernel.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.fused_op = fused_bn_gelu_relu

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        
        # Use fused kernel for BatchNorm + GELU + ReLU
        x = self.fused_op.fused_bn_gelu_relu_cuda(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        
        return x

batch_size = 16384
in_features = 4096
out_features = 4096

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]