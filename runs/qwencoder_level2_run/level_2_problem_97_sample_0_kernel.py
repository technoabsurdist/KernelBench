import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + batch norm + bias + div + swish
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

__global__ void fused_bn_bias_div_swish_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    const float* extra_bias,
    float* output,
    int batch_size,
    int features,
    float eps,
    float divide_value
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * features;
    
    if (idx < total_elements) {
        int batch_idx = idx / features;
        int feature_idx = idx % features;
        
        float val = input[idx];
        float mean = running_mean[feature_idx];
        float var = running_var[feature_idx];
        float w = weight[feature_idx];
        float b = bias[feature_idx];
        float extra_b = extra_bias[0];
        
        // Batch norm: (x - mean) / sqrt(var + eps) * weight + bias
        float normalized = (val - mean) * rsqrtf(var + eps);
        float bn_result = normalized * w + b;
        
        // Add extra bias
        float biased = bn_result + extra_b;
        
        // Divide
        float divided = biased / divide_value;
        
        // Swish activation: x * sigmoid(x)
        float sigmoid_val = 1.0f / (1.0f + expf(-divided));
        output[idx] = divided * sigmoid_val;
    }
}

torch::Tensor fused_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor extra_bias,
    float eps,
    float divide_value
) {
    auto batch_size = input.size(0);
    auto features = input.size(1);
    auto output = torch::zeros_like(input);
    
    const int block_size = 256;
    const int total_elements = batch_size * features;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_bn_bias_div_swish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        extra_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        features,
        eps,
        divide_value
    );
    
    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor extra_bias,
    float eps,
    float divide_value
);
"""

# Compile the inline CUDA code
fused_module = load_inline(
    name="fused_module",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_forward"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for matmul + batch norm + bias + div + swish
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.divide_value = divide_value
        
        # Linear layer parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.linear_bias = nn.Parameter(torch.randn(out_features))
        
        # Batch norm parameters
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        
        # Extra bias
        self.extra_bias = nn.Parameter(torch.randn(bias_shape))
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.linear_bias)

    def forward(self, x):
        # For simplicity, we're assuming the input is 2D (batch_size, in_features)
        # and we want to output (batch_size, out_features)
        
        # Perform fused operation
        return fused_module.fused_forward(
            torch.mm(x, self.weight.t()),
            self.bn_weight,
            self.bn_bias,
            self.running_mean,
            self.running_var,
            self.extra_bias,
            self.bn_eps,
            self.divide_value
        )

batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
bias_shape = (1,)
divide_value = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, bias_shape, divide_value]