import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Batch Normalization
batch_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void batch_norm_kernel(
    const float* input,
    const float* mean,
    const float* var,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int num_features,
    const int spatial_size,
    const float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_features * spatial_size;
    
    if (idx < total_elements) {
        int feature_idx = (idx / spatial_size) % num_features;
        float inv_std = rsqrtf(var[feature_idx] + eps);
        output[idx] = (input[idx] - mean[feature_idx]) * inv_std * weight[feature_idx] + bias[feature_idx];
    }
}

torch::Tensor batch_norm_cuda(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    const auto batch_size = input.size(0);
    const auto num_features = input.size(1);
    const auto spatial_size = input.size(2) * input.size(3);
    
    auto output = torch::zeros_like(input);
    
    const int total_elements = batch_size * num_features * spatial_size;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    batch_norm_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_features,
        spatial_size,
        eps
    );
    
    return output;
}
"""

batch_norm_cpp_source = """
torch::Tensor batch_norm_cuda(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
);
"""

# Compile the inline CUDA code for Batch Normalization
batch_norm = load_inline(
    name="batch_norm",
    cpp_sources=batch_norm_cpp_source,
    cuda_sources=batch_norm_source,
    functions=["batch_norm_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Batch Normalization with custom CUDA kernel.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.momentum = 0.1
        self.eps = 1e-5
        self.batch_norm = batch_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        if self.training:
            # Calculate batch statistics
            batch_mean = x.mean(dim=[0, 2, 3])
            batch_var = x.var(dim=[0, 2, 3], unbiased=False)
            
            # Update running statistics
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var
            
        return self.batch_norm.batch_norm_cuda(x, batch_mean, batch_var, self.weight, self.bias, self.eps)

batch_size = 64
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2, device='cuda')
    return [x]

def get_init_inputs():
    return [features]