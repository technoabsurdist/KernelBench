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

__global__ void compute_stats_kernel(
    const float* input,
    float* mean,
    float* var,
    const int batch_size,
    const int num_features,
    const int spatial_size
) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (feature_idx < num_features) {
        int elements_per_feature = batch_size * spatial_size;
        float sum = 0.0f;
        
        for (int i = 0; i < elements_per_feature; i++) {
            int idx = feature_idx * spatial_size + (i / spatial_size) * num_features * spatial_size + (i % spatial_size);
            sum += input[idx];
        }
        
        float mean_val = sum / elements_per_feature;
        mean[feature_idx] = mean_val;
        
        float sum_sq = 0.0f;
        for (int i = 0; i < elements_per_feature; i++) {
            int idx = feature_idx * spatial_size + (i / spatial_size) * num_features * spatial_size + (i % spatial_size);
            float diff = input[idx] - mean_val;
            sum_sq += diff * diff;
        }
        
        var[feature_idx] = sum_sq / elements_per_feature;
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
    
    const int block_size = 256;
    const int total_elements = batch_size * num_features * spatial_size;
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

std::tuple<torch::Tensor, torch::Tensor> compute_stats_cuda(torch::Tensor input) {
    const auto batch_size = input.size(0);
    const auto num_features = input.size(1);
    const auto spatial_size = input.size(2) * input.size(3);
    
    auto mean = torch::zeros({num_features}, input.options());
    auto var = torch::zeros({num_features}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (num_features + block_size - 1) / block_size;
    
    compute_stats_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        batch_size,
        num_features,
        spatial_size
    );
    
    return std::make_tuple(mean, var);
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

std::tuple<torch::Tensor, torch::Tensor> compute_stats_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code for Batch Normalization
batch_norm_module = load_inline(
    name="batch_norm",
    cpp_sources=batch_norm_cpp_source,
    cuda_sources=batch_norm_source,
    functions=["batch_norm_cuda", "compute_stats_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Batch Normalization with custom CUDA kernels.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        if self.training:
            # Compute batch statistics
            mean, var = batch_norm_module.compute_stats_cuda(x)
            
            # Update running statistics
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean, var = self.running_mean, self.running_var
            
        # Apply batch normalization
        return batch_norm_module.batch_norm_cuda(x, mean, var, self.weight, self.bias, self.eps)