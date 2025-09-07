import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Instance Normalization
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void instance_norm_kernel(
    const float* input,
    float* output,
    const float* mean,
    const float* invstd,
    const float* weight,
    const float* bias,
    int batch_size,
    int num_features,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_features * spatial_size;
    
    if (idx < total_elements) {
        int batch_idx = idx / (num_features * spatial_size);
        int feature_idx = (idx / spatial_size) % num_features;
        int spatial_idx = idx % spatial_size;
        
        float normalized = (input[idx] - mean[batch_idx * num_features + feature_idx]) * 
                          invstd[batch_idx * num_features + feature_idx];
        
        if (weight != nullptr && bias != nullptr) {
            output[idx] = normalized * weight[feature_idx] + bias[feature_idx];
        } else {
            output[idx] = normalized;
        }
    }
}

__global__ void compute_stats_kernel(
    const float* input,
    float* mean,
    float* var,
    int batch_size,
    int num_features,
    int spatial_size
) {
    int batch_feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_batch_features = batch_size * num_features;
    
    if (batch_feature_idx < total_batch_features) {
        int batch_idx = batch_feature_idx / num_features;
        int feature_idx = batch_feature_idx % num_features;
        
        int base_idx = batch_idx * num_features * spatial_size + feature_idx * spatial_size;
        
        // Compute mean
        float sum = 0.0f;
        for (int i = 0; i < spatial_size; i++) {
            sum += input[base_idx + i];
        }
        float m = sum / spatial_size;
        mean[batch_feature_idx] = m;
        
        // Compute variance
        float sum_sq_diff = 0.0f;
        for (int i = 0; i < spatial_size; i++) {
            float diff = input[base_idx + i] - m;
            sum_sq_diff += diff * diff;
        }
        var[batch_feature_idx] = sum_sq_diff / spatial_size;
    }
}

torch::Tensor instance_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto batch_size = input.size(0);
    auto num_features = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto spatial_size = height * width;
    
    auto mean = torch::empty({batch_size, num_features}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto var = torch::empty({batch_size, num_features}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto invstd = torch::empty({batch_size, num_features}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto output = torch::empty_like(input);
    
    // Compute statistics
    const int stats_block_size = 256;
    const int stats_num_blocks = (batch_size * num_features + stats_block_size - 1) / stats_block_size;
    
    compute_stats_kernel<<<stats_num_blocks, stats_block_size>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        batch_size,
        num_features,
        spatial_size
    );
    
    // Compute inverse standard deviation
    invstd = torch::rsqrt(var + eps);
    
    // Normalize
    const int norm_block_size = 256;
    const int total_elements = batch_size * num_features * spatial_size;
    const int norm_num_blocks = (total_elements + norm_block_size - 1) / norm_block_size;
    
    instance_norm_kernel<<<norm_num_blocks, norm_block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        weight.defined() ? weight.data_ptr<float>() : nullptr,
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        batch_size,
        num_features,
        spatial_size
    );
    
    return output;
}
"""

instance_norm_cpp_source = """
torch::Tensor instance_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
);
"""

# Compile the inline CUDA code for Instance Normalization
instance_norm = load_inline(
    name="instance_norm",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Instance Normalization with custom CUDA kernels.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5
        self.instance_norm = instance_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return self.instance_norm.instance_norm_cuda(x, self.weight, self.bias, self.eps)