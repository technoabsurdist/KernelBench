import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Atomic.cuh>

__global__ void group_norm_mean_kernel(
    const float* input,
    float* mean,
    int batch_size,
    int num_features,
    int spatial_size,
    int group_size) {
    
    int batch_idx = blockIdx.x;
    int group_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || group_idx >= (num_features / group_size)) return;
    
    // Shared memory for reduction
    extern __shared__ float shared_data[];
    
    float sum = 0.0f;
    int feature_start = group_idx * group_size;
    int feature_end = feature_start + group_size;
    int total_elements = group_size * spatial_size;
    
    // Each thread processes multiple elements
    for (int f = feature_start; f < feature_end; f++) {
        for (int s = tid; s < spatial_size; s += blockDim.x) {
            int idx = batch_idx * (num_features * spatial_size) + f * spatial_size + s;
            sum += input[idx];
        }
    }
    
    shared_data[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        mean[batch_idx * (num_features / group_size) + group_idx] = shared_data[0] / total_elements;
    }
}

__global__ void group_norm_var_kernel(
    const float* input,
    const float* mean,
    float* var,
    int batch_size,
    int num_features,
    int spatial_size,
    int group_size) {
    
    int batch_idx = blockIdx.x;
    int group_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || group_idx >= (num_features / group_size)) return;
    
    // Shared memory for reduction
    extern __shared__ float shared_data[];
    
    float sum = 0.0f;
    int feature_start = group_idx * group_size;
    int feature_end = feature_start + group_size;
    int total_elements = group_size * spatial_size;
    float group_mean = mean[batch_idx * (num_features / group_size) + group_idx];
    
    // Each thread processes multiple elements
    for (int f = feature_start; f < feature_end; f++) {
        for (int s = tid; s < spatial_size; s += blockDim.x) {
            int idx = batch_idx * (num_features * spatial_size) + f * spatial_size + s;
            float diff = input[idx] - group_mean;
            sum += diff * diff;
        }
    }
    
    shared_data[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        var[batch_idx * (num_features / group_size) + group_idx] = shared_data[0] / total_elements;
    }
}

__global__ void group_norm_normalize_kernel(
    const float* input,
    const float* mean,
    const float* var,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int num_features,
    int spatial_size,
    int group_size,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_features * spatial_size;
    
    if (idx >= total_elements) return;
    
    int batch_idx = idx / (num_features * spatial_size);
    int feature_idx = (idx % (num_features * spatial_size)) / spatial_size;
    int group_idx = feature_idx / group_size;
    
    float group_mean = mean[batch_idx * (num_features / group_size) + group_idx];
    float group_var = var[batch_idx * (num_features / group_size) + group_idx];
    float inv_std = rsqrtf(group_var + eps);
    
    float normalized = (input[idx] - group_mean) * inv_std;
    
    if (weight != nullptr && bias != nullptr) {
        normalized = normalized * weight[feature_idx] + bias[feature_idx];
    }
    
    output[idx] = normalized;
}

torch::Tensor group_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    double eps) {
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto batch_size = input.size(0);
    auto num_features = input.size(1);
    auto spatial_size = input.numel() / (batch_size * num_features);
    auto group_size = num_features / num_groups;
    
    // Allocate temporary tensors for mean and variance
    auto mean = torch::zeros({batch_size, num_groups}, input.options());
    auto var = torch::zeros({batch_size, num_groups}, input.options());
    auto output = torch::zeros_like(input);
    
    // Launch kernel to compute mean
    dim3 mean_blocks(batch_size, num_groups);
    dim3 mean_threads(256);
    int mean_shared_mem = mean_threads.x * sizeof(float);
    
    group_norm_mean_kernel<<<mean_blocks, mean_threads, mean_shared_mem>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        batch_size,
        num_features,
        spatial_size,
        group_size
    );
    
    // Launch kernel to compute variance
    dim3 var_blocks(batch_size, num_groups);
    dim3 var_threads(256);
    int var_shared_mem = var_threads.x * sizeof(float);
    
    group_norm_var_kernel<<<var_blocks, var_threads, var_shared_mem>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        batch_size,
        num_features,
        spatial_size,
        group_size
    );
    
    // Launch kernel to normalize
    int total_elements = batch_size * num_features * spatial_size;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    group_norm_normalize_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        weight.defined() ? weight.data_ptr<float>() : nullptr,
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        num_features,
        spatial_size,
        group_size,
        static_cast<float>(eps)
    );
    
    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    double eps);
"""

# Compile the inline CUDA code for Group Normalization
group_norm = load_inline(
    name="group_norm",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization with custom CUDA kernels.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return group_norm.group_norm_cuda(x, self.weight, self.bias, self.num_groups, self.eps)

batch_size = 112  # scaled up
features = 64
num_groups = 8
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [features, num_groups]  # num_features