import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused assignment computation (matmul + batchnorm + softmax)
assignment_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void fused_assignment_softmax_kernel(
    const float* x, const float* clusters, const float* bn_weight, 
    const float* bn_bias, const float* bn_mean, const float* bn_var,
    float* assignment, 
    int batch_size, int feature_size, int total_clusters, int cluster_size,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * total_clusters;
    
    if (idx >= total_elements) return;
    
    int batch_idx = idx / total_clusters;
    int cluster_idx = idx % total_clusters;
    
    // Compute matmul for this element
    float sum = 0.0f;
    for (int d = 0; d < feature_size; d++) {
        sum += x[batch_idx * feature_size + d] * clusters[d * total_clusters + cluster_idx];
    }
    
    // Apply batch normalization
    float normalized = (sum - bn_mean[cluster_idx]) / sqrtf(bn_var[cluster_idx] + eps);
    float bn_out = normalized * bn_weight[cluster_idx] + bn_bias[cluster_idx];
    
    // Store for softmax
    assignment[idx] = bn_out;
}

__global__ void softmax_kernel(float* assignment, int batch_size, int total_clusters, int cluster_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    extern __shared__ float shared_data[];
    
    int tid = threadIdx.x;
    int offset = batch_idx * total_clusters;
    
    // Find max
    float max_val = -1e30f;
    for (int i = tid; i < total_clusters; i += blockDim.x) {
        max_val = fmaxf(max_val, assignment[offset + i]);
    }
    
    // Reduce max
    shared_data[tid] = max_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + s]);
        }
        __syncthreads();
    }
    max_val = shared_data[0];
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = tid; i < total_clusters; i += blockDim.x) {
        float exp_val = expf(assignment[offset + i] - max_val);
        assignment[offset + i] = exp_val;
        sum += exp_val;
    }
    
    // Reduce sum
    shared_data[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    sum = shared_data[0];
    
    // Normalize - only keep cluster_size elements
    for (int i = tid; i < cluster_size; i += blockDim.x) {
        assignment[offset + i] = assignment[offset + i] / sum;
    }
}

torch::Tensor fused_assignment_cuda(torch::Tensor x, torch::Tensor clusters, 
                                    torch::Tensor bn_weight, torch::Tensor bn_bias,
                                    torch::Tensor bn_mean, torch::Tensor bn_var,
                                    int cluster_size, float eps) {
    auto batch_size = x.size(0);
    auto feature_size = x.size(1);
    auto total_clusters = clusters.size(1);
    
    auto assignment = torch::zeros({batch_size, total_clusters}, x.options());
    
    const int threads = 256;
    const int blocks = (batch_size * total_clusters + threads - 1) / threads;
    
    fused_assignment_softmax_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), clusters.data_ptr<float>(),
        bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(), bn_var.data_ptr<float>(),
        assignment.data_ptr<float>(),
        batch_size, feature_size, total_clusters, cluster_size, eps
    );
    
    softmax_kernel<<<batch_size, 256, 256 * sizeof(float)>>>(
        assignment.data_ptr<float>(), batch_size, total_clusters, cluster_size
    );
    
    return assignment.slice(1, 0, cluster_size);
}
"""

# Custom CUDA kernel for VLAD aggregation
vlad_aggregation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vlad_aggregation_kernel(
    const float* assignment, const float* x, const float* clusters2,
    float* vlad, int batch_size, int num_features, 
    int cluster_size, int feature_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * cluster_size * feature_size;
    
    if (idx >= total) return;
    
    int b = idx / (cluster_size * feature_size);
    int remainder = idx % (cluster_size * feature_size);
    int k = remainder / feature_size;
    int d = remainder % feature_size;
    
    float sum = 0.0f;
    float a_sum = 0.0f;
    
    for (int n = 0; n < num_features; n++) {
        float a_val = assignment[b * cluster_size * num_features + k * num_features + n];
        sum += a_val * x[b * num_features * feature_size + n * feature_size + d];
        a_sum += a_val;
    }
    
    float cluster_val = clusters2[d * cluster_size + k];
    vlad[b * feature_size * cluster_size + d * cluster_size + k] = sum - a_sum * cluster_val;
}

torch::Tensor vlad_aggregation_cuda(torch::Tensor assignment, torch::Tensor x, 
                                    torch::Tensor clusters2) {
    auto batch_size = x.size(0);
    auto num_features = x.size(1);
    auto feature_size = x.size(2);
    auto cluster_size = assignment.size(2);
    
    auto vlad = torch::zeros({batch_size, feature_size, cluster_size}, x.options());
    
    const int threads = 256;
    const int total = batch_size * cluster_size * feature_size;
    const int blocks = (total + threads - 1) / threads;
    
    vlad_aggregation_kernel<<<blocks, threads>>>(
        assignment.data_ptr<float>(), x.data_ptr<float>(), 
        clusters2.data_ptr<float>(), vlad.data_ptr<float>(),
        batch_size, num_features, cluster_size, feature_size
    );
    
    return vlad;
}
"""

# Custom CUDA kernel for L2 normalization
l2_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void l2_normalize_kernel(float* data, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    extern __shared__ float shared_sum[];
    
    int tid = threadIdx.x;
    int offset = batch_idx * dim;
    
    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = data[offset + i];
        sum += val * val;
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    float norm = sqrtf(shared_sum[0] + 1e-12f);
    
    for (int i = tid; i < dim; i += blockDim.x) {
        data[offset + i] /= norm;
    }
}

torch::Tensor l2_normalize_cuda(torch::Tensor x) {
    auto shape = x.sizes();
    auto batch_size = shape[0];
    auto last_dims = 1;
    for (int i = 1; i < shape.size(); i++) {
        last_dims *= shape[i];
    }
    
    auto x_flat = x.contiguous().view({batch_size, last_dims});
    auto output = x_flat.clone();
    
    const int threads = 256;
    l2_normalize_kernel<<<batch_size, threads, threads * sizeof(float)>>>(
        output.data_ptr<float>(), batch_size, last_dims
    );
    
    return output.view(shape);
}
"""

assignment_cpp_source = "torch::Tensor fused_assignment_cuda(torch::Tensor x, torch::Tensor clusters, torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var, int cluster_size, float eps);"
vlad_cpp_source = "torch::Tensor vlad_aggregation_cuda(torch::Tensor assignment, torch::Tensor x, torch::Tensor clusters2);"
l2_norm_cpp_source = "torch::Tensor l2_normalize_cuda(torch::Tensor x);"

assignment_module = load_inline(
    name="assignment_module",
    cpp_sources=assignment_cpp_source,
    cuda_sources=assignment_kernel_source,
    functions=["fused_assignment_cuda"],
    verbose=True,
)

vlad_module = load_inline(
    name="vlad_module",
    cpp_sources=vlad_cpp_source,
    cuda_sources=vlad_aggregation_source,
    functions=["vlad_aggregation_cuda"],
    verbose=True,
)

l2_norm_module = load_inline(
    name="l2_norm_module",
    cpp_sources=l2_norm_cpp_source,
    cuda_sources=l2_norm_source,
    functions=["l2_normalize_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

    def forward(self, x, mask=None):
        max_sample = x.size()[1]
        x_flat = x.view(-1, self.feature_size)

        if x.device != self.clusters.device:
            msg = f"x.device {x.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        # Fused assignment computation
        assignment = assignment_module.fused_assignment_cuda(
            x_flat, self.clusters, 
            self.batch_norm.weight, self.batch_norm.bias,
            self.batch_norm.running_mean, self.batch_norm.running_var,
            self.cluster_size, self.batch_norm.eps
        )
        
        assignment = assignment.view(-1, max_sample, self.cluster_size)
        assignment = assignment.transpose(1, 2)

        x = x.view(-1, max_sample, self.feature_size)
        
        # Fused VLAD aggregation
        vlad = vlad_module.vlad_aggregation_cuda(
            assignment, x, self.clusters2.squeeze(0)
        )

        # L2 intra norm
        vlad = l2_norm_module.l2_normalize_cuda(vlad)

        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = l2_norm_module.l2_normalize_cuda(vlad)
        
        return vlad