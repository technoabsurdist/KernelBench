import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused matmul + softmax with ghost cluster removal
fused_assignment_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void fused_matmul_softmax_kernel(
    const float* __restrict__ x,
    const float* __restrict__ clusters,
    float* __restrict__ output,
    int batch_size,
    int feature_size,
    int total_clusters,
    int output_clusters
) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    if (bid >= batch_size) return;
    
    const float* x_row = x + bid * feature_size;
    float* out_row = output + bid * output_clusters;
    
    // Compute matmul and find max for numerical stability
    float max_val = -1e20f;
    for (int c = tid; c < total_clusters; c += blockDim.x) {
        float sum = 0.0f;
        for (int f = 0; f < feature_size; f++) {
            sum += x_row[f] * clusters[f * total_clusters + c];
        }
        if (c < output_clusters) {
            shared_mem[c] = sum;
            max_val = fmaxf(max_val, sum);
        }
    }
    
    // Reduce max across threads
    __shared__ float shared_max;
    if (tid == 0) shared_max = -1e20f;
    __syncthreads();
    atomicMax((int*)&shared_max, __float_as_int(max_val));
    __syncthreads();
    max_val = shared_max;
    
    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int c = tid; c < output_clusters; c += blockDim.x) {
        float exp_val = expf(shared_mem[c] - max_val);
        shared_mem[c] = exp_val;
        sum_exp += exp_val;
    }
    
    // Reduce sum
    __shared__ float shared_sum;
    if (tid == 0) shared_sum = 0.0f;
    __syncthreads();
    atomicAdd(&shared_sum, sum_exp);
    __syncthreads();
    
    // Normalize
    for (int c = tid; c < output_clusters; c += blockDim.x) {
        out_row[c] = shared_mem[c] / shared_sum;
    }
}

torch::Tensor fused_assignment_cuda(
    torch::Tensor x,
    torch::Tensor clusters,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    int cluster_size,
    int ghost_clusters
) {
    int batch_size = x.size(0);
    int feature_size = x.size(1);
    int total_clusters = cluster_size + ghost_clusters;
    
    auto output = torch::zeros({batch_size, cluster_size}, x.options());
    
    // Apply batch norm inline
    auto clusters_bn = clusters.clone();
    for (int c = 0; c < total_clusters; c++) {
        float scale = bn_weight[c].item<float>() / sqrtf(bn_var[c].item<float>() + 1e-5f);
        float shift = bn_bias[c].item<float>() - bn_mean[c].item<float>() * scale;
        for (int f = 0; f < feature_size; f++) {
            clusters_bn[f][c] = clusters_bn[f][c] * scale;
        }
    }
    
    int threads = 256;
    int blocks = batch_size;
    size_t shared_size = cluster_size * sizeof(float);
    
    fused_matmul_softmax_kernel<<<blocks, threads, shared_size>>>(
        x.data_ptr<float>(),
        clusters_bn.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        feature_size,
        total_clusters,
        cluster_size
    );
    
    return output;
}
"""

fused_assignment_cpp_source = """
torch::Tensor fused_assignment_cuda(
    torch::Tensor x,
    torch::Tensor clusters,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    int cluster_size,
    int ghost_clusters
);
"""

# Custom CUDA kernel for VLAD aggregation
vlad_aggregation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vlad_aggregation_kernel(
    const float* __restrict__ assignment,
    const float* __restrict__ x,
    const float* __restrict__ clusters2,
    const float* __restrict__ a_sum,
    float* __restrict__ output,
    int batch_size,
    int num_features,
    int cluster_size,
    int feature_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * feature_size * cluster_size;
    
    if (idx < total_elements) {
        int b = idx / (feature_size * cluster_size);
        int remainder = idx % (feature_size * cluster_size);
        int d = remainder / cluster_size;
        int k = remainder % cluster_size;
        
        float sum = 0.0f;
        for (int n = 0; n < num_features; n++) {
            sum += assignment[b * cluster_size * num_features + k * num_features + n] * 
                   x[b * num_features * feature_size + n * feature_size + d];
        }
        
        float a_val = a_sum[b * cluster_size + k] * clusters2[d * cluster_size + k];
        output[b * feature_size * cluster_size + d * cluster_size + k] = sum - a_val;
    }
}

torch::Tensor vlad_aggregation_cuda(
    torch::Tensor assignment,
    torch::Tensor x,
    torch::Tensor clusters2,
    torch::Tensor a_sum
) {
    int batch_size = x.size(0);
    int num_features = x.size(1);
    int feature_size = x.size(2);
    int cluster_size = assignment.size(1);
    
    auto output = torch::zeros({batch_size, feature_size, cluster_size}, x.options());
    
    int total_elements = batch_size * feature_size * cluster_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    vlad_aggregation_kernel<<<blocks, threads>>>(
        assignment.data_ptr<float>(),
        x.data_ptr<float>(),
        clusters2.data_ptr<float>(),
        a_sum.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_features,
        cluster_size,
        feature_size
    );
    
    return output;
}
"""

vlad_aggregation_cpp_source = """
torch::Tensor vlad_aggregation_cuda(
    torch::Tensor assignment,
    torch::Tensor x,
    torch::Tensor clusters2,
    torch::Tensor a_sum
);
"""

# Compile the inline CUDA code
fused_assignment = load_inline(
    name="fused_assignment",
    cpp_sources=fused_assignment_cpp_source,
    cuda_sources=fused_assignment_source,
    functions=["fused_assignment_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

vlad_aggregation = load_inline(
    name="vlad_aggregation",
    cpp_sources=vlad_aggregation_cpp_source,
    cuda_sources=vlad_aggregation_source,
    functions=["vlad_aggregation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
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
        
        self.fused_assignment = fused_assignment
        self.vlad_aggregation = vlad_aggregation

    def forward(self, x, mask=None):
        max_sample = x.size()[1]
        batch_size = x.size()[0]
        x_flat = x.view(-1, self.feature_size)

        if x.device != self.clusters.device:
            msg = f"x.device {x.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        # Use custom fused kernel for assignment computation
        assignment = self.fused_assignment.fused_assignment_cuda(
            x_flat.contiguous(),
            self.clusters.contiguous(),
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.cluster_size,
            self.ghost_clusters
        )
        
        assignment = assignment.view(batch_size, max_sample, self.cluster_size)
        a_sum = th.sum(assignment, dim=1, keepdim=False)
        assignment = assignment.transpose(1, 2)

        # Use custom VLAD aggregation kernel
        clusters2_expanded = self.clusters2.squeeze(0)
        vlad = self.vlad_aggregation.vlad_aggregation_cuda(
            assignment.contiguous(),
            x.contiguous(),
            clusters2_expanded.contiguous(),
            a_sum.contiguous()
        )

        # L2 intra norm
        vlad = F.normalize(vlad)

        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad)
        return vlad