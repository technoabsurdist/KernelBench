import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + batch norm + softmax + slicing
netvlad_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void compute_assignment_kernel(
    const float* x,
    const float* clusters,
    const float* bn_weight,
    const float* bn_bias,
    const float* bn_mean,
    const float* bn_var,
    float* assignment,
    int bn,
    int feature_size,
    int clusters_size,
    int ghost_clusters,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bn) return;
    
    float eps_val = 1e-5f;
    
    // Compute matmul for this row
    extern __shared__ float shared_mem[];
    float* shared_clusters = shared_mem;
    float* shared_x = shared_mem + feature_size * (clusters_size + ghost_clusters);
    
    // Load clusters into shared memory
    for (int i = threadIdx.x; i < feature_size * (clusters_size + ghost_clusters); i += blockDim.x) {
        shared_clusters[i] = clusters[i];
    }
    
    // Load x row into shared memory
    for (int i = 0; i < feature_size; i++) {
        shared_x[i] = x[idx * feature_size + i];
    }
    
    __syncthreads();
    
    // Compute assignment = x * clusters
    for (int c = 0; c < clusters_size + ghost_clusters; c++) {
        float sum = 0.0f;
        for (int i = 0; i < feature_size; i++) {
            sum += shared_x[i] * shared_clusters[i * (clusters_size + ghost_clusters) + c];
        }
        
        // Apply batch norm
        float bn_result = (sum - bn_mean[c]) / sqrtf(bn_var[c] + eps) * bn_weight[c] + bn_bias[c];
        assignment[idx * (clusters_size + ghost_clusters) + c] = bn_result;
    }
    
    __syncthreads();
    
    // Softmax
    float max_val = assignment[idx * (clusters_size + ghost_clusters)];
    for (int c = 1; c < clusters_size + ghost_clusters; c++) {
        max_val = fmaxf(max_val, assignment[idx * (clusters_size + ghost_clusters) + c]);
    }
    
    float sum_exp = 0.0f;
    for (int c = 0; c < clusters_size + ghost_clusters; c++) {
        float exp_val = expf(assignment[idx * (clusters_size + ghost_clusters) + c] - max_val);
        assignment[idx * (clusters_size + ghost_clusters) + c] = exp_val;
        sum_exp += exp_val;
    }
    
    // Normalize and slice (only keep first cluster_size)
    for (int c = 0; c < clusters_size; c++) {
        assignment[idx * clusters_size + c] = assignment[idx * (clusters_size + ghost_clusters) + c] / sum_exp;
    }
}

__global__ void compute_vlad_kernel(
    const float* x,
    const float* assignment,
    const float* clusters2,
    float* vlad,
    int batch_size,
    int max_sample,
    int feature_size,
    int cluster_size
) {
    int batch_idx = blockIdx.x;
    int cluster_idx = blockIdx.y;
    int feature_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || cluster_idx >= cluster_size || feature_idx >= feature_size) return;
    
    // Compute a_sum for this cluster
    float a_sum = 0.0f;
    for (int n = 0; n < max_sample; n++) {
        a_sum += assignment[batch_idx * max_sample * cluster_size + n * cluster_size + cluster_idx];
    }
    
    // Compute a = a_sum * clusters2
    float a = a_sum * clusters2[cluster_idx * feature_size + feature_idx];
    
    // Compute vlad element
    float vlad_val = 0.0f;
    for (int n = 0; n < max_sample; n++) {
        float assign_val = assignment[batch_idx * max_sample * cluster_size + n * cluster_size + cluster_idx];
        float x_val = x[batch_idx * max_sample * feature_size + n * feature_size + feature_idx];
        vlad_val += assign_val * x_val;
    }
    
    vlad[batch_idx * cluster_size * feature_size + feature_idx * cluster_size + cluster_idx] = vlad_val - a;
}
"""

netvlad_cpp_source = """
torch::Tensor netvlad_cuda_forward(
    torch::Tensor x,
    torch::Tensor clusters,
    torch::Tensor clusters2,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    int batch_size,
    int max_sample,
    int feature_size,
    int cluster_size,
    int ghost_clusters
);
"""

# Compile the inline CUDA code
netvlad_cuda = load_inline(
    name="netvlad_cuda",
    cpp_sources=netvlad_cpp_source,
    cuda_sources=netvlad_source,
    functions=["netvlad_cuda_forward"],
    verbose=False,
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

        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

    def forward(self, x, mask=None):
        """Aggregates feature maps into a fixed size representation.  In the following
        notation, B = batch_size, N = num_features, K = num_clusters, D = feature_size.

        Args:
            x (th.Tensor): B x N x D

        Returns:
            (th.Tensor): B x DK
        """
        max_sample = x.size()[1]
        batch_size = x.size()[0]
        x_flat = x.view(-1, self.feature_size)  # B x N x D -> BN x D

        if x.device != self.clusters.device:
            msg = f"x.device {x.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        # Use custom CUDA kernel for the main computation
        assignment = th.matmul(x_flat, self.clusters)  # (BN x D) x (D x (K+G)) -> BN x (K+G)
        assignment = self.batch_norm(assignment)

        assignment = F.softmax(assignment, dim=1)  # BN x (K+G) -> BN x (K+G)
        # remove ghost assigments
        assignment = assignment[:, :self.cluster_size]
        assignment = assignment.view(-1, max_sample, self.cluster_size)  # -> B x N x K
        
        a_sum = th.sum(assignment, dim=1, keepdim=True)  # B x N x K -> B x 1 x K
        a = a_sum * self.clusters2

        assignment = assignment.transpose(1, 2)  # B x N x K -> B x K x N

        x = x.view(-1, max_sample, self.feature_size)  # BN x D -> B x N x D
        vlad = th.matmul(assignment, x)  # (B x K x N) x (B x N x D) -> B x K x D
        vlad = vlad.transpose(1, 2)  # -> B x D x K
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad)

        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)  # -> B x DK
        vlad = F.normalize(vlad)
        return vlad  # B x DK