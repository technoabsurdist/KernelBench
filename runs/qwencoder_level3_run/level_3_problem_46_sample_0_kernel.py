import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + softmax + slicing
netvlad_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void compute_assignment_kernel(
    const float* x,
    const float* clusters,
    const float* running_mean,
    const float* running_var,
    float* assignment,
    int bn,
    int feature_size,
    int clusters_size,
    int ghost_clusters,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bn) return;
    
    for (int k = 0; k < clusters_size; k++) {
        float sum = 0.0f;
        for (int d = 0; d < feature_size; d++) {
            sum += x[idx * feature_size + d] * clusters[d * (clusters_size + ghost_clusters) + k];
        }
        
        // Batch norm: (x - mean) / sqrt(var + eps)
        float normalized = (sum - running_mean[k]) / sqrtf(running_var[k] + eps);
        assignment[idx * (clusters_size + ghost_clusters) + k] = normalized;
    }
}

__global__ void softmax_slice_kernel(
    const float* input,
    float* output,
    int rows,
    int cols,
    int output_cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    
    // Compute softmax for all columns first
    float max_val = -INFINITY;
    for (int c = 0; c < cols; c++) {
        max_val = fmaxf(max_val, input[row * cols + c]);
    }
    
    float sum = 0.0f;
    for (int c = 0; c < cols; c++) {
        float exp_val = expf(input[row * cols + c] - max_val);
        sum += exp_val;
    }
    
    // Only output the first output_cols columns
    for (int c = 0; c < output_cols; c++) {
        output[row * output_cols + c] = expf(input[row * cols + c] - max_val) / sum;
    }
}

__global__ void vlad_computation_kernel(
    const float* assignment,
    const float* x,
    const float* clusters2,
    float* vlad,
    int batch_size,
    int max_sample,
    int cluster_size,
    int feature_size
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
    
    // Compute 'a' value
    float a_val = a_sum * clusters2[cluster_idx * feature_size + feature_idx];
    
    // Compute VLAD component
    float vlad_val = 0.0f;
    for (int n = 0; n < max_sample; n++) {
        float assign_val = assignment[batch_idx * max_sample * cluster_size + n * cluster_size + cluster_idx];
        float x_val = x[batch_idx * max_sample * feature_size + n * feature_size + feature_idx];
        vlad_val += assign_val * x_val;
    }
    
    vlad[batch_idx * cluster_size * feature_size + cluster_idx * feature_size + feature_idx] = vlad_val - a_val;
}

torch::Tensor netvlad_cuda(
    torch::Tensor x,
    torch::Tensor clusters,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor clusters2,
    int ghost_clusters,
    float eps
) {
    auto batch_size = x.size(0);
    auto max_sample = x.size(1);
    auto feature_size = x.size(2);
    auto clusters_size = clusters2.size(2);
    
    // Reshape x to (BN x D)
    auto x_flat = x.view({-1, feature_size});
    auto bn = x_flat.size(0);
    
    // Step 1: MatMul + BatchNorm
    auto assignment_raw = torch::zeros({bn, clusters.size(1)}, x.options());
    
    const int block_size = 256;
    const int num_blocks = (bn + block_size - 1) / block_size;
    
    compute_assignment_kernel<<<num_blocks, block_size>>>(
        x_flat.data_ptr<float>(),
        clusters.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        assignment_raw.data_ptr<float>(),
        bn,
        feature_size,
        clusters_size,
        ghost_clusters,
        eps
    );
    
    // Step 2: Softmax + Slicing
    auto assignment_sliced = torch::zeros({bn, clusters_size}, x.options());
    const int softmax_blocks = (bn + block_size - 1) / block_size;
    
    softmax_slice_kernel<<<softmax_blocks, block_size>>>(
        assignment_raw.data_ptr<float>(),
        assignment_sliced.data_ptr<float>(),
        bn,
        clusters.size(1),
        clusters_size
    );
    
    // Reshape for next steps
    auto assignment = assignment_sliced.view({batch_size, max_sample, clusters_size});
    
    // Step 3: VLAD computation
    auto vlad = torch::zeros({batch_size, cluster_size, feature_size}, x.options());
    
    dim3 grid(batch_size, cluster_size);
    dim3 block(feature_size);
    
    vlad_computation_kernel<<<grid, block>>>(
        assignment.data_ptr<float>(),
        x.data_ptr<float>(),
        clusters2.data_ptr<float>(),
        vlad.data_ptr<float>(),
        batch_size,
        max_sample,
        cluster_size,
        feature_size
    );
    
    // Transpose and reshape
    vlad = vlad.transpose(1, 2);
    
    // L2 normalization (simplified)
    vlad = torch::nn::functional::normalize(vlad, torch::nn::functional::NormalizeFuncOptions().dim(-1));
    vlad = vlad.reshape({-1, cluster_size * feature_size});
    vlad = torch::nn::functional::normalize(vlad, torch::nn::functional::NormalizeFuncOptions().dim(-1));
    
    return vlad;
}
"""

netvlad_cpp_source = """
torch::Tensor netvlad_cuda(
    torch::Tensor x,
    torch::Tensor clusters,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor clusters2,
    int ghost_clusters,
    float eps
);
"""

# Compile the inline CUDA code
netvlad_ops = load_inline(
    name="netvlad_ops",
    cpp_sources=netvlad_cpp_source,
    cuda_sources=netvlad_source,
    functions=["netvlad_cuda"],
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
        
        # Store ops
        self.netvlad_ops = netvlad_ops

    def forward(self, x, mask=None):
        """Aggregates feature maps into a fixed size representation."""
        # Use custom CUDA implementation
        if x.is_cuda:
            # Get batch norm parameters
            running_mean = self.batch_norm.running_mean
            running_var = self.batch_norm.running_var
            eps = self.batch_norm.eps
            
            # Ensure clusters2 is properly shaped
            clusters2_reshaped = self.clusters2.squeeze(0).transpose(0, 1).contiguous()  # K x D
            
            return self.netvlad_ops.netvlad_cuda(
                x,
                self.clusters,
                running_mean,
                running_var,
                clusters2_reshaped,
                self.ghost_clusters,
                eps
            )
        else:
            # Fallback to original implementation for CPU
            max_sample = x.size()[1]
            x = x.view(-1, self.feature_size)  # B x N x D -> BN x D

            if x.device != self.clusters.device:
                msg = f"x.device {x.device} != cluster.device {self.clusters.device}"
                raise ValueError(msg)

            assignment = th.matmul(x, self.clusters)  # (BN x D) x (D x (K+G)) -> BN x (K+G)
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

batch_size = 2048
num_features = 100
num_clusters = 32
feature_size = 512
ghost_clusters = 16

def get_inputs():
    return [torch.rand(batch_size, num_features, feature_size).cuda()]

def get_init_inputs():
    return [num_clusters, feature_size, ghost_clusters]