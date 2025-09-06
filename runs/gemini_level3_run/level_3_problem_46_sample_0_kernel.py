import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math
import torch as th

# Define the custom CUDA kernel for the fused VLAD core operations
fused_vlad_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_vlad_norm_kernel(
    const float* __restrict__ x,          // Input features: B x N x D
    const float* __restrict__ assignment, // Soft assignments: B x N x K
    const float* __restrict__ clusters2,  // Cluster centers: 1 x D x K
    float* __restrict__ out,              // Output VLAD vectors: B x D x K
    const int B,                          // Batch size
    const int N,                          // Number of features per item
    const int D,                          // Feature dimension
    const int K) {                        // Number of clusters

    // Each block computes the VLAD vector for one batch item and one cluster.
    // Grid dimensions: (B, K)
    const int b = blockIdx.x;
    const int k = blockIdx.y;

    // Allocate shared memory for both the unnormalized vector and the squared values for reduction.
    // This avoids re-computation of the unnormalized vector.
    extern __shared__ float s_mem[];
    float* s_vlad = s_mem;
    float* s_norm_sq = &s_mem[D];

    // Step 1: Calculate the unnormalized VLAD vector.
    // Each thread computes the sum for one dimension 'd'.
    // The result is stored in s_vlad, and its square is stored in s_norm_sq.
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float sum_val = 0.0f;
        // Loop over all N features for the current batch item 'b'.
        for (int n = 0; n < N; ++n) {
            // Calculate the residual: x_i - c_k
            const float residual = x[b * N * D + n * D + d] - clusters2[d * K + k];
            // Get the soft assignment weight a_k(x_i)
            const float weight = assignment[b * N * K + n * K + k];
            // Accumulate the weighted residual
            sum_val += weight * residual;
        }
        s_vlad[d] = sum_val;
        s_norm_sq[d] = sum_val * sum_val;
    }
    __syncthreads();

    // Step 2: Perform parallel reduction to get the L2 norm squared.
    // The reduction is performed in-place on the s_norm_sq array.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            // Check boundary to handle cases where D is not a multiple of blockDim.x
            if (threadIdx.x + s < D) {
                s_norm_sq[threadIdx.x] += s_norm_sq[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    // The L2 norm is the square root of the first element, with an epsilon for stability.
    const float norm = sqrtf(s_norm_sq[0]) + 1e-6f;

    // Step 3: Normalize the vector using the computed norm and write to global output memory.
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        out[b * D * K + d * K + k] = s_vlad[d] / norm;
    }
}

// C++ wrapper function that PyTorch will call.
// This function sets up the kernel launch.
torch::Tensor fused_vlad_norm_cuda(
    torch::Tensor x,
    torch::Tensor assignment,
    torch::Tensor clusters2) {

    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor 'x' must be on a CUDA device");
    TORCH_CHECK(assignment.is_cuda(), "Input tensor 'assignment' must be on a CUDA device");
    TORCH_CHECK(clusters2.is_cuda(), "Input tensor 'clusters2' must be on a CUDA device");
    TORCH_CHECK(x.is_contiguous(), "Input tensor 'x' must be contiguous");
    TORCH_CHECK(assignment.is_contiguous(), "Input tensor 'assignment' must be contiguous");
    TORCH_CHECK(clusters2.is_contiguous(), "Input tensor 'clusters2' must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensors must be of type float32");
    TORCH_CHECK(assignment.scalar_type() == torch::kFloat32, "Input tensors must be of type float32");
    TORCH_CHECK(clusters2.scalar_type() == torch::kFloat32, "Input tensors must be of type float32");


    const auto B = x.size(0);
    const auto N = x.size(1);
    const auto D = x.size(2);
    const auto K = assignment.size(2);

    // Create the output tensor
    auto out = torch::empty({B, D, K}, x.options());

    // Kernel launch configuration
    const dim3 threads(256);
    const dim3 blocks(B, K);
    // Shared memory size: D floats for s_vlad + D floats for s_norm_sq
    const int shared_mem_size = 2 * D * sizeof(float);

    // Launch the kernel
    fused_vlad_norm_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        assignment.data_ptr<float>(),
        clusters2.data_ptr<float>(),
        out.data_ptr<float>(),
        B, N, D, K
    );

    return out;
}
"""

# C++ source for the function signature, required by load_inline
fused_vlad_cpp_source = """
torch::Tensor fused_vlad_norm_cuda(torch::Tensor x, torch::Tensor assignment, torch::Tensor clusters2);
"""

# JIT compile the inline CUDA code. This happens once when the module is imported.
fused_vlad_op = load_inline(
    name="fused_vlad_op",
    cpp_sources=fused_vlad_cpp_source,
    cuda_sources=fused_vlad_source,
    functions=["fused_vlad_norm_cuda"],
    verbose=False,
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

        # Store the compiled custom operator
        self.fused_vlad_op = fused_vlad_op

    def forward(self, x, mask=None):
        """Aggregates feature maps into a fixed size representation.  In the following
        notation, B = batch_size, N = num_features, K = num_clusters, D = feature_size.
        This version uses a custom CUDA kernel to fuse the core VLAD operations.

        Args:
            x (th.Tensor): B x N x D

        Returns:
            (th.Tensor): B x DK
        """
        max_sample = x.size()[1]
        
        # Ensure input is on the same device as model parameters
        if x.device != self.clusters.device:
            x = x.to(self.clusters.device)

        # --- Part 1: Soft-assignment (using standard PyTorch ops) ---
        x_flat = x.view(-1, self.feature_size)
        assignment = th.matmul(x_flat, self.clusters)
        assignment = self.batch_norm(assignment)
        assignment = F.softmax(assignment, dim=1)
        assignment = assignment[:, :self.cluster_size]
        assignment = assignment.view(-1, max_sample, self.cluster_size)  # B x N x K

        # --- Part 2: Fused VLAD core and Intra-normalization (custom CUDA kernel) ---
        # The custom kernel fuses the following operations:
        # 1. Summing assignments: a_sum = th.sum(assignment, dim=1)
        # 2. Calculating residuals: vlad = matmul(assignment.T, x) - a_sum * clusters2
        # 3. L2 intra-normalization: F.normalize(vlad, dim=1)
        vlad = self.fused_vlad_op.fused_vlad_norm_cuda(x, assignment, self.clusters2) # Output: B x D x K

        # --- Part 3: Final flattening and normalization (using standard PyTorch ops) ---
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad)
        return vlad