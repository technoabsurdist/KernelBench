import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.utils.cpp_extension import load_inline

# Fused CUDA kernel for the core VLAD aggregation logic.
# This kernel fuses the following operations into a single launch:
# 1. Sum of assignments per cluster (a_sum)
# 2. Batched matrix multiplication (assignment.T @ x)
# 3. Subtraction of the weighted cluster centers (vlad - a)
# 4. Intra-normalization (L2 norm per cluster vector)
# This avoids creating large intermediate tensors and reduces memory bandwidth,
# which is often the bottleneck in such operations.
vlad_aggregation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// A device function for block-wide sum reduction.
// It uses a temporary shared memory buffer provided by the caller.
__device__ __forceinline__ float blockReduceSum(float val, float* sdata) {
    int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // The result is in sdata[0], all threads read it after sync.
    return sdata[0];
}

__global__ void vlad_aggregation_kernel(
    const float* assignment, // (B, N, K)
    const float* x,          // (B, N, D)
    const float* clusters2,  // (1, D, K)
    float* vlad_out,         // (B, K, D) -> This is pre-transpose
    const int B, const int N, const int K, const int D, const int block_size
) {
    // Grid: (K, B), Block: (block_size)
    // Each block computes one vector of size D for a given (b, k) -> vlad_out[b, k, :]

    // Use dynamically allocated shared memory
    extern __shared__ float s_mem[];
    float* s_vlad_unnormalized = s_mem; // size D
    float* s_reduce_helper = &s_mem[D]; // size block_size

    const int k = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;

    // Step 1: Compute a_sum for this (b, k)
    // Parallel reduction over N
    float thread_a_sum = 0.0f;
    for (int n = tid; n < N; n += block_size) {
        thread_a_sum += assignment[b * N * K + n * K + k];
    }
    const float a_sum = blockReduceSum(thread_a_sum, s_reduce_helper);

    // Step 2: Compute unnormalized vlad vector and store in shared memory
    // Threads in the block cooperate to compute the D-dimensional vector
    for (int d = tid; d < D; d += block_size) {
        // Compute v_kd = sum over n of (assignment[b,n,k] * x[b,n,d])
        float v_kd = 0.0f;
        for (int n = 0; n < N; ++n) {
            v_kd += assignment[b * N * K + n * K + k] * x[b * N * D + n * D + d];
        }

        // Compute a_kd = a_sum * clusters2[d,k]
        // clusters2 has shape (1, D, K), so first index is 0
        float a_kd = a_sum * clusters2[d * K + k];

        s_vlad_unnormalized[d] = v_kd - a_kd;
    }
    __syncthreads();

    // Step 3: Compute L2 norm squared from shared memory
    float thread_norm_sq_sum = 0.0f;
    for (int d = tid; d < D; d += block_size) {
        float val = s_vlad_unnormalized[d];
        thread_norm_sq_sum += val * val;
    }
    const float norm_sq = blockReduceSum(thread_norm_sq_sum, s_reduce_helper);

    // Step 4: Normalize and write to global memory
    const float norm_inv = rsqrtf(norm_sq + 1e-6f);
    for (int d = tid; d < D; d += block_size) {
        // Output layout is (B, K, D) for coalesced writes
        vlad_out[b * K * D + k * D + d] = s_vlad_unnormalized[d] * norm_inv;
    }
}

torch::Tensor vlad_aggregation_cuda(
    torch::Tensor assignment, // (B, N, K)
    torch::Tensor x,          // (B, N, D)
    torch::Tensor clusters2   // (1, D, K)
) {
    TORCH_CHECK(assignment.is_cuda(), "assignment must be a CUDA tensor");
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(clusters2.is_cuda(), "clusters2 must be a CUDA tensor");

    TORCH_CHECK(assignment.is_contiguous(), "assignment must be contiguous");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(clusters2.is_contiguous(), "clusters2 must be contiguous");

    const int B = assignment.size(0);
    const int N = assignment.size(1);
    const int K = assignment.size(2);
    const int D = x.size(2);

    // The kernel computes the (B, K, D) tensor, which is then transposed
    auto vlad_out_bkd = torch::empty({B, K, D}, assignment.options());

    const int block_size = 256;
    dim3 block(block_size);
    dim3 grid(K, B);

    // Shared memory size: D for the unnormalized vector + block_size for reduction helper
    size_t shared_mem_size = (D + block_size) * sizeof(float);

    vlad_aggregation_kernel<<<grid, block, shared_mem_size>>>(
        assignment.data_ptr<float>(),
        x.data_ptr<float>(),
        clusters2.data_ptr<float>(),
        vlad_out_bkd.data_ptr<float>(),
        B, N, K, D, block_size
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // The original code transposes from (B, K, D) to (B, D, K) after this stage.
    // The final output of this fused op should match the output of F.normalize(vlad)
    // which has shape (B, D, K). We call .contiguous() to ensure the memory layout
    // is correct for the subsequent reshape operation.
    return vlad_out_bkd.transpose(1, 2).contiguous();
}
"""

vlad_aggregation_cpp_source = """
torch::Tensor vlad_aggregation_cuda(torch::Tensor assignment, torch::Tensor x, torch::Tensor clusters2);
"""

# Use a try-except block to handle potential compilation errors gracefully,
# especially in environments where the CUDA toolkit might not be perfectly configured.
try:
    vlad_aggregation = load_inline(
        name="vlad_aggregation",
        cpp_sources=vlad_aggregation_cpp_source,
        cuda_sources=vlad_aggregation_source,
        functions=["vlad_aggregation_cuda"],
        verbose=False,
    )
except Exception as e:
    print(f"Warning: Failed to load custom CUDA kernel for VLAD aggregation. Falling back to PyTorch's default implementation. Error: {e}")
    vlad_aggregation = None


class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters
        self.vlad_aggregation_kernel = vlad_aggregation

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
        x_reshaped = x.view(-1, self.feature_size)  # B x N x D -> BN x D

        if x.device != self.clusters.device:
            msg = f"x.device {x.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        assignment = th.matmul(x_reshaped, self.clusters)  # (BN x D) x (D x (K+G)) -> BN x (K+G)
        assignment = self.batch_norm(assignment)

        assignment = F.softmax(assignment, dim=1)  # BN x (K+G) -> BN x (K+G)
        # remove ghost assigments
        assignment = assignment[:, :self.cluster_size]
        assignment = assignment.view(-1, max_sample, self.cluster_size)  # -> B x N x K

        # --- Fused CUDA Kernel Section ---
        # Check if the kernel was compiled successfully and if the input is on CUDA
        if self.vlad_aggregation_kernel is not None and x.is_cuda:
            # Use the custom fused kernel for aggregation and intra-normalization
            vlad = self.vlad_aggregation_kernel.vlad_aggregation_cuda(assignment, x, self.clusters2)
        else:
            # Fallback to the original PyTorch implementation
            a_sum = th.sum(assignment, dim=1, keepdim=True)  # B x N x K -> B x 1 x K
            a = a_sum * self.clusters2

            assignment_T = assignment.transpose(1, 2)  # B x N x K -> B x K x N

            vlad = th.matmul(assignment_T, x)  # (B x K x N) x (B x N x D) -> B x K x D
            vlad = vlad.transpose(1, 2)  # -> B x D x K
            vlad = vlad - a

            # L2 intra norm
            vlad = F.normalize(vlad)
        # --- End Fused Section ---

        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)  # -> B x DK
        vlad = F.normalize(vlad)
        return vlad  # B x DK