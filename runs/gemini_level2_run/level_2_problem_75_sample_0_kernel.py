import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for the fused operation: GroupNorm + BiasAdd + MinReduction
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <algorithm>

// Device function for a block-level reduction to find the minimum value.
// It uses the provided shared memory pointer for the reduction.
template <typename T>
__device__ void block_reduce_min(T& val, T* shared_mem) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    shared_mem[tid] = val;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] = min(shared_mem[tid], shared_mem[tid + s]);
        }
        __syncthreads();
    }
    val = shared_mem[0];
}


__global__ void fused_groupnorm_bias_min_kernel(
    const float* __restrict__ x,      // Input tensor (N, C)
    const float* __restrict__ gamma,  // GroupNorm weight (C)
    const float* __restrict__ beta,   // GroupNorm bias (C)
    const float* __restrict__ bias,   // Extra bias (C)
    float* __restrict__ out,          // Output tensor (N, 1)
    int N,
    int C,
    int num_groups,
    float eps) {

    // One block per sample in the batch
    int n = blockIdx.x;
    if (n >= N) return;

    int group_size = C / num_groups;
    
    // Dynamic shared memory is used for several purposes.
    // Layout for stats calculation:
    // | s_x (C floats) | s_mean (num_groups floats) | s_var (num_groups floats) |
    extern __shared__ float sdata[];
    float* s_x = sdata;
    float* s_mean = &s_x[C];
    float* s_var = &s_mean[num_groups];

    const float* x_row = x + n * C;

    // --- Step 1: Load input row into shared memory ---
    // Each thread loads C / blockDim.x elements.
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        s_x[i] = x_row[i];
    }
    __syncthreads();

    // --- Step 2: Compute mean and variance for each group ---
    // The first `num_groups` threads each compute stats for one group.
    if (threadIdx.x < num_groups) {
        int g = threadIdx.x;
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        // Each thread iterates over its assigned group's elements in shared memory.
        #pragma unroll
        for (int i = 0; i < group_size; ++i) {
            float val = s_x[g * group_size + i];
            sum += val;
            sum_sq += val * val;
        }
        float mean = sum / group_size;
        s_mean[g] = mean;
        s_var[g] = sum_sq / group_size - mean * mean;
    }
    __syncthreads();

    // --- Step 3: Normalize, apply biases, and find thread-local minimum ---
    float local_min = FLT_MAX;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        int group_idx = i / group_size;
        float mean = s_mean[group_idx];
        float var = s_var[group_idx];

        float normalized_val = (s_x[i] - mean) * rsqrtf(var + eps);
        float final_val = normalized_val * gamma[i] + beta[i] + bias[i];
        
        local_min = min(local_min, final_val);
    }

    // --- Step 4: Block-level reduction to find the minimum for the row ---
    // We can reuse the beginning of the shared memory buffer for this reduction.
    float* s_reduce_mem = sdata;
    block_reduce_min(local_min, s_reduce_mem);

    // --- Step 5: Write result ---
    if (threadIdx.x == 0) {
        out[n] = local_min;
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor bias,
    int num_groups,
    float eps) {

    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "Input gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "Input beta must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Input bias must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "Input gamma must be contiguous");
    TORCH_CHECK(beta.is_contiguous(), "Input beta must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Input bias must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Inputs must be float32 tensors");

    const auto N = x.size(0);
    const auto C = x.size(1);

    auto out = torch::empty({N, 1}, x.options());

    const int block_size = 512;
    const int grid_size = N;

    // Shared memory size calculation.
    // The buffer is used for two purposes at different times:
    // 1. Storing input + stats: needs (C + 2 * num_groups) floats.
    // 2. Block-wide reduction: needs `block_size` floats.
    // We allocate enough for the larger of the two.
    size_t stats_mem_req = (C + 2 * num_groups) * sizeof(float);
    size_t reduce_mem_req = block_size * sizeof(float);
    size_t shared_mem_size = std::max(stats_mem_req, reduce_mem_req);

    fused_groupnorm_bias_min_kernel<<<grid_size, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        C,
        num_groups,
        eps
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor bias,
    int num_groups,
    float eps);
"""

# JIT compile the custom CUDA kernel
fused_op = load_inline(
    name="fused_op_v2", # Use a unique name to avoid caching issues
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=["-std=c++17"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses Group Normalization, an additional Bias addition, 
    and a Min reduction across features into a single custom CUDA kernel.
    The nn.Linear operation remains unchanged as it is already highly optimized.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        
        # Parameters for the fused operation, equivalent to nn.GroupNorm and the extra bias
        self.gn_weight = nn.Parameter(torch.ones(out_features))  # gamma
        self.gn_bias = nn.Parameter(torch.zeros(out_features)) # beta
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        self.num_groups = num_groups
        self.eps = 1e-5  # Default eps for GroupNorm

    def forward(self, x):
        # Step 1: GEMM (handled by highly optimized cuBLAS via torch.nn.Linear)
        x = self.gemm(x)
        
        # The original model's bias shape (1, C, 1, 1) is ambiguous for a 2D tensor.
        # We squeeze it to (C,) which is a reasonable interpretation for adding a
        # bias vector to the features before the reduction.
        squeezed_bias = self.bias.squeeze()
        
        # Step 2: Fused GroupNorm + BiasAdd + MinReduction
        return fused_op.fused_op_cuda(
            x, self.gn_weight, self.gn_bias, squeezed_bias, self.num_groups, self.eps
        )