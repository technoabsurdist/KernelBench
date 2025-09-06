import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Helper function for block-level reduction using shared memory
template <typename T>
__device__ __forceinline__ T block_reduce_sum(T val, T* shared_mem) {
    int tid = threadIdx.x;
    shared_mem[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    return shared_mem[0];
}

__global__ void GroupNormForwardKernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    const int N,
    const int C,
    const int HxW,
    const int num_groups,
    const float eps) {

    // Shared memory for reduction. We need space for two reductions (sum and sum_sq).
    extern __shared__ float shared_data[];
    float* s_sum = shared_data;
    float* s_sum_sq = &shared_data[blockDim.x];

    const int group_idx = blockIdx.x;
    const int C_per_group = C / num_groups;
    const int group_size = C_per_group * HxW;

    // Identify which batch and group this block is responsible for
    const int n = group_idx / num_groups;
    const int g = group_idx % num_groups;

    // --- 1. Calculate sum and sum of squares for the group ---
    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;

    const int start_channel_in_batch = g * C_per_group;
    const int start_offset_global = n * C * HxW + start_channel_in_batch * HxW;

    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        const float val = x[start_offset_global + i];
        thread_sum += val;
        thread_sum_sq += val * val;
    }

    // Reduce sums across the block
    const float total_sum = block_reduce_sum(thread_sum, s_sum);
    const float total_sum_sq = block_reduce_sum(thread_sum_sq, s_sum_sq);

    // --- 2. Calculate mean and rstd (inverse standard deviation) ---
    // Only thread 0 does this calculation and broadcasts the result via shared memory
    float mean, rstd;
    if (threadIdx.x == 0) {
        mean = total_sum / group_size;
        float var = total_sum_sq / group_size - mean * mean;
        rstd = rsqrtf(var + eps);

        // Store mean and rstd in shared memory to broadcast to other threads
        s_sum[0] = mean;
        s_sum_sq[0] = rstd;
    }

    // Synchronize to make sure all threads see the calculated mean and rstd
    __syncthreads();

    // Load broadcasted mean and rstd from shared memory
    mean = s_sum[0];
    rstd = s_sum_sq[0];

    // --- 3. Apply normalization, scale, and shift ---
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        const int c_local = i / HxW;
        const int c_global = start_channel_in_batch + c_local;
        const int global_idx = start_offset_global + i;

        const float val = x[global_idx];
        const float normalized_val = (val - mean) * rstd;
        y[global_idx] = normalized_val * weight[c_global] + bias[c_global];
    }
}

torch::Tensor group_norm_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {

    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Input weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Input bias must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input x must be a float32 tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "Input weight must be a float32 tensor");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "Input bias must be a float32 tensor");


    const int N = x.size(0);
    const int C = x.size(1);
    const int HxW = x.numel() / (N * C);

    TORCH_CHECK(C % num_groups == 0, "Number of channels must be divisible by num_groups");

    auto y = torch::empty_like(x);

    const int threads_per_block = 512;
    const int blocks_per_grid = N * num_groups;

    // Shared memory size: 2 reductions * threads_per_block * sizeof(float)
    const int shared_mem_size = 2 * threads_per_block * sizeof(float);

    GroupNormForwardKernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        N,
        C,
        HxW,
        num_groups,
        static_cast<float>(eps)
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return y;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps);
"""

# JIT compile the custom CUDA kernel
group_norm_impl = load_inline(
    name="group_norm_impl",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_forward_cuda"],
    verbose=False,
)

class GroupNormCuda(nn.Module):
    """
    Custom Group Normalization module using a fused CUDA kernel.
    """
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super(GroupNormCuda, self).__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")
        
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            return group_norm_impl.group_norm_forward_cuda(x, self.weight, self.bias, self.num_groups, self.eps)
        else:
            # If not affine, create non-learnable tensors for weight and bias
            weight = torch.ones(self.num_channels, dtype=x.dtype, device=x.device)
            bias = torch.zeros(self.num_channels, dtype=x.dtype, device=x.device)
            return group_norm_impl.group_norm_forward_cuda(x, weight, bias, self.num_groups, self.eps)

    def extra_repr(self) -> str:
        return f'{self.num_channels}, num_groups={self.num_groups}, eps={self.eps}, affine={self.affine}'


class ModelNew(nn.Module):
    """
    Simple model that performs Group Normalization using a custom fused CUDA kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the custom GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.gn = GroupNormCuda(num_groups=num_groups, num_channels=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return self.gn(x)