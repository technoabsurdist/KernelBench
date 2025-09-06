import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Instance Normalization
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// A standard block-level reduction helper function using shared memory
__device__ void block_reduce_sum(float* sdata, float& val) {
    int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    val = sdata[0];
}

// Kernel to compute mean and inverse standard deviation for each instance and channel
__global__ void instance_norm_stats_kernel(
    const float* x,
    float* mean,
    float* inv_std,
    int N, int C, int H, int W,
    float epsilon) {

    // Each block processes one (N, C) slice
    int n = blockIdx.x;
    int c = blockIdx.y;
    int instance_idx = n * C + c;

    int image_size = H * W;
    const float* data = x + instance_idx * image_size;

    // Use shared memory for reduction
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // Step 1: Compute sum
    float sum = 0.0f;
    for (int i = tid; i < image_size; i += blockDim.x) {
        sum += data[i];
    }
    block_reduce_sum(sdata, sum);

    // Thread 0 computes mean, writes to global and shared memory for broadcasting
    if (tid == 0) {
        float mu = sum / image_size;
        mean[instance_idx] = mu;
        sdata[0] = mu;
    }
    __syncthreads();

    // All threads read the mean from shared memory
    float mu = sdata[0];

    // Step 2: Compute sum of squared differences
    float sum_sq_diff = 0.0f;
    for (int i = tid; i < image_size; i += blockDim.x) {
        float diff = data[i] - mu;
        sum_sq_diff += diff * diff;
    }
    block_reduce_sum(sdata, sum_sq_diff);

    // Thread 0 computes inv_std
    if (tid == 0) {
        float var = sum_sq_diff / image_size;
        inv_std[instance_idx] = rsqrtf(var + epsilon);
    }
}

// Kernel to apply the normalization using the computed stats
__global__ void instance_norm_apply_kernel(
    const float* x,
    const float* mean,
    const float* inv_std,
    const float* gamma,
    const float* beta,
    float* y,
    int N, int C, int H, int W) {

    int image_size = H * W;
    int total_elements = N * C * image_size;

    // Use a grid-stride loop to process all elements
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += blockDim.x * gridDim.x) {

        int c = (idx / image_size) % C;
        int n = idx / (C * image_size);
        int instance_idx = n * C + c;

        float mu = mean[instance_idx];
        float rsigma = inv_std[instance_idx];
        float g = gamma[c];
        float b = beta[c];

        y[idx] = (x[idx] - mu) * rsigma * g + b;
    }
}

torch::Tensor instance_norm_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double epsilon) {

    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "Gamma tensor must be contiguous");
    TORCH_CHECK(beta.is_contiguous(), "Beta tensor must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor");

    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);

    auto y = torch::empty_like(x);
    auto mean = torch::empty({N, C}, x.options());
    auto inv_std = torch::empty({N, C}, x.options());

    const int block_size_stats = 256;
    const int shared_mem_size = block_size_stats * sizeof(float);

    // Launch stats kernel
    dim3 grid_stats(N, C);
    dim3 block_stats(block_size_stats);
    instance_norm_stats_kernel<<<grid_stats, block_stats, shared_mem_size>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        N, C, H, W,
        static_cast<float>(epsilon)
    );

    // Launch apply kernel
    const int total_elements = N * C * H * W;
    const int block_size_apply = 256;
    const int num_blocks_apply = (total_elements + block_size_apply - 1) / block_size_apply;
    dim3 grid_apply(num_blocks_apply);
    dim3 block_apply(block_size_apply);
    instance_norm_apply_kernel<<<grid_apply, block_apply>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W
    );

    return y;
}
"""

instance_norm_cpp_source = """
torch::Tensor instance_norm_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double epsilon);
"""

# JIT compile the custom CUDA kernel
instance_norm_impl = load_inline(
    name="instance_norm_impl",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Model that performs Instance Normalization using a custom CUDA kernel.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the custom InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float): A value added to the denominator for numerical stability.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

        # These are the learnable affine parameters, matching nn.InstanceNorm2d
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies custom Instance Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return instance_norm_impl.instance_norm_cuda(x, self.weight, self.bias, self.eps)