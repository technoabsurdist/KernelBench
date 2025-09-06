import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for BatchNorm2d
bn_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Device function for block-level reduction using shared memory
template <typename T>
__device__ void block_reduce_sum(T* x) {
    // x is a shared memory array of size blockDim.x
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        __syncthreads();
        if (threadIdx.x < offset) {
            x[threadIdx.x] += x[threadIdx.x + offset];
        }
    }
}

// Kernel to calculate sum and sum of squares for each channel
// GridDim.x should be C (num_features)
// BlockDim.x should be a power of 2, e.g., 512 or 1024
__global__ void calculate_stats_kernel(
    const float* __restrict__ x,
    float* __restrict__ sum,
    float* __restrict__ sum_sq,
    int N, int C, int H, int W) {

    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sum_sq = &sdata[blockDim.x];

    int c = blockIdx.x; // This block handles channel c
    int plane_size = H * W;
    int channel_size = N * plane_size;

    // Each thread computes a partial sum and sum_sq
    float my_sum = 0.0f;
    float my_sum_sq = 0.0f;

    // Stride loop over all elements in the channel for this block
    for (int i = threadIdx.x; i < channel_size; i += blockDim.x) {
        int n = i / plane_size;
        int plane_idx = i % plane_size;
        int data_idx = n * C * plane_size + c * plane_size + plane_idx;
        float val = x[data_idx];
        my_sum += val;
        my_sum_sq += val * val;
    }

    s_sum[threadIdx.x] = my_sum;
    s_sum_sq[threadIdx.x] = my_sum_sq;
    __syncthreads();

    // Block-level reduction
    block_reduce_sum(s_sum);
    block_reduce_sum(s_sum_sq);

    // Thread 0 writes the final result for this channel
    if (threadIdx.x == 0) {
        sum[c] = s_sum[0];
        sum_sq[c] = s_sum_sq[0];
    }
}


// Kernel to apply the normalization: y = gamma * (x - mean) * invstd + beta
// Grid is 1D over all elements
__global__ void normalize_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ mean,
    const float* __restrict__ invstd,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C, int H, int W) {

    int plane_size = H * W;
    int total_size = N * C * plane_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_size) {
        int c = (idx / plane_size) % C; // Get channel index

        // Fetch channel-specific parameters
        float m = mean[c];
        float inv = invstd[c];
        float w = weight[c];
        float b = bias[c];

        // Apply normalization
        y[idx] = w * (x[idx] - m) * inv + b;
    }
}

// C++ wrapper function that dispatches to the correct kernels
torch::Tensor batch_norm_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    double momentum,
    double eps) {

    // Ensure inputs are contiguous and on the correct device
    x = x.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();
    running_mean = running_mean.contiguous();
    running_var = running_var.contiguous();

    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);

    auto out = torch::empty_like(x);
    auto options = x.options();

    const int block_size_stats = 512;
    const int block_size_norm = 256;

    if (training) {
        // Allocate tensors for batch stats
        auto batch_mean = torch::empty({C}, options);
        auto batch_var = torch::empty({C}, options);
        auto invstd = torch::empty({C}, options);

        // Allocate intermediate tensors for reduction
        auto sum = torch::zeros({C}, options);
        auto sum_sq = torch::zeros({C}, options);

        // Launch kernel to calculate sum and sum_sq
        // Shared memory size: 2 * block_size * sizeof(float)
        const int shared_mem_size = 2 * block_size_stats * sizeof(float);
        calculate_stats_kernel<<<C, block_size_stats, shared_mem_size>>>(
            x.data_ptr<float>(),
            sum.data_ptr<float>(),
            sum_sq.data_ptr<float>(),
            N, C, H, W);

        // Use PyTorch's C++ API for the final, low-dimensional calculations.
        // This is efficient for small C and simplifies the CUDA code.
        float count = static_cast<float>(N * H * W);
        batch_mean = sum / count;
        batch_var = sum_sq / count - batch_mean.pow(2);

        // Update running stats in-place
        running_mean.copy_((1.0 - momentum) * running_mean + momentum * batch_mean);
        running_var.copy_((1.0 - momentum) * running_var + momentum * batch_var);

        // Calculate invstd for normalization
        invstd = (batch_var + eps).rsqrt();

        // Launch normalization kernel
        const int num_blocks_norm = (N * C * H * W + block_size_norm - 1) / block_size_norm;
        normalize_kernel<<<num_blocks_norm, block_size_norm>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            batch_mean.data_ptr<float>(),
            invstd.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            N, C, H, W);

    } else { // Inference
        auto invstd = (running_var + eps).rsqrt();

        const int num_blocks_norm = (N * C * H * W + block_size_norm - 1) / block_size_norm;
        normalize_kernel<<<num_blocks_norm, block_size_norm>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            running_mean.data_ptr<float>(),
            invstd.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            N, C, H, W);
    }

    return out;
}
"""

# C++ source for function signature
bn_cpp_source = """
torch::Tensor batch_norm_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    double momentum,
    double eps);
"""

# Compile the inline CUDA code
custom_bn_impl = load_inline(
    name="custom_bn_impl",
    cpp_sources=bn_cpp_source,
    cuda_sources=bn_cuda_source,
    functions=["batch_norm_forward_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Model that performs Batch Normalization using a custom CUDA kernel.
    This module mimics the behavior of nn.BatchNorm2d.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        """
        Initializes the custom BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float): A value added to the denominator for numerical stability.
            momentum (float): The value used for the running_mean and running_var computation.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Parameters: gamma (weight) and beta (bias)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Buffers: running mean and variance
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies custom Batch Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, H, W).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        # The `training` attribute is automatically set by `model.train()` or `model.eval()`.
        return custom_bn_impl.batch_norm_forward_cuda(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.training,
            self.momentum,
            self.eps,
        )

    def extra_repr(self):
        return "{num_features}, eps={eps}, momentum={momentum}".format(**self.__dict__)