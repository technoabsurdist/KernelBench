import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel source for the fused operation
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel to compute the mean over spatial dimensions (D, H, W) for each batch and channel.
// Each block computes the mean for one (N, C) slice.
__global__ void spatial_mean_kernel(
    const float* __restrict__ x,
    float* __restrict__ mean_out,
    const int N,
    const int C,
    const int D,
    const int H,
    const int W) {

    // Shared memory for reduction within a block
    extern __shared__ float sdata[];

    const int spatial_size = D * H * W;
    const int tid = threadIdx.x;
    const int block_id = blockIdx.x; // Corresponds to the n*C + c index

    // Each block processes one channel for one batch item
    if (block_id >= N * C) return;

    // Pointer to the start of the current spatial slice
    const float* x_slice = x + block_id * spatial_size;

    float sum = 0.0f;
    // Each thread sums up a portion of the spatial volume in a grid-stride loop
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        sum += x_slice[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final mean to global memory
    if (tid == 0) {
        mean_out[block_id] = sdata[0] / spatial_size;
    }
}

// Element-wise kernel to apply the final transformation:
// out = scale * (x - spatial_mean)
__global__ void apply_transform_kernel(
    const float* __restrict__ x,
    const float* __restrict__ spatial_mean,
    const float* __restrict__ scale,
    float* __restrict__ out,
    const int total_elements,
    const int C,
    const int D,
    const int H,
    const int W) {

    const int spatial_size = D * H * W;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < total_elements) {
        // Find the corresponding channel index and (batch, channel) index
        const int c_idx = (i / spatial_size) % C;
        const int nc_idx = i / spatial_size; // Index for the (N, C) mean tensor

        out[i] = scale[c_idx] * (x[i] - spatial_mean[nc_idx]);
    }
}

// C++ function to orchestrate the kernel launches
torch::Tensor fused_bn_mean_sub_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor running_var,
    double epsilon) {

    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 5, "Input tensor must be 5D");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto D = x.size(2);
    const auto H = x.size(3);
    const auto W = x.size(4);
    const auto total_elements = x.numel();

    // Pre-calculate the scale factor: scale = gamma / sqrt(running_var + eps)
    // This is the core of the fusion optimization.
    auto scale = gamma / torch::sqrt(running_var + epsilon);

    // Allocate output and intermediate tensors
    auto spatial_mean = torch::empty({N, C}, x.options());
    auto out = torch::empty_like(x);

    // --- Launch Mean Kernel ---
    const int num_blocks_mean = N * C;
    // Using 512 threads is a good choice for reduction on modern GPUs
    const int threads_per_block_mean = 512;
    const int shared_mem_size = threads_per_block_mean * sizeof(float);

    spatial_mean_kernel<<<num_blocks_mean, threads_per_block_mean, shared_mem_size>>>(
        x.data_ptr<float>(),
        spatial_mean.data_ptr<float>(),
        N, C, D, H, W
    );

    // --- Launch Transform Kernel ---
    const int threads_per_block_transform = 256;
    const int num_blocks_transform = (total_elements + threads_per_block_transform - 1) / threads_per_block_transform;

    apply_transform_kernel<<<num_blocks_transform, threads_per_block_transform>>>(
        x.data_ptr<float>(),
        spatial_mean.data_ptr<float>(),
        scale.data_ptr<float>(),
        out.data_ptr<float>(),
        total_elements,
        C, D, H, W
    );

    return out;
}
"""

# C++ source for the function signature
fused_op_cpp_source = """
torch::Tensor fused_bn_mean_sub_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor running_var,
    double epsilon);
"""

# Compile the inline CUDA code
# This fuses BatchNorm3d (inference mode) and the subsequent spatial mean subtraction
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_bn_mean_sub_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    An optimized version of the model where the BatchNorm and mean subtraction
    are fused into a single custom CUDA operation for faster inference.
    The ConvTranspose3d layer is kept as is, as its cuDNN implementation is highly optimized.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        # We still need the batch_norm layer to hold the parameters (weight, bias)
        # and buffers (running_mean, running_var)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        # The custom fused operator
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)

        # The custom kernel is designed for inference mode.
        # During training, BatchNorm updates its running stats, which our kernel doesn't do.
        # Therefore, we fall back to the original PyTorch implementation for training.
        if self.training:
            x = self.batch_norm(x)
            x = x - torch.mean(x, dim=(2, 3, 4), keepdim=True)
        else:
            # In inference mode, we use the highly optimized fused kernel.
            # The mathematical simplification shows that beta and running_mean are cancelled out,
            # so they are not passed to the kernel.
            x = self.fused_op.fused_bn_mean_sub_cuda(
                x,
                self.batch_norm.weight,
                self.batch_norm.running_var,
                self.batch_norm.eps
            )
        return x