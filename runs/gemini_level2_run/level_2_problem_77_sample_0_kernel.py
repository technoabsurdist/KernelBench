import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused scaling, batch normalization, and global average pooling
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_scale_norm_avgpool_kernel(
    const float* __restrict__ x,      // Input tensor (N, C, D, H, W)
    const float* __restrict__ weight,   // Batch norm weight (gamma) (C)
    const float* __restrict__ bias,     // Batch norm bias (beta) (C)
    const float* __restrict__ mean,     // Batch norm running_mean (C)
    const float* __restrict__ var,      // Batch norm running_var (C)
    float* __restrict__ out,            // Output tensor (N, C, 1, 1, 1)
    const float scale_factor,
    const float eps,
    const int N,
    const int C,
    const int D,
    const int H,
    const int W) {

    // Each block computes the average for one (n, c) pair.
    const int n = blockIdx.x;
    const int c = blockIdx.y;

    // Total number of spatial elements to reduce over
    const long long spatial_size = (long long)D * H * W;
    if (spatial_size == 0) return;
    const float inv_spatial_size = 1.0f / spatial_size;

    // Load batch norm parameters for this channel
    const float c_weight = weight[c];
    const float c_bias = bias[c];
    const float c_mean = mean[c];
    const float c_var = var[c];

    // Pre-calculate the normalization scale and shift for a fused operation.
    // y = (x * scale_factor * norm_scale) + norm_shift
    // where norm_scale = gamma / sqrt(var + eps)
    // and norm_shift = beta - mean * norm_scale
    const float inv_std = rsqrtf(c_var + eps);
    const float norm_scale = c_weight * inv_std;
    const float norm_shift = c_bias - c_mean * norm_scale;

    // Pointer to the start of the data for this (n, c) slice
    const float* x_ptr = x + n * C * spatial_size + c * spatial_size;

    // Use shared memory for reduction
    extern __shared__ float sdata[];
    
    // Each thread computes a partial sum
    float my_sum = 0.0f;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    for (long long i = tid; i < spatial_size; i += block_size) {
        // 1. Scale
        float val = x_ptr[i] * scale_factor;
        // 2. Batch Norm (fused)
        val = val * norm_scale + norm_shift;
        // 3. Accumulate for pooling
        my_sum += val;
    }

    sdata[tid] = my_sum;
    __syncthreads();

    // Perform reduction in shared memory. Assumes block_size is a power of 2.
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // The first thread in the block writes the final result
    if (tid == 0) {
        // The final sum is in sdata[0]. Divide by spatial size to get the average.
        out[n * C + c] = sdata[0] * inv_spatial_size;
    }
}

torch::Tensor fused_scale_norm_avgpool_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mean,
    torch::Tensor var,
    double scale_factor,
    double eps) {

    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 5, "Input x must be a 5D tensor");
    TORCH_CHECK(weight.is_cuda() && bias.is_cuda() && mean.is_cuda() && var.is_cuda(), "BN params must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Only Float32 is supported");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    const int N = x.size(0);
    const int C = x.size(1);
    const int D = x.size(2);
    const int H = x.size(3);
    const int W = x.size(4);

    auto out = torch::empty({N, C, 1, 1, 1}, x.options());

    // Choose a block size, preferably a power of 2. 512 is a good default.
    const int block_size = 512;
    const dim3 grid_dim(N, C);
    const dim3 block_dim(block_size);
    
    // Shared memory size: one float per thread in the block
    const int shared_mem_size = block_size * sizeof(float);

    fused_scale_norm_avgpool_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<float>(scale_factor),
        static_cast<float>(eps),
        N, C, D, H, W
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# C++ source for function signature binding
fused_op_cpp_source = """
torch::Tensor fused_scale_norm_avgpool_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mean,
    torch::Tensor var,
    double scale_factor,
    double eps);
"""

# JIT compile the CUDA kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_scale_norm_avgpool_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses scaling, batch normalization (inference), and global average pooling
    into a single custom CUDA kernel. This provides a significant speedup by reducing kernel launch
    overhead and, more importantly, by avoiding the need to write large intermediate tensors
    (after scaling and after batch norm) to global memory.

    This model is designed for inference. To use it, first train the original `Model`,
    then create an instance of `ModelNew` and load the state_dict from the trained `Model`.
    """

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # The ConvTranspose3d layer is kept as a standard PyTorch module,
        # as its implementation (cuDNN) is already highly optimized.
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)

        # This BatchNorm3d layer is NOT used directly in the forward pass. Instead, it serves
        # as a convenient container for the batch norm parameters (weight, bias, running_mean,
        # running_var). When loading a state_dict from a trained `Model`, these parameters
        # will be populated and then passed to our custom kernel.
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)

        self.scale_factor = scale_factor
        self.fused_op = fused_op

    def forward(self, x):
        # 1. Perform the 3D transposed convolution using the standard PyTorch operator.
        x = self.conv_transpose(x)

        # 2. Call the custom CUDA kernel to perform the fused operation:
        #    - Element-wise scaling (x * scale_factor)
        #    - Batch normalization (using running mean/var for inference)
        #    - Global average pooling
        # This single kernel replaces three separate operations from the original model.
        x = self.fused_op.fused_scale_norm_avgpool_cuda(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.scale_factor,
            self.batch_norm.eps,
        )
        return x