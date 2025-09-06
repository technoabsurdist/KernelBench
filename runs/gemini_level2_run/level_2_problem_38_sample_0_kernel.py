import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: clamp + spatial_softmax + scale
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat> // For FLT_MAX

// Device function for block-level reduction for sum using shared memory
__device__ void block_reduce_sum(volatile float* sdata, int tid) {
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
}

// Device function for block-level reduction for max using shared memory
__device__ void block_reduce_max(volatile float* sdata, int tid) {
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
}

__global__ void fused_clamp_softmax_scale_kernel(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    float* __restrict__ output,
    const float clamp_min,
    const float clamp_max,
    const int B,
    const int C,
    const int N) {

    // N is the size of the spatial dimensions (D * H * W)
    // Each block handles one spatial slice (one (b, c) pair)
    const int block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Get batch and channel index for this block
    const int c = block_idx % C;
    const int b = block_idx / C;

    // Pointers to the start of the data for this slice
    const float* input_slice = input + b * C * N + c * N;
    float* output_slice = output + b * C * N + c * N;

    extern __shared__ float sdata[];

    // --- Pass 1: Find max value in the slice ---
    float thread_max = -FLT_MAX;
    for (int i = tid; i < N; i += block_size) {
        float val = input_slice[i];
        float clamped_val = fmaxf(clamp_min, fminf(val, clamp_max));
        thread_max = fmaxf(thread_max, clamped_val);
    }
    sdata[tid] = thread_max;
    block_reduce_max(sdata, tid);
    const float block_max = sdata[0];

    // --- Pass 2: Calculate sum of exponentials ---
    float thread_sum = 0.0f;
    for (int i = tid; i < N; i += block_size) {
        float val = input_slice[i];
        float clamped_val = fmaxf(clamp_min, fminf(val, clamp_max));
        thread_sum += expf(clamped_val - block_max);
    }
    sdata[tid] = thread_sum;
    block_reduce_sum(sdata, tid);
    const float block_sum = sdata[0];

    // --- Pass 3: Calculate final output and write to global memory ---
    const float scale_val = scale[c];
    // Add a small epsilon to avoid division by zero
    const float inv_sum = 1.0f / (block_sum + 1e-12f);

    for (int i = tid; i < N; i += block_size) {
        float val = input_slice[i];
        float clamped_val = fmaxf(clamp_min, fminf(val, clamp_max));
        output_slice[i] = expf(clamped_val - block_max) * inv_sum * scale_val;
    }
}

torch::Tensor fused_clamp_softmax_scale_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    const float clamp_min,
    const float clamp_max) {

    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "Scale must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(scale.is_contiguous(), "Scale must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "Scale must be a float32 tensor");
    TORCH_CHECK(input.dim() == 5, "Input must be a 5D tensor");
    TORCH_CHECK(scale.dim() == 1, "Scale must be a 1D tensor");
    TORCH_CHECK(input.size(1) == scale.size(0), "Input channels must match scale size");


    const auto B = input.size(0);
    const auto C = input.size(1);
    const auto D = input.size(2);
    const auto H = input.size(3);
    const auto W = input.size(4);
    const auto N = D * H * W;

    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = B * C;
    const int shared_mem_size = block_size * sizeof(float);

    fused_clamp_softmax_scale_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        clamp_min,
        clamp_max,
        B,
        C,
        N
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

# C++ source for the wrapper function signature
fused_op_cpp_source = """
torch::Tensor fused_clamp_softmax_scale_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    const float clamp_min,
    const float clamp_max);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_clamp_softmax_scale_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Model that performs average pooling, 3D transposed convolution, and a fused
    operation for clamping, spatial softmax, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        # The custom kernel expects a 1D scale tensor of size `out_channels`.
        self.scale = nn.Parameter(torch.ones(out_channels))
        self.fused_op = fused_op.fused_clamp_softmax_scale_cuda

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth, height, width).
        """
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        
        # Call the fused CUDA kernel to replace clamp, reshape, softmax, reshape, and scale operations.
        x = self.fused_op(x, self.scale, self.clamp_min, self.clamp_max)
        return x