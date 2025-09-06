import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA source code for the fused operators
fused_ops_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat> // For FLT_MAX

// Kernel for Fused (Scalar Multiplication + MaxPool3D)
// Each thread computes one output element.
__global__ void fused_scale_maxpool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float scale,
    const int num_outputs,
    // Input dims
    const int channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    // Output dims
    const int out_depth,
    const int out_height,
    const int out_width,
    // Pool params
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= num_outputs) return;

    // De-flatten index to get (n, c, od, oh, ow)
    int temp_idx = index;
    const int ow = temp_idx % out_width;
    temp_idx /= out_width;
    const int oh = temp_idx % out_height;
    temp_idx /= out_height;
    const int od = temp_idx % out_depth;
    temp_idx /= out_depth;
    const int c = temp_idx % channels;
    const int n = temp_idx / channels;

    // Find top-left corner of pooling window in input
    const int id_start = od * stride_d;
    const int ih_start = oh * stride_h;
    const int iw_start = ow * stride_w;

    float max_val = -FLT_MAX;

    const int in_chw = in_depth * in_height * in_width;
    const int in_hw = in_height * in_width;
    const int input_nc_offset = n * channels * in_chw + c * in_chw;

    // Loop over pooling window
    for (int kd = 0; kd < kernel_d; ++kd) {
        const int id = id_start + kd;
        if (id >= in_depth) continue;
        for (int kh = 0; kh < kernel_h; ++kh) {
            const int ih = ih_start + kh;
            if (ih >= in_height) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int iw = iw_start + kw;
                if (iw < in_width) {
                    const int input_idx = input_nc_offset + id * in_hw + ih * in_width + iw;
                    max_val = fmaxf(max_val, input[input_idx] * scale);
                }
            }
        }
    }
    output[index] = max_val;
}

// Kernel for Fused (Global Average Pool + Clamp)
// Each block computes the reduction for one (batch, channel) pair.
__global__ void fused_global_avg_pool_clamp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float clamp_min,
    const float clamp_max,
    const int spatial_size // D * H * W
) {
    const int nc_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;
    const int block_dim = blockDim.x;

    extern __shared__ float sdata[];

    float my_sum = 0.0f;
    const int input_offset = nc_idx * spatial_size;

    // Each thread computes a partial sum from global memory
    for (int i = thread_idx; i < spatial_size; i += block_dim) {
        my_sum += input[input_offset + i];
    }
    sdata[thread_idx] = my_sum;
    __syncthreads();

    // In-place reduction in shared memory
    for (unsigned int s = block_dim / 2; s > 0; s >>= 1) {
        if (thread_idx < s) {
            sdata[thread_idx] += sdata[thread_idx + s];
        }
        __syncthreads();
    }

    // First thread writes the final result after averaging and clamping
    if (thread_idx == 0) {
        float total_sum = sdata[0];
        float avg = total_sum / (float)spatial_size;
        output[nc_idx] = fmaxf(clamp_min, fminf(avg, clamp_max));
    }
}

// C++ Wrapper for Fused (Scalar Multiplication + MaxPool3D)
torch::Tensor fused_scale_maxpool_cuda(
    torch::Tensor input,
    float scale,
    int kernel_size,
    int stride
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(input.dim() == 5, "Input must be a 5D tensor");

    const int kernel_d = kernel_size;
    const int kernel_h = kernel_size;
    const int kernel_w = kernel_size;
    const int stride_d = stride;
    const int stride_h = stride;
    const int stride_w = stride;

    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto in_depth = input.size(2);
    const auto in_height = input.size(3);
    const auto in_width = input.size(4);

    const auto out_depth = (in_depth - kernel_d) / stride_d + 1;
    const auto out_height = (in_height - kernel_h) / stride_h + 1;
    const auto out_width = (in_width - kernel_w) / stride_w + 1;

    auto output = torch::empty({batch_size, channels, out_depth, out_height, out_width}, input.options());
    const int num_outputs = output.numel();
    if (num_outputs == 0) return output;

    const int block_size = 256;
    const int num_blocks = (num_outputs + block_size - 1) / block_size;

    fused_scale_maxpool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scale,
        num_outputs,
        channels, in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w
    );
    return output;
}

// C++ Wrapper for Fused (Global Average Pool + Clamp)
torch::Tensor fused_global_avg_pool_clamp_cuda(
    torch::Tensor input,
    float clamp_min,
    float clamp_max
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(input.dim() == 5, "Input must be a 5D tensor");

    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto in_depth = input.size(2);
    const auto in_height = input.size(3);
    const auto in_width = input.size(4);
    const auto spatial_size = in_depth * in_height * in_width;

    auto output = torch::empty({batch_size, channels, 1, 1, 1}, input.options());
    const int num_outputs = batch_size * channels;
    if (num_outputs == 0) return output;

    const int num_blocks = num_outputs;
    const int block_size = 256; // Must be a power of 2 for the reduction
    const size_t shared_mem_size = block_size * sizeof(float);

    fused_global_avg_pool_clamp_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        clamp_min,
        clamp_max,
        spatial_size
    );
    return output;
}
"""

# Define the C++ source for function signatures
fused_ops_cpp_source = """
torch::Tensor fused_scale_maxpool_cuda(torch::Tensor input, float scale, int kernel_size, int stride);
torch::Tensor fused_global_avg_pool_clamp_cuda(torch::Tensor input, float clamp_min, float clamp_max);
"""

# Compile the inline CUDA code using torch.utils.cpp_extension
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_cuda_source,
    functions=["fused_scale_maxpool_cuda", "fused_global_avg_pool_clamp_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that uses custom CUDA kernels to fuse operations.
    - Fuses (scalar multiplication + MaxPool3D) into one kernel.
    - Fuses (Global Average Pool + Clamp) into another kernel.
    - Keeps the original ConvTranspose3d as it's highly optimized by cuDNN.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        # 1. Keep the complex ConvTranspose3d layer from PyTorch
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        
        # 2. Store parameters for our custom kernels
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size
        # In nn.MaxPool3d, if stride is not specified, it defaults to kernel_size
        self.maxpool_stride = maxpool_kernel_size
        self.clamp_min = 0.0
        self.clamp_max = 1.0

    def forward(self, x):
        # Step 1: Use the standard, highly-optimized ConvTranspose3d
        x = self.conv_transpose(x)
        
        # Step 2: Apply the first fused kernel: (x * scale) followed by MaxPool3d
        x = fused_ops.fused_scale_maxpool_cuda(x, self.scale, self.maxpool_kernel_size, self.maxpool_stride)
        
        # Step 3: Apply the second fused kernel: GlobalAvgPool followed by Clamp
        x = fused_ops.fused_global_avg_pool_clamp_cuda(x, self.clamp_min, self.clamp_max)
        
        return x