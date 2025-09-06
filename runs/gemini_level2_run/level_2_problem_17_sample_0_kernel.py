import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for the fused kernel that performs Instance Normalization and division
fused_instance_norm_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_instance_norm_div_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    float epsilon,
    float divide_by) {

    // Each block processes one channel of one batch item.
    // Grid dimension is (N, C).
    const int batch_idx = blockIdx.x;
    const int channel_idx = blockIdx.y;

    // Total number of pixels per channel
    const int image_size = H * W;

    // Pointer to the start of the current channel's data
    const float* x = input + (batch_idx * C + channel_idx) * image_size;
    float* y = output + (batch_idx * C + channel_idx) * image_size;

    // Use shared memory for reduction
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sum_sq = &sdata[blockDim.x];

    // Step 1: Calculate sum and sum of squares in parallel
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    for (int i = threadIdx.x; i < image_size; i += blockDim.x) {
        float val = x[i];
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Reduce sum and sum_sq across the block
    s_sum[threadIdx.x] = local_sum;
    s_sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 computes mean and variance
    if (threadIdx.x == 0) {
        float sum = s_sum[0];
        float sum_sq = s_sum_sq[0];
        float mean = sum / image_size;
        float var = (sum_sq / image_size) - (mean * mean);
        
        // Store mean and inv_std_dev in shared memory for all threads to access
        s_sum[0] = mean;
        s_sum_sq[0] = rsqrtf(var + epsilon);
    }
    __syncthreads();

    // Get the calculated mean and inv_std_dev
    float mean = s_sum[0];
    float inv_std_dev = s_sum_sq[0];
    float inv_divisor = 1.0f / divide_by;

    // Step 2: Apply normalization and division
    for (int i = threadIdx.x; i < image_size; i += blockDim.x) {
        y[i] = (x[i] - mean) * inv_std_dev * inv_divisor;
    }
}

torch::Tensor fused_instance_norm_div_cuda(
    torch::Tensor input,
    double divide_by,
    double epsilon) {

    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input tensor must be 4-dimensional (N, C, H, W)");
    // The kernel requires contiguous memory for correct indexing
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);

    auto output = torch::empty_like(input);

    const int threads_per_block = 256;
    const dim3 blocks(N, C);

    // Shared memory size: 2 * threads_per_block * sizeof(float)
    const int shared_mem_size = 2 * threads_per_block * sizeof(float);

    fused_instance_norm_div_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        static_cast<float>(epsilon),
        static_cast<float>(divide_by)
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

# C++ source for function signature
fused_instance_norm_div_cpp_source = """
torch::Tensor fused_instance_norm_div_cuda(torch::Tensor input, double divide_by, double epsilon);
"""

# JIT compile the CUDA kernel. This is done once when the module is imported.
fused_instance_norm_div = load_inline(
    name="fused_instance_norm_div",
    cpp_sources=fused_instance_norm_div_cpp_source,
    cuda_sources=fused_instance_norm_div_source,
    functions=["fused_instance_norm_div_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses Instance Normalization and division into a single CUDA kernel.
    The convolution operation remains the standard PyTorch implementation, as it is already
    highly optimized in libraries like cuDNN.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        # Keep the original, highly optimized convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
        # Store parameters for the custom kernel
        self.divide_by = divide_by
        # Use the default epsilon from nn.InstanceNorm2d for numerical stability
        self.epsilon = 1e-5

        # Assign the compiled custom operator
        self.fused_op = fused_instance_norm_div

    def forward(self, x):
        # 1. Apply the standard convolution
        x = self.conv(x)
        
        # 2. Apply the fused custom kernel for InstanceNorm + Division
        # Ensure input is contiguous for the custom kernel's memory access pattern
        x = self.fused_op.fused_instance_norm_div_cuda(x.contiguous(), self.divide_by, self.epsilon)
        return x