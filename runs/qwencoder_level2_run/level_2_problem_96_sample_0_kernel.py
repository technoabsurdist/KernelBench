import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations: scale + maxpool + global_avg_pool + clamp
fused_operations_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_depth,
    int input_height,
    int input_width,
    int pool_kernel_d,
    int pool_kernel_h,
    int pool_kernel_w,
    float scale,
    float clamp_min,
    float clamp_max
) {
    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int output_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || channel_idx >= channels) return;
    
    // Max pooling dimensions
    int pooled_depth = (input_depth + pool_kernel_d - 1) / pool_kernel_d;
    int pooled_height = (input_height + pool_kernel_h - 1) / pool_kernel_h;
    int pooled_width = (input_width + pool_kernel_w - 1) / pool_kernel_w;
    
    if (output_idx >= pooled_depth * pooled_height * pooled_width) return;
    
    // Calculate 3D indices in pooled output
    int pd = output_idx / (pooled_height * pooled_width);
    int ph = (output_idx % (pooled_height * pooled_width)) / pooled_width;
    int pw = output_idx % pooled_width;
    
    // Calculate corresponding input region for max pooling
    int start_d = pd * pool_kernel_d;
    int end_d = min(start_d + pool_kernel_d, input_depth);
    int start_h = ph * pool_kernel_h;
    int end_h = min(start_h + pool_kernel_h, input_height);
    int start_w = pw * pool_kernel_w;
    int end_w = min(start_w + pool_kernel_w, input_width);
    
    // Find maximum in the pooling region
    float max_val = -1e30f;
    for (int d = start_d; d < end_d; d++) {
        for (int h = start_h; h < end_h; h++) {
            for (int w = start_w; w < end_w; w++) {
                int input_idx = ((batch_idx * channels + channel_idx) * input_depth + d) * input_height * input_width +
                                h * input_width + w;
                float val = input[input_idx] * scale;
                if (val > max_val) max_val = val;
            }
        }
    }
    
    // Clamp the max value
    if (max_val < clamp_min) max_val = clamp_min;
    if (max_val > clamp_max) max_val = clamp_max;
    
    // Write to output (global average pooling will be done by summing later)
    int out_idx = batch_idx * channels + channel_idx;
    atomicAdd(&output[out_idx], max_val);
}

torch::Tensor fused_operations_cuda(
    torch::Tensor input,
    int pool_kernel_d,
    int pool_kernel_h,
    int pool_kernel_w,
    float scale,
    float clamp_min,
    float clamp_max
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto input_depth = input.size(2);
    auto input_height = input.size(3);
    auto input_width = input.size(4);
    
    // Output tensor for global average pooling result
    auto output = torch::zeros({batch_size, channels, 1, 1, 1}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    // Grid and block dimensions
    dim3 grid(batch_size, channels);
    dim3 block(min(pool_kernel_d * pool_kernel_h * pool_kernel_w, 1024));
    
    // Initialize output to zero
    cudaMemset(output.data_ptr<float>(), 0, output.numel() * sizeof(float));
    
    fused_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_depth,
        input_height,
        input_width,
        pool_kernel_d,
        pool_kernel_h,
        pool_kernel_w,
        scale,
        clamp_min,
        clamp_max
    );
    
    // Normalize by number of elements for global average pooling
    float norm_factor = static_cast<float>(pool_kernel_d * pool_kernel_h * pool_kernel_w);
    output /= norm_factor;
    
    return output;
}
"""

fused_operations_cpp_source = """
torch::Tensor fused_operations_cuda(
    torch::Tensor input,
    int pool_kernel_d,
    int pool_kernel_h,
    int pool_kernel_w,
    float scale,
    float clamp_min,
    float clamp_max
);
"""

# Compile the inline CUDA code for fused operations
fused_operations = load_inline(
    name="fused_operations",
    cpp_sources=fused_operations_cpp_source,
    cuda_sources=fused_operations_source,
    functions=["fused_operations_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernels for fused operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = 0
        self.clamp_max = 1
        self.fused_ops = fused_operations

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fuse max pooling, scaling, global average pooling, and clamping
        x = self.fused_ops.fused_operations_cuda(
            x,
            self.maxpool_kernel_size,
            self.maxpool_kernel_size,
            self.maxpool_kernel_size,
            self.scale,
            self.clamp_min,
            self.clamp_max
        )
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale = 0.5
maxpool_kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size]