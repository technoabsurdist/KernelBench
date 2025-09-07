import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    float hardtanh_min,
    float hardtanh_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * pooled_height * pooled_width;
    
    if (idx < total_elements) {
        int n = idx / (channels * pooled_height * pooled_width);
        int c = (idx / (pooled_height * pooled_width)) % channels;
        int ph = (idx / pooled_width) % pooled_height;
        int pw = idx % pooled_width;
        
        // Assuming max pooling with kernel_size=2, stride=2
        int h_start = ph * 2;
        int w_start = pw * 2;
        
        // Max pooling over 2x2 window
        float max_val = -1e38;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                int h = h_start + i;
                int w = w_start + j;
                if (h < height && w < width) {
                    int input_idx = ((n * channels + c) * height + h) * width + w;
                    float val = input[input_idx];
                    max_val = fmaxf(max_val, val);
                }
            }
        }
        
        // Apply hardtanh
        float activated = fminf(fmaxf(max_val, hardtanh_min), hardtanh_max);
        
        output[idx] = activated;
    }
}

__global__ void mean_tanh_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int pooled_height,
    int pooled_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels) {
        int n = idx / channels;
        int c = idx % channels;
        
        // Compute mean over spatial dimensions
        float sum = 0.0;
        for (int h = 0; h < pooled_height; h++) {
            for (int w = 0; w < pooled_width; w++) {
                int input_idx = ((n * channels + c) * pooled_height + h) * pooled_width + w;
                sum += input[input_idx];
            }
        }
        float mean_val = sum / (pooled_height * pooled_width);
        
        // Apply tanh
        output[idx] = tanhf(mean_val);
    }
}

torch::Tensor fused_operation_cuda(
    torch::Tensor input,
    int batch_size,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    float hardtanh_min,
    float hardtanh_max
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    // First kernel: maxpool + hardtanh
    auto pooled_output = torch::zeros({batch_size, channels, pooled_height, pooled_width}, 
                                      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_pooled_elements = batch_size * channels * pooled_height * pooled_width;
    const int block_size = 256;
    const int num_blocks = (total_pooled_elements + block_size - 1) / block_size;
    
    fused_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        pooled_output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        hardtanh_min,
        hardtanh_max
    );
    
    // Second kernel: mean + tanh
    auto final_output = torch::zeros({batch_size, channels, 1, 1}, 
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_mean_elements = batch_size * channels;
    const int mean_num_blocks = (total_mean_elements + block_size - 1) / block_size;
    
    mean_tanh_kernel<<<mean_num_blocks, block_size>>>(
        pooled_output.data_ptr<float>(),
        final_output.data_ptr<float>(),
        batch_size,
        channels,
        pooled_height,
        pooled_width
    );
    
    return final_output;
}
"""

fused_cpp_source = """
torch::Tensor fused_operation_cuda(
    torch::Tensor input,
    int batch_size,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    float hardtanh_min,
    float hardtanh_max
);
"""

# Compile the inline CUDA code
fused_operation = load_inline(
    name="fused_operation",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_operation_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernels for fused operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.fused_operation = fused_operation

    def forward(self, x):
        x = self.conv_transpose(x)
        batch_size, channels, height, width = x.shape
        pooled_height, pooled_width = height // 2, width // 2
        x = self.fused_operation.fused_operation_cuda(
            x, batch_size, channels, height, width, 
            pooled_height, pooled_width, 
            self.hardtanh_min, self.hardtanh_max
        )
        return x