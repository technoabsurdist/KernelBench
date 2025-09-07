import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv transpose + scale + global average pooling
fused_conv_transpose_scale_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void scale_and_pool_kernel(const float* input, float* output, int size, float multiplier, int h, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] * multiplier;
        output[idx] = val;
    }
}

__global__ void global_avg_pool_2d_kernel(const float* input, float* output, int batch, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_channels = batch * channels;
    
    if (idx < total_channels) {
        float sum = 0.0f;
        int batch_idx = idx / channels;
        int channel_idx = idx % channels;
        
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int input_idx = ((batch_idx * channels + channel_idx) * height + h) * width + w;
                sum += input[input_idx];
            }
        }
        output[idx] = sum / (height * width);
    }
}

torch::Tensor fused_conv_transpose_scale_pool_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    float multiplier) {
    
    // Perform conv transpose using PyTorch's built-in function for simplicity
    // In a full implementation, this would also be custom CUDA
    auto conv_output = torch::conv_transpose2d(input, weight, bias, 
                                              std::vector<int64_t>{stride, stride},
                                              std::vector<int64_t>{padding, padding},
                                              std::vector<int64_t>{output_padding, output_padding});
    
    auto batch = conv_output.size(0);
    auto channels = conv_output.size(1);
    auto height = conv_output.size(2);
    auto width = conv_output.size(3);
    auto size = conv_output.numel();
    
    // Scale the output
    auto scaled_output = torch::zeros_like(conv_output);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    scale_and_pool_kernel<<<num_blocks, block_size>>>(
        conv_output.data_ptr<float>(),
        scaled_output.data_ptr<float>(),
        size,
        multiplier,
        height,
        width
    );
    
    // First global average pooling
    auto pooled_output = torch::zeros({batch, channels, 1, 1}, 
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto pooled_size = batch * channels;
    const int pool_num_blocks = (pooled_size + block_size - 1) / block_size;
    
    global_avg_pool_2d_kernel<<<pool_num_blocks, block_size>>>(
        scaled_output.data_ptr<float>(),
        pooled_output.data_ptr<float>(),
        batch,
        channels,
        height,
        width
    );
    
    // Second global average pooling (identity since already pooled)
    // Just return the same tensor since it's already [B, C, 1, 1]
    
    return pooled_output;
}
"""

fused_conv_transpose_scale_pool_cpp_source = """
torch::Tensor fused_conv_transpose_scale_pool_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    float multiplier);
"""

# Compile the inline CUDA code
fused_conv_transpose_scale_pool = load_inline(
    name="fused_conv_transpose_scale_pool",
    cpp_sources=fused_conv_transpose_scale_pool_cpp_source,
    cuda_sources=fused_conv_transpose_scale_pool_source,
    functions=["fused_conv_transpose_scale_pool_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused custom CUDA kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.multiplier = multiplier
        
        # Create the conv transpose weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Load the custom CUDA extension
        self.fused_op = fused_conv_transpose_scale_pool

    def forward(self, x):
        return self.fused_op.fused_conv_transpose_scale_pool_cuda(
            x,
            self.weight,
            self.bias,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.multiplier
        )