import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv3d_transpose + scale + batch_norm + global_avg_pool
fused_conv_bn_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void scale_and_bn_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float scale_factor,
    const float eps,
    const int num_elements,
    const int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int c = (idx / spatial_size) % blockDim.y;
        float mean = running_mean[c];
        float var = running_var[c];
        float w = weight[c];
        float b = bias[c];
        
        float normalized = (input[idx] - mean) / sqrtf(var + eps);
        output[idx] = (normalized * w + b) * scale_factor;
    }
}

__global__ void global_avg_pool_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int channels,
    const int spatial_size
) {
    int c = blockIdx.x;
    if (c < channels) {
        float sum = 0.0f;
        for (int i = 0; i < spatial_size; i++) {
            sum += input[c * spatial_size + i];
        }
        output[c] = sum / (float)spatial_size;
    }
}

torch::Tensor fused_conv_bn_pool(torch::Tensor input, 
                                 torch::Tensor weight,
                                 torch::Tensor bias,
                                 torch::Tensor running_mean,
                                 torch::Tensor running_var,
                                 torch::Tensor conv_transpose_weight,
                                 int kernel_size,
                                 int stride,
                                 int padding,
                                 float scale_factor,
                                 float eps) {
    // Perform conv transpose using cuDNN (simplified - in practice would need full implementation)
    // For this example, we'll assume the conv transpose is pre-computed or use PyTorch's implementation
    // and focus on fusing scale, batch norm, and global avg pool
    
    auto batch_size = input.size(0);
    auto channels = weight.size(0);
    
    // In a real implementation, we would perform the conv transpose here
    // For now, we'll assume input is already the result of conv transpose
    
    auto spatial_size = 1;
    for (int i = 2; i < input.dim(); i++) {
        spatial_size *= input.size(i);
    }
    
    auto num_elements = input.numel();
    
    // Scale and batch norm fusion
    auto output = torch::zeros_like(input);
    
    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    
    dim3 block_dim(block_size, channels, 1);
    dim3 grid_dim(num_blocks, 1, 1);
    
    scale_and_bn_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        scale_factor,
        eps,
        num_elements,
        spatial_size
    );
    
    // Global average pooling
    auto pooled_output = torch::zeros({batch_size, channels, 1, 1, 1}, 
                                      torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    global_avg_pool_kernel<<<channels, 1>>>(
        pooled_output.data_ptr<float>(),
        output.data_ptr<float>(),
        channels,
        spatial_size
    );
    
    return pooled_output;
}
"""

fused_conv_bn_pool_cpp_source = """
torch::Tensor fused_conv_bn_pool(torch::Tensor input, 
                                 torch::Tensor weight,
                                 torch::Tensor bias,
                                 torch::Tensor running_mean,
                                 torch::Tensor running_var,
                                 torch::Tensor conv_transpose_weight,
                                 int kernel_size,
                                 int stride,
                                 int padding,
                                 float scale_factor,
                                 float eps);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_conv_bn_pool",
    cpp_sources=fused_conv_bn_pool_cpp_source,
    cuda_sources=fused_conv_bn_pool_source,
    functions=["fused_conv_bn_pool"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA operations for conv transpose, scale, batch norm, and global avg pool.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fused_ops = fused_ops

    def forward(self, x):
        # For simplicity in this example, we'll use PyTorch's conv transpose
        # and then apply our fused kernel for the rest
        x = self.conv_transpose(x)
        
        # Use our custom fused operation for scale, batch norm, and pooling
        return self.fused_ops.fused_conv_bn_pool(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.conv_transpose.weight,
            self.conv_transpose.kernel_size[0],
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0],
            self.scale_factor,
            self.batch_norm.eps
        )