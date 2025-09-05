import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused BatchNorm3D + Mean Subtraction
fused_bn_mean_sub_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void fused_bn3d_mean_sub_kernel(
    const float* __restrict__ input,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * depth * height * width;
    
    if (idx >= total_elements) return;
    
    int w = idx % width;
    int h = (idx / width) % height;
    int d = (idx / (width * height)) % depth;
    int c = (idx / (width * height * depth)) % channels;
    int b = idx / (width * height * depth * channels);
    
    // Apply batch normalization
    float mean_val = running_mean[c];
    float var_val = running_var[c];
    float scale = weight[c];
    float shift = bias[c];
    
    float normalized = (input[idx] - mean_val) / sqrtf(var_val + eps);
    float bn_output = normalized * scale + shift;
    
    // Compute spatial mean for this channel and batch
    __shared__ float spatial_sum;
    if (threadIdx.x == 0) {
        spatial_sum = 0.0f;
    }
    __syncthreads();
    
    // Simple mean computation - not the most efficient but functional
    float spatial_mean = 0.0f;
    int spatial_size = depth * height * width;
    for (int i = 0; i < spatial_size; i++) {
        int curr_idx = b * channels * spatial_size + c * spatial_size + i;
        float curr_mean_val = running_mean[c];
        float curr_var_val = running_var[c];
        float curr_normalized = (input[curr_idx] - curr_mean_val) / sqrtf(curr_var_val + eps);
        spatial_mean += curr_normalized * scale + shift;
    }
    spatial_mean /= spatial_size;
    
    output[idx] = bn_output - spatial_mean;
}

torch::Tensor fused_bn3d_mean_sub_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    
    auto output = torch::zeros_like(input);
    
    int total_elements = batch_size * channels * depth * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_bn3d_mean_sub_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        depth,
        height,
        width,
        eps
    );
    
    return output;
}
"""

fused_bn_mean_sub_cpp_source = """
torch::Tensor fused_bn3d_mean_sub_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
);
"""

# Compile the inline CUDA code
fused_bn_mean_sub = load_inline(
    name="fused_bn_mean_sub",
    cpp_sources=fused_bn_mean_sub_cpp_source,
    cuda_sources=fused_bn_mean_sub_source,
    functions=["fused_bn3d_mean_sub_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized 3D convolutional transpose layer with fused BatchNorm and mean subtraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.fused_bn_mean_sub = fused_bn_mean_sub

    def forward(self, x):
        x = self.conv_transpose(x)
        
        # Use fused kernel for BatchNorm + mean subtraction
        if self.training:
            # During training, use standard PyTorch for proper gradient computation
            x = self.batch_norm(x)
            x = x - torch.mean(x, dim=(2, 3, 4), keepdim=True)
        else:
            # During inference, use fused kernel
            x = self.fused_bn_mean_sub.fused_bn3d_mean_sub_cuda(
                x,
                self.batch_norm.running_mean,
                self.batch_norm.running_var,
                self.batch_norm.weight,
                self.batch_norm.bias,
                self.batch_norm.eps
            )
        
        return x

batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]