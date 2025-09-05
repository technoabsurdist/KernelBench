import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused double max pooling
fused_double_maxpool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void fused_double_maxpool3d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_depth,
    int in_height,
    int in_width,
    int pool_size,
    int out_depth,
    int out_height,
    int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_depth * out_height * out_width;
    
    if (idx < total_elements) {
        int w = idx % out_width;
        int h = (idx / out_width) % out_height;
        int d = (idx / (out_width * out_height)) % out_depth;
        int c = (idx / (out_width * out_height * out_depth)) % channels;
        int b = idx / (out_width * out_height * out_depth * channels);
        
        // First pooling coordinates
        int mid_depth = in_depth / pool_size;
        int mid_height = in_height / pool_size;
        int mid_width = in_width / pool_size;
        
        float max_val = -FLT_MAX;
        
        // Iterate through the receptive field for double pooling
        for (int pd1 = 0; pd1 < pool_size; pd1++) {
            for (int ph1 = 0; ph1 < pool_size; ph1++) {
                for (int pw1 = 0; pw1 < pool_size; pw1++) {
                    // First pool coordinates
                    int d1 = d * pool_size + pd1;
                    int h1 = h * pool_size + ph1;
                    int w1 = w * pool_size + pw1;
                    
                    if (d1 < mid_depth && h1 < mid_height && w1 < mid_width) {
                        // Now do second level pooling
                        for (int pd2 = 0; pd2 < pool_size; pd2++) {
                            for (int ph2 = 0; ph2 < pool_size; ph2++) {
                                for (int pw2 = 0; pw2 < pool_size; pw2++) {
                                    int d2 = d1 * pool_size + pd2;
                                    int h2 = h1 * pool_size + ph2;
                                    int w2 = w1 * pool_size + pw2;
                                    
                                    if (d2 < in_depth && h2 < in_height && w2 < in_width) {
                                        int input_idx = ((b * channels + c) * in_depth + d2) * in_height * in_width + 
                                                       h2 * in_width + w2;
                                        max_val = fmaxf(max_val, input[input_idx]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        output[idx] = max_val;
    }
}

torch::Tensor fused_double_maxpool3d_cuda(torch::Tensor input, int pool_size) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);
    
    int out_depth = in_depth / (pool_size * pool_size);
    int out_height = in_height / (pool_size * pool_size);
    int out_width = in_width / (pool_size * pool_size);
    
    auto output = torch::zeros({batch_size, channels, out_depth, out_height, out_width}, 
                              input.options());
    
    int total_elements = batch_size * channels * out_depth * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_double_maxpool3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_depth,
        in_height,
        in_width,
        pool_size,
        out_depth,
        out_height,
        out_width
    );
    
    return output;
}
"""

fused_double_maxpool3d_cpp_source = "torch::Tensor fused_double_maxpool3d_cuda(torch::Tensor input, int pool_size);"

# Custom CUDA kernel for optimized softmax
softmax_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void softmax_3d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = batch_size * spatial_size;
    
    if (idx < total_spatial) {
        int spatial_idx = idx % spatial_size;
        int batch_idx = idx / spatial_size;
        
        // Find max for numerical stability
        float max_val = -FLT_MAX;
        for (int c = 0; c < channels; c++) {
            int offset = (batch_idx * channels + c) * spatial_size + spatial_idx;
            max_val = fmaxf(max_val, input[offset]);
        }
        
        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int c = 0; c < channels; c++) {
            int offset = (batch_idx * channels + c) * spatial_size + spatial_idx;
            float exp_val = expf(input[offset] - max_val);
            output[offset] = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize
        for (int c = 0; c < channels; c++) {
            int offset = (batch_idx * channels + c) * spatial_size + spatial_idx;
            output[offset] /= sum_exp;
        }
    }
}

torch::Tensor softmax_3d_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto spatial_size = depth * height * width;
    
    auto output = torch::zeros_like(input);
    
    int total_spatial = batch_size * spatial_size;
    const int block_size = 256;
    const int num_blocks = (total_spatial + block_size - 1) / block_size;
    
    softmax_3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        spatial_size
    );
    
    return output;
}
"""

softmax_3d_cpp_source = "torch::Tensor softmax_3d_cuda(torch::Tensor input);"

# Compile the inline CUDA code
fused_double_maxpool3d = load_inline(
    name="fused_double_maxpool3d",
    cpp_sources=fused_double_maxpool3d_cpp_source,
    cuda_sources=fused_double_maxpool3d_source,
    functions=["fused_double_maxpool3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

softmax_3d = load_inline(
    name="softmax_3d",
    cpp_sources=softmax_3d_cpp_source,
    cuda_sources=softmax_3d_source,
    functions=["softmax_3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.fused_double_maxpool3d = fused_double_maxpool3d
        self.softmax_3d = softmax_3d

    def forward(self, x):
        x = self.conv(x)
        x = self.softmax_3d.softmax_3d_cuda(x)
        x = self.fused_double_maxpool3d.fused_double_maxpool3d_cuda(x, self.pool_kernel_size)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]