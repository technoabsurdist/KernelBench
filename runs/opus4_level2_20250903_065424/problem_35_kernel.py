import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused subtract + hardswish
subtract_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void subtract_hardswish_kernel(const float* input, float* output, float subtract_value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx] - subtract_value;
        // HardSwish: x * (min(max(x+3, 0), 6) / 6)
        float x_plus_3 = x + 3.0f;
        float clamped = fminf(fmaxf(x_plus_3, 0.0f), 6.0f);
        output[idx] = x * (clamped / 6.0f);
    }
}

torch::Tensor subtract_hardswish_cuda(torch::Tensor input, float subtract_value) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    subtract_hardswish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        subtract_value,
        size
    );
    
    return output;
}
"""

subtract_hardswish_cpp_source = "torch::Tensor subtract_hardswish_cuda(torch::Tensor input, float subtract_value);"

# Custom CUDA kernel for fused maxpool2d + mish
maxpool_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void maxpool_mish_kernel(const float* input, float* output, 
                                     int batch, int channels, int height, int width,
                                     int pool_size, int out_height, int out_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = batch * channels * out_height * out_width;
    
    if (idx < total_out) {
        int w_out = idx % out_width;
        int h_out = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % channels;
        int b = idx / (channels * out_width * out_height);
        
        float max_val = -1e10f;
        
        for (int ph = 0; ph < pool_size; ph++) {
            for (int pw = 0; pw < pool_size; pw++) {
                int h_in = h_out * pool_size + ph;
                int w_in = w_out * pool_size + pw;
                
                if (h_in < height && w_in < width) {
                    int in_idx = b * channels * height * width + 
                                c * height * width + 
                                h_in * width + w_in;
                    max_val = fmaxf(max_val, input[in_idx]);
                }
            }
        }
        
        // Apply Mish: x * tanh(softplus(x))
        float softplus = logf(1.0f + expf(max_val));
        float mish_val = max_val * tanhf(softplus);
        output[idx] = mish_val;
    }
}

torch::Tensor maxpool_mish_cuda(torch::Tensor input, int pool_size) {
    auto batch = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    int out_height = height / pool_size;
    int out_width = width / pool_size;
    
    auto output = torch::empty({batch, channels, out_height, out_width}, input.options());
    
    int total_out = batch * channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_out + block_size - 1) / block_size;
    
    maxpool_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, height, width,
        pool_size, out_height, out_width
    );
    
    return output;
}
"""

maxpool_mish_cpp_source = "torch::Tensor maxpool_mish_cuda(torch::Tensor input, int pool_size);"

# Compile the inline CUDA code
subtract_hardswish = load_inline(
    name="subtract_hardswish",
    cpp_sources=subtract_hardswish_cpp_source,
    cuda_sources=subtract_hardswish_source,
    functions=["subtract_hardswish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

maxpool_mish = load_inline(
    name="maxpool_mish",
    cpp_sources=maxpool_mish_cpp_source,
    cuda_sources=maxpool_mish_source,
    functions=["maxpool_mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size
        self.subtract_hardswish = subtract_hardswish
        self.maxpool_mish = maxpool_mish

    def forward(self, x):
        x = self.conv(x)
        x = self.subtract_hardswish.subtract_hardswish_cuda(x, self.subtract_value)
        x = self.maxpool_mish.maxpool_mish_cuda(x, self.pool_kernel_size)
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]