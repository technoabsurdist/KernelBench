import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused MaxPool2d + Hardtanh kernel
maxpool_hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void maxpool_hardtanh_kernel(
    const float* input, 
    float* output,
    int batch_size, int channels, 
    int in_height, int in_width,
    int out_height, int out_width,
    int kernel_size, int stride,
    float min_val, float max_val) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = batch_size * channels * out_height * out_width;
    
    if (idx < total_out) {
        int w_out = idx % out_width;
        int h_out = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % channels;
        int b = idx / (out_width * out_height * channels);
        
        float max_val_pool = -FLT_MAX;
        
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = h_out * stride + kh;
                int w_in = w_out * stride + kw;
                
                if (h_in < in_height && w_in < in_width) {
                    int in_idx = b * channels * in_height * in_width +
                                c * in_height * in_width +
                                h_in * in_width + w_in;
                    max_val_pool = fmaxf(max_val_pool, input[in_idx]);
                }
            }
        }
        
        // Apply hardtanh
        output[idx] = fminf(fmaxf(max_val_pool, min_val), max_val);
    }
}

torch::Tensor maxpool_hardtanh_cuda(
    torch::Tensor input, 
    int kernel_size, int stride,
    float min_val, float max_val) {
    
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    
    auto out_height = (in_height - kernel_size) / stride + 1;
    auto out_width = (in_width - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, channels, out_height, out_width}, input.options());
    
    int total_out = batch_size * channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_out + block_size - 1) / block_size;
    
    maxpool_hardtanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_height, in_width,
        out_height, out_width,
        kernel_size, stride,
        min_val, max_val
    );
    
    return output;
}
"""

maxpool_hardtanh_cpp_source = "torch::Tensor maxpool_hardtanh_cuda(torch::Tensor input, int kernel_size, int stride, float min_val, float max_val);"

# Fused spatial mean + tanh kernel
mean_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_tanh_kernel(
    const float* input,
    float* output,
    int batch_size, int channels,
    int height, int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels;
    
    if (idx < total) {
        int c = idx % channels;
        int b = idx / channels;
        
        float sum = 0.0f;
        int spatial_size = height * width;
        
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int in_idx = b * channels * height * width + 
                            c * height * width + 
                            h * width + w;
                sum += input[in_idx];
            }
        }
        
        float mean = sum / spatial_size;
        output[idx] = tanhf(mean);
    }
}

torch::Tensor mean_tanh_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto output = torch::zeros({batch_size, channels, 1, 1}, input.options());
    
    int total = batch_size * channels;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    mean_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        height, width
    );
    
    return output;
}
"""

mean_tanh_cpp_source = "torch::Tensor mean_tanh_cuda(torch::Tensor input);"

# Compile the custom CUDA kernels
maxpool_hardtanh = load_inline(
    name="maxpool_hardtanh",
    cpp_sources=maxpool_hardtanh_cpp_source,
    cuda_sources=maxpool_hardtanh_source,
    functions=["maxpool_hardtanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

mean_tanh = load_inline(
    name="mean_tanh",
    cpp_sources=mean_tanh_cpp_source,
    cuda_sources=mean_tanh_source,
    functions=["mean_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.maxpool_hardtanh = maxpool_hardtanh
        self.mean_tanh = mean_tanh

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool_hardtanh.maxpool_hardtanh_cuda(
            x, self.maxpool_kernel_size, self.maxpool_stride, 
            self.hardtanh_min, self.hardtanh_max
        )
        x = self.mean_tanh.mean_tanh_cuda(x)
        return x

batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 256  
kernel_size  = 3
stride = 1
padding = 1
maxpool_kernel_size = 2
maxpool_stride = 2
hardtanh_min = -1
hardtanh_max = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max]