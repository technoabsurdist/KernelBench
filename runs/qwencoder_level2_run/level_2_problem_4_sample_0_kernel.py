import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv + mish + mish
conv_mish_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__device__ float mish(float x) {
    float exp_x = expf(x);
    float exp_2x = exp_x * exp_x;
    return x * (exp_x * (exp_2x - 1.0f)) / (exp_2x + 2.0f * exp_x + 1.0f);
}

__global__ void conv_mish_mish_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int pad,
    int stride
) {
    int out_height = (height + 2 * pad - kernel_size) / stride + 1;
    int out_width = (width + 2 * pad - kernel_size) / stride + 1;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    int n = idx / (out_channels * out_height * out_width);
    int c_out = (idx / (out_height * out_width)) % out_channels;
    int h_out = (idx / out_width) % out_height;
    int w_out = idx % out_width;
    
    float sum = 0.0f;
    
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h_out * stride - pad + kh;
                int w_in = w_out * stride - pad + kw;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = ((n * in_channels + c_in) * height + h_in) * width + w_in;
                    int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    sum += bias[c_out];
    output[idx] = mish(mish(sum));
}

torch::Tensor conv_mish_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int pad,
    int stride
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    
    int out_height = (height + 2 * pad - kernel_size) / stride + 1;
    int out_width = (width + 2 * pad - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_elements = batch_size * out_channels * out_height * out_width;
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    conv_mish_mish_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        pad,
        stride
    );
    
    return output;
}
"""

conv_mish_mish_cpp_source = """
torch::Tensor conv_mish_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int pad,
    int stride
);
"""

# Compile the inline CUDA code for fused conv + mish + mish
conv_mish_mish = load_inline(
    name="conv_mish_mish",
    cpp_sources=conv_mish_mish_cpp_source,
    cuda_sources=conv_mish_mish_source,
    functions=["conv_mish_mish_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused conv + mish + mish operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_mish_mish = conv_mish_mish

    def forward(self, x):
        return self.conv_mish_mish.conv_mish_mish_cuda(
            x,
            self.conv.weight,
            self.conv.bias,
            self.conv.kernel_size[0],
            self.conv.padding[0],
            self.conv.stride[0]
        )

batch_size   = 64  
in_channels  = 64  
out_channels = 128  
height = width = 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]