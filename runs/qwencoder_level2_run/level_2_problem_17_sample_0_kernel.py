import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv + instance norm + div
fused_conv_inorm_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

__global__ void fused_conv_inorm_div_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int pad,
    float eps,
    float divide_by
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * height * width;
    
    if (idx >= total_elements) return;
    
    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % out_channels;
    int b = idx / (width * height * out_channels);
    
    // Convolution computation
    float conv_result = 0.0f;
    int kh = kernel_size;
    int kw = kernel_size;
    
    for (int i = 0; i < in_channels; i++) {
        for (int ky = 0; ky < kh; ky++) {
            for (int kx = 0; kx < kw; kx++) {
                int in_h = h + ky - pad;
                int in_w = w + kx - pad;
                
                if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                    int input_idx = b * (in_channels * height * width) + 
                                   i * (height * width) + 
                                   in_h * width + in_w;
                                   
                    int weight_idx = c * (in_channels * kh * kw) + 
                                    i * (kh * kw) + 
                                    ky * kw + kx;
                                    
                    conv_result += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    conv_result += bias[c];
    
    // Instance normalization
    float mean = running_mean[c];
    float var = running_var[c];
    float normalized = (conv_result - mean) / sqrtf(var + eps);
    
    // Divide by constant
    output[idx] = normalized / divide_by;
}

torch::Tensor fused_conv_inorm_div_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    int kernel_size,
    float eps,
    float divide_by
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int pad = kernel_size / 2;
    const int total_elements = batch_size * out_channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_conv_inorm_div_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        pad,
        eps,
        divide_by
    );
    
    return output;
}
"""

fused_conv_inorm_div_cpp_source = """
torch::Tensor fused_conv_inorm_div_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    int kernel_size,
    float eps,
    float divide_by
);
"""

# Compile the inline CUDA code
fused_conv_inorm_div = load_inline(
    name="fused_conv_inorm_div",
    cpp_sources=fused_conv_inorm_div_cpp_source,
    cuda_sources=fused_conv_inorm_div_source,
    functions=["fused_conv_inorm_div_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused convolution, instance normalization, and division.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.divide_by = divide_by
        
        # Create convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        # Create instance norm layer
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        
        # Store references to the fused operation
        self.fused_op = fused_conv_inorm_div
        
    def forward(self, x):
        # Use the fused CUDA kernel
        return self.fused_op.fused_conv_inorm_div_cuda(
            x,
            self.conv.weight,
            self.conv.bias,
            self.instance_norm.running_mean,
            self.instance_norm.running_var,
            self.kernel_size,
            self.instance_norm.eps,
            self.divide_by
        )

batch_size = 128
in_channels  = 64  
out_channels = 128  
height = width = 128  
kernel_size = 3
divide_by = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]