import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv + relu + hardswish
fused_conv_relu_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_conv_relu_hardswish_kernel(
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
    int out_height,
    int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % out_channels;
    int b = idx / (out_width * out_height * out_channels);
    
    float sum = 0.0f;
    
    for (int ic = 0; ic < in_channels; ic++) {
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_y = h + ky - pad;
                int in_x = w + kx - pad;
                
                if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                    int input_idx = b * (in_channels * height * width) + 
                                   ic * (height * width) + 
                                   in_y * width + in_x;
                                   
                    int weight_idx = c * (in_channels * kernel_size * kernel_size) + 
                                    ic * (kernel_size * kernel_size) + 
                                    ky * kernel_size + kx;
                                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    sum += bias[c];
    
    // ReLU activation
    sum = fmaxf(sum, 0.0f);
    
    // HardSwish activation: x * clamp((x + 3) / 6, 0, 1)
    float hardswish_factor = fminf(fmaxf((sum + 3.0f) / 6.0f, 0.0f), 1.0f);
    output[idx] = sum * hardswish_factor;
}

torch::Tensor fused_conv_relu_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int pad
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    int out_channels = weight.size(0);
    int out_height = (height + 2 * pad - kernel_size) + 1;
    int out_width = (width + 2 * pad - kernel_size) + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    int total_elements = batch_size * out_channels * out_height * out_width;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_conv_relu_hardswish_kernel<<<num_blocks, block_size>>>(
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
        out_height,
        out_width
    );
    
    return output;
}
"""

fused_conv_relu_hardswish_cpp_source = """
torch::Tensor fused_conv_relu_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int pad
);
"""

# Compile the inline CUDA code for fused conv + relu + hardswish
fused_conv_relu_hardswish = load_inline(
    name="fused_conv_relu_hardswish",
    cpp_sources=fused_conv_relu_hardswish_cpp_source,
    cuda_sources=fused_conv_relu_hardswish_source,
    functions=["fused_conv_relu_hardswish_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused conv + relu + hardswish operation in custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Bind the custom CUDA function
        self.fused_op = fused_conv_relu_hardswish

    def forward(self, x):
        return self.fused_op.fused_conv_relu_hardswish_cuda(
            x, self.weight, self.bias, self.kernel_size, self.pad
        )

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]