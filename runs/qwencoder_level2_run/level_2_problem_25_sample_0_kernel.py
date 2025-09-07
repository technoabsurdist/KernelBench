import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv-min-tanh-tanh operation
fused_conv_min_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void conv_min_tanh_kernel(
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
    int total_elements = batch_size * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    int b = idx / (out_height * out_width);
    int hw = idx % (out_height * out_width);
    int h = hw / out_width;
    int w = hw % out_width;
    
    int kernel_radius = kernel_size / 2;
    
    // Compute convolution for each output channel and find minimum
    float min_val = 0.0f;
    bool first = true;
    
    for (int oc = 0; oc < out_channels; oc++) {
        float sum = bias[oc];
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int in_y = h * stride + ky - pad;
                    int in_x = w * stride + kx - pad;
                    
                    if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                        float val = input[((b * in_channels + ic) * height + in_y) * width + in_x];
                        float wgt = weight[((oc * in_channels + ic) * kernel_size + ky) * kernel_size + kx];
                        sum += val * wgt;
                    }
                }
            }
        }
        
        // Apply tanh twice
        sum = tanhf(sum);
        sum = tanhf(sum);
        
        if (first || sum < min_val) {
            min_val = sum;
            first = false;
        }
    }
    
    output[idx] = min_val;
}

torch::Tensor fused_conv_min_tanh_cuda(
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
    
    auto output = torch::zeros({batch_size, 1, out_height, out_width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    const int num_elements = batch_size * out_height * out_width;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    
    conv_min_tanh_kernel<<<num_blocks, block_size>>>(
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

fused_conv_min_tanh_cpp_source = """
torch::Tensor fused_conv_min_tanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int pad,
    int stride
);
"""

# Compile the inline CUDA code for fused operation
fused_conv_min_tanh = load_inline(
    name="fused_conv_min_tanh",
    cpp_sources=fused_conv_min_tanh_cpp_source,
    cuda_sources=fused_conv_min_tanh_source,
    functions=["fused_conv_min_tanh_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused conv-min-tanh-tanh operation using custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.stride = 1
        
        # Initialize convolution parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Store the CUDA extension
        self.fused_op = fused_conv_min_tanh

    def forward(self, x):
        return self.fused_op.fused_conv_min_tanh_cuda(
            x, self.weight, self.bias, self.kernel_size, self.pad, self.stride
        )