import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv + multiply + leaky_relu + gelu
fused_conv_act_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_conv_act_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* multiplier,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int pad,
    int stride,
    int out_height,
    int out_width
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (out_idx >= total_elements) return;
    
    int w_out = out_idx % out_width;
    int h_out = (out_idx / out_width) % out_height;
    int c_out = (out_idx / (out_width * out_height)) % out_channels;
    int n = out_idx / (out_width * out_height * out_channels);
    
    float sum = 0.0f;
    
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
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
    sum *= multiplier[c_out];
    
    // Leaky ReLU
    if (sum < 0) {
        sum *= 0.01f;
    }
    
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (sum + 0.044715f * sum * sum * sum)));
    sum = sum * cdf;
    
    output[out_idx] = sum;
}

torch::Tensor fused_conv_act_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor multiplier
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int pad = kernel_size / 2;
    int stride = 1;
    int out_height = (height + 2 * pad - kernel_size) / stride + 1;
    int out_width = (width + 2 * pad - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_elements = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_conv_act_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        pad,
        stride,
        out_height,
        out_width
    );
    
    return output;
}
"""

fused_conv_act_cpp_source = """
torch::Tensor fused_conv_act_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor multiplier
);
"""

# Compile the inline CUDA code for fused conv + activation
fused_conv_act = load_inline(
    name="fused_conv_act",
    cpp_sources=fused_conv_act_cpp_source,
    cuda_sources=fused_conv_act_source,
    functions=["fused_conv_act_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused conv + multiply + leaky_relu + gelu operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.fused_conv_act = fused_conv_act

    def forward(self, x):
        return self.fused_conv_act.fused_conv_act_cuda(x, self.weight, self.bias, self.multiplier)