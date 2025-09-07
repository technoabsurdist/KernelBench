import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv3d + leaky_relu + add + clamp + gelu
fused_conv3d_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void fused_conv3d_activation_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* sum_tensor,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_size,
    int pad) {
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    if (out_idx >= total_elements) return;
    
    int w = out_idx % output_w;
    out_idx /= output_w;
    int h = out_idx % output_h;
    out_idx /= output_h;
    int d = out_idx % output_d;
    out_idx /= output_d;
    int c_out = out_idx % out_channels;
    int b = out_idx / out_channels;
    
    float sum_val = sum_tensor[c_out];
    float val = bias[c_out];
    
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int in_d = d + kd - pad;
                    int in_h = h + kh - pad;
                    int in_w = w + kw - pad;
                    
                    if (in_d >= 0 && in_d < input_d &&
                        in_h >= 0 && in_h < input_h &&
                        in_w >= 0 && in_w < input_w) {
                        
                        int input_idx = b * (in_channels * input_d * input_h * input_w) +
                                       c_in * (input_d * input_h * input_w) +
                                       in_d * (input_h * input_w) +
                                       in_h * input_w +
                                       in_w;
                                       
                        int weight_idx = c_out * (in_channels * kernel_size * kernel_size * kernel_size) +
                                        c_in * (kernel_size * kernel_size * kernel_size) +
                                        kd * (kernel_size * kernel_size) +
                                        kh * kernel_size +
                                        kw;
                                        
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Leaky ReLU with negative_slope=0.2
    val = val > 0 ? val : val * 0.2f;
    
    // Add sum_tensor
    val += sum_val;
    
    // Clamp between -1.0 and 1.0
    val = fmaxf(-1.0f, fminf(1.0f, val));
    
    // GELU activation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (val + 0.044715f * val * val * val)));
    val = val * cdf;
    
    output[out_idx * (out_channels * output_d * output_h * output_w) +
           c_out * (output_d * output_h * output_w) +
           d * (output_h * output_w) +
           h * output_w +
           w] = val;
}

torch::Tensor fused_conv3d_activation_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor sum_tensor) {
    
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_d = input_sizes[2];
    int input_h = input_sizes[3];
    int input_w = input_sizes[4];
    
    int out_channels = weight_sizes[0];
    int kernel_size = weight_sizes[2];
    int pad = kernel_size / 2;
    
    int output_d = input_d;
    int output_h = input_h;
    int output_w = input_w;
    
    auto output = torch::zeros({batch_size, out_channels, output_d, output_h, output_w}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_conv3d_activation_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        sum_tensor.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_size,
        pad);
        
    return output;
}
"""

fused_conv3d_activation_cpp_source = """
torch::Tensor fused_conv3d_activation_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor sum_tensor);
"""

# Compile the inline CUDA code for fused conv3d + activations
fused_conv3d_activation = load_inline(
    name="fused_conv3d_activation",
    cpp_sources=fused_conv3d_activation_cpp_source,
    cuda_sources=fused_conv3d_activation_source,
    functions=["fused_conv3d_activation_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model with fused CUDA kernel for conv3d + leaky_relu + add + clamp + gelu
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sum_tensor_shape = sum_tensor_shape
        
        # Initialize convolution parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        
        # Load the custom CUDA function
        self.fused_op = fused_conv3d_activation

    def forward(self, x):
        return self.fused_op.fused_conv3d_activation_cuda(x, self.weight, self.bias, self.sum_tensor)