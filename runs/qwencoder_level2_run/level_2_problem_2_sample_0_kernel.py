import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_conv_transpose_bias_clamp_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const float scaling_factor,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (out_idx >= total_elements) return;
    
    int tmp = out_idx;
    int w_out = tmp % out_width;
    tmp /= out_width;
    int h_out = tmp % out_height;
    tmp /= out_height;
    int c_out = tmp % out_channels;
    int n = tmp / out_channels;
    
    float sum = 0.0f;
    
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = h_out / stride - padding + kh;
                int w_in = w_out / stride - padding + kw;
                
                if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                    if ((h_out + output_padding) % stride == 0 && (w_out + output_padding) % stride == 0) {
                        int h_in_check = (h_out + output_padding) / stride - padding + kh;
                        int w_in_check = (w_out + output_padding) / stride - padding + kw;
                        if (h_in_check == h_in && w_in_check == w_in) {
                            int input_idx = ((n * in_channels + c_in) * in_height + h_in) * in_width + w_in;
                            int weight_idx = ((c_in * out_channels + c_out) * kernel_size + kh) * kernel_size + kw;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    sum += bias[c_out];
    
    // Clamp, scale, clamp, divide
    sum = fmaxf(0.0f, fminf(1.0f, sum));
    sum *= scaling_factor;
    sum = fmaxf(0.0f, fminf(1.0f, sum));
    sum /= scaling_factor;
    
    output[out_idx] = sum;
}

torch::Tensor fused_conv_transpose_bias_clamp_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor,
    int stride,
    int padding,
    int output_padding,
    int kernel_size
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    
    auto out_channels = weight.size(1);
    auto out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    auto out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    auto total_elements = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        output_padding
    );
    
    return output;
}
"""

fused_conv_transpose_bias_clamp_scale_cpp_source = """
torch::Tensor fused_conv_transpose_bias_clamp_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor,
    int stride,
    int padding,
    int output_padding,
    int kernel_size
);
"""

# Compile the inline CUDA code for fused operations
fused_conv_transpose = load_inline(
    name="fused_conv_transpose",
    cpp_sources=fused_conv_transpose_bias_clamp_scale_cpp_source,
    cuda_sources=fused_conv_transpose_bias_clamp_scale_source,
    functions=["fused_conv_transpose_bias_clamp_scale_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for conv transpose + bias + clamp + scale + clamp + divide
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scaling_factor = scaling_factor
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Initialize the CUDA module
        self.fused_conv_transpose = fused_conv_transpose

    def forward(self, x):
        return self.fused_conv_transpose.fused_conv_transpose_bias_clamp_scale_cuda(
            x, 
            self.weight, 
            self.bias, 
            self.scaling_factor,
            self.stride,
            self.padding,
            self.output_padding,
            self.kernel_size
        )

batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 128 
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]