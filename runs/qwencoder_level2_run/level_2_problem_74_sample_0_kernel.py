import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ConvTranspose3d + LeakyReLU + Scale + LeakyReLU
conv_transpose_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_conv_transpose_leaky_relu_scale_leaky_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* multiplier,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    float negative_slope
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (out_idx >= total_elements) return;
    
    int b = out_idx / (out_channels * output_depth * output_height * output_width);
    int c_out = (out_idx / (output_depth * output_height * output_width)) % out_channels;
    int d_out = (out_idx / (output_height * output_width)) % output_depth;
    int h_out = (out_idx / output_width) % output_height;
    int w_out = out_idx % output_width;
    
    float sum = (bias != nullptr) ? bias[c_out] : 0.0f;
    
    // Calculate input coordinates for transposed convolution
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int d_in = (d_out + padding - kd) / stride;
                int h_in = (h_out + padding - kh) / stride;
                int w_in = (w_out + padding - kw) / stride;
                
                if ((d_out + padding - kd) % stride == 0 &&
                    (h_out + padding - kh) % stride == 0 &&
                    (w_out + padding - kw) % stride == 0 &&
                    d_in >= 0 && d_in < input_depth &&
                    h_in >= 0 && h_in < input_height &&
                    w_in >= 0 && w_in < input_width) {
                    
                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        int weight_idx = c_out * (in_channels * kernel_size * kernel_size * kernel_size) +
                                         c_in * (kernel_size * kernel_size * kernel_size) +
                                         kd * (kernel_size * kernel_size) +
                                         kh * kernel_size + kw;
                        int input_idx = b * (in_channels * input_depth * input_height * input_width) +
                                        c_in * (input_depth * input_height * input_width) +
                                        d_in * (input_height * input_width) +
                                        h_in * input_width + w_in;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // First LeakyReLU
    float val = (sum > 0) ? sum : sum * negative_slope;
    
    // Scale by multiplier
    val *= multiplier[c_out];
    
    // Second LeakyReLU
    val = (val > 0) ? val : val * negative_slope;
    
    output[out_idx] = val;
}

torch::Tensor fused_conv_transpose_leaky_relu_scale_leaky_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor multiplier,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    float negative_slope
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, weight.size(0), output_depth, output_height, output_width}, 
                               torch::dtype(input.dtype()).device(input.device()));
    
    int total_elements = batch_size * weight.size(0) * output_depth * output_height * output_width;
    
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_conv_transpose_leaky_relu_scale_leaky_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        multiplier.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        weight.size(0),
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        negative_slope
    );
    
    return output;
}
"""

conv_transpose_fused_cpp_source = """
torch::Tensor fused_conv_transpose_leaky_relu_scale_leaky_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor multiplier,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    float negative_slope
);
"""

# Define the custom CUDA kernel for 3D max pooling
max_pool_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void max_pool_3d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_size
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * output_depth * output_height * output_width;
    
    if (out_idx >= total_elements) return;
    
    int b = out_idx / (channels * output_depth * output_height * output_width);
    int c = (out_idx / (output_depth * output_height * output_width)) % channels;
    int d_out = (out_idx / (output_height * output_width)) % output_depth;
    int h_out = (out_idx / output_width) % output_height;
    int w_out = out_idx % output_width;
    
    float max_val = -1e38f;
    
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int d_in = d_out * 2 + kd;
                int h_in = h_out * 2 + kh;
                int w_in = w_out * 2 + kw;
                
                if (d_in < input_depth && h_in < input_height && w_in < input_width) {
                    int input_idx = b * (channels * input_depth * input_height * input_width) +
                                    c * (input_depth * input_height * input_width) +
                                    d_in * (input_height * input_width) +
                                    h_in * input_width + w_in;
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }
    }
    
    output[out_idx] = max_val;
}

torch::Tensor max_pool_3d_cuda(torch::Tensor input, int kernel_size) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int output_depth = input_depth / 2;
    int output_height = input_height / 2;
    int output_width = input_width / 2;
    
    auto output = torch::zeros({batch_size, channels, output_depth, output_height, output_width}, 
                               torch::dtype(input.dtype()).device(input.device()));
    
    int total_elements = batch_size * channels * output_depth * output_height * output_width;
    
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    max_pool_3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_size
    );
    
    return output;
}
"""

max_pool_3d_cpp_source = """
torch::Tensor max_pool_3d_cuda(torch::Tensor input, int kernel_size);
"""

# Compile the inline CUDA code
fused_conv_transpose = load_inline(
    name="fused_conv_transpose",
    cpp_sources=conv_transpose_fused_cpp_source,
    cuda_sources=conv_transpose_fused_source,
    functions=["fused_conv_transpose_leaky_relu_scale_leaky_relu_cuda"],
    verbose=False,
)

max_pool_3d = load_inline(
    name="max_pool_3d",
    cpp_sources=max_pool_3d_cpp_source,
    cuda_sources=max_pool_3d_source,
    functions=["max_pool_3d_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernels for:
    1. Fused ConvTranspose3d + LeakyReLU + Scale + LeakyReLU
    2. Custom 3D max pooling
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.negative_slope = 0.2
        
        # Initialize weight and bias for the transposed convolution
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        
        # Initialize custom CUDA modules
        self.fused_conv_transpose = fused_conv_transpose
        self.max_pool_3d = max_pool_3d

    def forward(self, x):
        # Apply fused ConvTranspose3d + LeakyReLU + Scale + LeakyReLU
        x = self.fused_conv_transpose.fused_conv_transpose_leaky_relu_scale_leaky_relu_cuda(
            x, self.weight, self.bias, self.multiplier,
            self.kernel_size, self.stride, self.padding, self.output_padding, self.negative_slope
        )
        
        # Apply custom 3D max pooling
        x = self.max_pool_3d.max_pool_3d_cuda(x, 2)
        
        return x

batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape]