import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv transpose + max pool + max pool + sum
fused_convtranspose3d_maxpool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int output_depth,
    int output_height,
    int output_width
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (out_idx >= total_elements) return;
    
    int w = out_idx % output_width;
    out_idx /= output_width;
    int h = out_idx % output_height;
    out_idx /= output_height;
    int d = out_idx % output_depth;
    out_idx /= output_depth;
    int c = out_idx % out_channels;
    int b = out_idx / out_channels;
    
    float sum = 0.0f;
    
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int in_d = d - kd * stride + padding;
                    int in_h = h - kh * stride + padding;
                    int in_w = w - kw * stride + padding;
                    
                    if (in_d >= 0 && in_d < input_depth &&
                        in_h >= 0 && in_h < input_height &&
                        in_w >= 0 && in_w < input_width) {
                        int input_idx = b * in_channels * input_depth * input_height * input_width +
                                        ic * input_depth * input_height * input_width +
                                        in_d * input_height * input_width +
                                        in_h * input_width +
                                        in_w;
                        int weight_idx = c * in_channels * kernel_size * kernel_size * kernel_size +
                                         ic * kernel_size * kernel_size * kernel_size +
                                         kd * kernel_size * kernel_size +
                                         kh * kernel_size +
                                         kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    output[out_idx * out_channels * output_depth * output_height * output_width +
           c * output_depth * output_height * output_width +
           d * output_height * output_width +
           h * output_width +
           w] = sum + bias[c];
}

__global__ void maxpool3d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_size,
    int output_depth,
    int output_height,
    int output_width
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * output_depth * output_height * output_width;
    
    if (out_idx >= total_elements) return;
    
    int w = out_idx % output_width;
    out_idx /= output_width;
    int h = out_idx % output_height;
    out_idx /= output_height;
    int d = out_idx % output_depth;
    out_idx /= output_depth;
    int c = out_idx % channels;
    int b = out_idx / channels;
    
    float max_val = -1e38f;
    
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int in_d = d * kernel_size + kd;
                int in_h = h * kernel_size + kh;
                int in_w = w * kernel_size + kw;
                
                if (in_d < input_depth && in_h < input_height && in_w < input_width) {
                    int input_idx = b * channels * input_depth * input_height * input_width +
                                    c * input_depth * input_height * input_width +
                                    in_d * input_height * input_width +
                                    in_h * input_width +
                                    in_w;
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }
    }
    
    output[out_idx * channels * output_depth * output_height * output_width +
           c * output_depth * output_height * output_width +
           d * output_height * output_width +
           h * output_width +
           w] = max_val;
}

__global__ void sum_channels_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * depth * height * width;
    
    if (out_idx >= total_elements) return;
    
    int w = out_idx % width;
    out_idx /= width;
    int h = out_idx % height;
    out_idx /= height;
    int d = out_idx % depth;
    int b = out_idx / depth;
    
    float sum = 0.0f;
    
    for (int c = 0; c < channels; c++) {
        int input_idx = b * channels * depth * height * width +
                        c * depth * height * width +
                        d * height * width +
                        h * width +
                        w;
        sum += input[input_idx];
    }
    
    int output_idx = b * depth * height * width +
                     d * height * width +
                     h * width +
                     w;
    output[output_idx] = sum;
}

torch::Tensor fused_convtranspose3d_maxpool_maxpool_sum_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int out_channels = weight.size(0);
    
    // Calculate conv transpose output dimensions
    int conv_depth = (input_depth - 1) * stride - 2 * padding + kernel_size;
    int conv_height = (input_height - 1) * stride - 2 * padding + kernel_size;
    int conv_width = (input_width - 1) * stride - 2 * padding + kernel_size;
    
    // First max pool (kernel=2)
    int pool1_depth = conv_depth / 2;
    int pool1_height = conv_height / 2;
    int pool1_width = conv_width / 2;
    
    // Second max pool (kernel=3)
    int pool2_depth = pool1_depth / 3;
    int pool2_height = pool1_height / 3;
    int pool2_width = pool1_width / 3;
    
    // Conv transpose
    auto conv_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto conv_output = torch::zeros({batch_size, out_channels, conv_depth, conv_height, conv_width}, conv_options);
    
    const int conv_block_size = 256;
    const int conv_num_blocks = (batch_size * out_channels * conv_depth * conv_height * conv_width + conv_block_size - 1) / conv_block_size;
    
    conv_transpose3d_kernel<<<conv_num_blocks, conv_block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        conv_output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        conv_depth,
        conv_height,
        conv_width
    );
    
    // First max pool
    auto pool1_output = torch::zeros({batch_size, out_channels, pool1_depth, pool1_height, pool1_width}, conv_options);
    
    const int pool1_block_size = 256;
    const int pool1_num_blocks = (batch_size * out_channels * pool1_depth * pool1_height * pool1_width + pool1_block_size - 1) / pool1_block_size;
    
    maxpool3d_kernel<<<pool1_num_blocks, pool1_block_size>>>(
        conv_output.data_ptr<float>(),
        pool1_output.data_ptr<float>(),
        batch_size,
        out_channels,
        conv_depth,
        conv_height,
        conv_width,
        2,
        pool1_depth,
        pool1_height,
        pool1_width
    );
    
    // Second max pool
    auto pool2_output = torch::zeros({batch_size, out_channels, pool2_depth, pool2_height, pool2_width}, conv_options);
    
    const int pool2_block_size = 256;
    const int pool2_num_blocks = (batch_size * out_channels * pool2_depth * pool2_height * pool2_width + pool2_block_size - 1) / pool2_block_size;
    
    maxpool3d_kernel<<<pool2_num_blocks, pool2_block_size>>>(
        pool1_output.data_ptr<float>(),
        pool2_output.data_ptr<float>(),
        batch_size,
        out_channels,
        pool1_depth,
        pool1_height,
        pool1_width,
        3,
        pool2_depth,
        pool2_height,
        pool2_width
    );
    
    // Sum across channels
    auto final_output = torch::zeros({batch_size, 1, pool2_depth, pool2_height, pool2_width}, conv_options);
    
    const int sum_block_size = 256;
    const int sum_num_blocks = (batch_size * pool2_depth * pool2_height * pool2_width + sum_block_size - 1) / sum_block_size;
    
    sum_channels_kernel<<<sum_num_blocks, sum_block_size>>>(
        pool2_output.data_ptr<float>(),
        final_output.data_ptr<float>(),
        batch_size,
        out_channels,
        pool2_depth,
        pool2_height,
        pool2_width
    );
    
    return final_output;
}
"""

fused_convtranspose3d_maxpool_cpp_source = """
torch::Tensor fused_convtranspose3d_maxpool_maxpool_sum_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding
);
"""

# Compile the inline CUDA code
fused_convtranspose3d_maxpool = load_inline(
    name="fused_convtranspose3d_maxpool",
    cpp_sources=fused_convtranspose3d_maxpool_cpp_source,
    cuda_sources=fused_convtranspose3d_maxpool_source,
    functions=["fused_convtranspose3d_maxpool_maxpool_sum_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernels for conv transpose + max pool + max pool + sum
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Create parameters for the conv transpose layer
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Initialize weights similar to PyTorch's default
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / fan_in**0.5
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_op = fused_convtranspose3d_maxpool

    def forward(self, x):
        return self.fused_op.fused_convtranspose3d_maxpool_maxpool_sum_cuda(
            x, self.weight, self.bias, self.kernel_size, self.stride, self.padding
        )

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size = 5
stride = 2
padding = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]