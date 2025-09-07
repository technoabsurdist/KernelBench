import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv_transpose + softmax + bias + scale + sigmoid
fused_conv_transpose_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

__global__ void fused_conv_transpose_bias_scale_sigmoid_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    float scaling_factor
) {
    // Calculate output indices
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * out_height * out_width;
    
    if (out_idx >= total_output_elements) return;
    
    // Decode output indices
    int temp = out_idx;
    int w_out = temp % out_width;
    temp /= out_width;
    int h_out = temp % out_height;
    temp /= out_height;
    int c_out = temp % out_channels;
    int n = temp / out_channels;
    
    // Calculate convolution sum
    float sum = 0.0f;
    
    // Compute input region that contributes to this output element
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            // Calculate corresponding input position
            int h_in = (h_out + padding - kh) / stride;
            int w_in = (w_out + padding - kw) / stride;
            
            // Check if division was exact (valid contribution)
            if ((h_out + padding - kh) % stride == 0 && 
                (w_out + padding - kw) % stride == 0 &&
                h_in >= 0 && h_in < in_height &&
                w_in >= 0 && w_in < in_width) {
                
                // Accumulate over input channels
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    int input_idx = ((n * in_channels + c_in) * in_height + h_in) * in_width + w_in;
                    int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Apply softmax (approximated by applying it per channel later)
    // For now, just store the conv result
    output[out_idx] = sum;
}

__global__ void softmax_channel_kernel(float* data, int batch_size, int channels, int height, int width) {
    int n = blockIdx.x;
    int h = blockIdx.y;
    int w = blockIdx.z;
    
    if (n >= batch_size || h >= height || w >= width) return;
    
    int thread_id = threadIdx.x;
    int channel_size = height * width;
    int batch_channel_size = channels * channel_size;
    
    // Shared memory for reduction
    extern __shared__ float shared_data[];
    float* max_vals = shared_data;
    float* sum_vals = &shared_data[channels];
    
    // Load data into shared memory
    for (int c = thread_id; c < channels; c += blockDim.x) {
        int idx = (n * channels + c) * channel_size + h * width + w;
        max_vals[c] = data[idx];
    }
    __syncthreads();
    
    // Find maximum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride && thread_id + stride < channels) {
            max_vals[thread_id] = fmaxf(max_vals[thread_id], max_vals[thread_id + stride]);
        }
        __syncthreads();
    }
    
    float max_val = max_vals[0];
    __syncthreads();
    
    // Compute exp and sum
    for (int c = thread_id; c < channels; c += blockDim.x) {
        int idx = (n * channels + c) * channel_size + h * width + w;
        float exp_val = expf(data[idx] - max_val);
        data[idx] = exp_val;
        sum_vals[c] = exp_val;
    }
    __syncthreads();
    
    // Compute sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride && thread_id + stride < channels) {
            sum_vals[thread_id] += sum_vals[thread_id + stride];
        }
        __syncthreads();
    }
    
    float sum_val = sum_vals[0];
    __syncthreads();
    
    // Normalize
    for (int c = thread_id; c < channels; c += blockDim.x) {
        int idx = (n * channels + c) * channel_size + h * width + w;
        data[idx] = data[idx] / sum_val;
    }
}

__global__ void bias_scale_sigmoid_kernel(float* data, const float* bias, float scaling_factor, 
                                          int total_elements, int channels, int channel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;
    
    int c = (idx / channel_size) % channels;
    float biased = data[idx] + bias[c];
    float scaled = biased * scaling_factor;
    data[idx] = 1.0f / (1.0f + expf(-scaled));
}

torch::Tensor fused_conv_transpose_softmax_bias_scale_sigmoid(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    float scaling_factor
) {
    // Get dimensions
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int in_height = input_sizes[2];
    int in_width = input_sizes[3];
    
    int out_channels = weight_sizes[0];
    int kernel_size = weight_sizes[2];
    
    // Calculate output dimensions
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Launch convolution kernel
    int total_output_elements = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_output_elements + block_size - 1) / block_size;
    
    fused_conv_transpose_bias_scale_sigmoid_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
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
        output_padding,
        scaling_factor
    );
    
    // Apply softmax per channel
    dim3 softmax_block_size(256);
    dim3 softmax_grid_size(batch_size, out_height, out_width);
    size_t shared_mem_size = 2 * out_channels * sizeof(float);
    
    softmax_channel_kernel<<<softmax_grid_size, softmax_block_size, shared_mem_size>>>(
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        out_height,
        out_width
    );
    
    // Apply bias, scaling, and sigmoid
    int channel_size = out_height * out_width;
    bias_scale_sigmoid_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        scaling_factor,
        total_output_elements,
        out_channels,
        channel_size
    );
    
    return output;
}
"""

fused_conv_transpose_softmax_cpp_source = """
torch::Tensor fused_conv_transpose_softmax_bias_scale_sigmoid(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    float scaling_factor
);
"""

# Compile the inline CUDA code
fused_conv_transpose_softmax = load_inline(
    name="fused_conv_transpose_softmax",
    cpp_sources=fused_conv_transpose_softmax_cpp_source,
    cuda_sources=fused_conv_transpose_softmax_source,
    functions=["fused_conv_transpose_softmax_bias_scale_sigmoid"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernels for conv_transpose + softmax + bias + scale + sigmoid
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
        
        # Initialize weight parameter
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Load CUDA module
        self.fused_op = fused_conv_transpose_softmax

    def forward(self, x):
        return self.fused_op.fused_conv_transpose_softmax_bias_scale_sigmoid(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.output_padding, 
            self.scaling_factor
        )