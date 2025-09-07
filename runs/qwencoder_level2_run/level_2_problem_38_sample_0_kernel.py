import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void avg_pool3d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int in_channels,
    int in_depth,
    int in_height,
    int in_width,
    int pool_size
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_elements = batch_size * in_channels * (in_depth/pool_size) * (in_height/pool_size) * (in_width/pool_size);
    
    if (out_idx >= total_out_elements) return;
    
    int pooled_d = in_depth / pool_size;
    int pooled_h = in_height / pool_size;
    int pooled_w = in_width / pool_size;
    
    int b = out_idx / (in_channels * pooled_d * pooled_h * pooled_w);
    int c = (out_idx / (pooled_d * pooled_h * pooled_w)) % in_channels;
    int d = (out_idx / (pooled_h * pooled_w)) % pooled_d;
    int h = (out_idx / pooled_w) % pooled_h;
    int w = out_idx % pooled_w;
    
    float sum = 0.0f;
    int count = 0;
    
    for (int pd = 0; pd < pool_size; pd++) {
        for (int ph = 0; ph < pool_size; ph++) {
            for (int pw = 0; pw < pool_size; pw++) {
                int in_d = d * pool_size + pd;
                int in_h = h * pool_size + ph;
                int in_w = w * pool_size + pw;
                
                if (in_d < in_depth && in_h < in_height && in_w < in_width) {
                    int in_idx = b * (in_channels * in_depth * in_height * in_width) +
                                 c * (in_depth * in_height * in_width) +
                                 in_d * (in_height * in_width) +
                                 in_h * in_width +
                                 in_w;
                    sum += input[in_idx];
                    count++;
                }
            }
        }
    }
    
    output[out_idx] = sum / count;
}

__global__ void clamp_and_softmax_kernel(
    const float* input,
    float* output,
    float clamp_min,
    float clamp_max,
    int batch_size,
    int channels,
    int spatial_size
) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int tid = threadIdx.x;
    
    if (b >= batch_size || c >= channels) return;
    
    // Shared memory for reduction
    extern __shared__ float shared_data[];
    float* shared_vals = shared_data;
    float* shared_exp_vals = shared_data + blockDim.x;
    
    float thread_max = -INFINITY;
    
    // Clamp and find max
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = b * (channels * spatial_size) + c * spatial_size + i;
        float val = fmaxf(clamp_min, fminf(clamp_max, input[idx]));
        shared_vals[i] = val;
        thread_max = fmaxf(thread_max, val);
    }
    
    // Reduction to find maximum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            shared_exp_vals[tid] = (tid < spatial_size) ? shared_vals[tid] : -INFINITY;
            shared_exp_vals[tid + stride] = (tid + stride < spatial_size) ? shared_vals[tid + stride] : -INFINITY;
            shared_exp_vals[tid] = fmaxf(shared_exp_vals[tid], shared_exp_vals[tid + stride]);
        }
    }
    
    __syncthreads();
    float max_val = shared_exp_vals[0];
    __syncthreads();
    
    // Compute exponentials and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        float exp_val = expf(shared_vals[i] - max_val);
        shared_exp_vals[i] = exp_val;
        thread_sum += exp_val;
    }
    
    // Reduction to find sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            float left = (tid < spatial_size) ? shared_exp_vals[tid] : 0.0f;
            float right = (tid + stride < spatial_size) ? shared_exp_vals[tid + stride] : 0.0f;
            shared_exp_vals[tid] = left + right;
        }
    }
    
    __syncthreads();
    float sum_exp = shared_exp_vals[0];
    __syncthreads();
    
    // Write output
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = b * (channels * spatial_size) + c * spatial_size + i;
        output[idx] = shared_exp_vals[i] / sum_exp;
    }
}

__global__ void scale_kernel(
    float* input,
    const float* scale,
    int total_elements,
    int channels,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    int c = (idx / spatial_size) % channels;
    input[idx] *= scale[c];
}

torch::Tensor fused_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    int pool_size,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    float clamp_min,
    float clamp_max
) {
    // Get dimensions
    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int in_depth = input_sizes[2];
    int in_height = input_sizes[3];
    int in_width = input_sizes[4];
    
    // Calculate pooled dimensions
    int pooled_depth = in_depth / pool_size;
    int pooled_height = in_height / pool_size;
    int pooled_width = in_width / pool_size;
    
    // Calculate output dimensions after conv transpose
    int out_depth = (pooled_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_height = (pooled_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (pooled_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_channels = weight.size(0);
    
    // Allocate output tensor
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, options);
    
    // Step 1: Average pooling
    auto pooled = torch::zeros({batch_size, in_channels, pooled_depth, pooled_height, pooled_width}, options);
    
    const int pool_block_size = 256;
    int pool_total_elements = batch_size * in_channels * pooled_depth * pooled_height * pooled_width;
    const int pool_num_blocks = (pool_total_elements + pool_block_size - 1) / pool_block_size;
    
    avg_pool3d_kernel<<<pool_num_blocks, pool_block_size>>>(
        input.data_ptr<float>(),
        pooled.data_ptr<float>(),
        batch_size,
        in_channels,
        in_depth,
        in_height,
        in_width,
        pool_size
    );
    
    // Step 2: Conv transpose (using PyTorch's implementation for simplicity)
    auto conv_output = torch::conv_transpose3d(pooled, weight, bias, {stride, stride, stride}, {padding, padding, padding}, {output_padding, output_padding, output_padding});
    
    // Step 3: Clamp and softmax
    auto conv_sizes = conv_output.sizes();
    int spatial_size = conv_sizes[2] * conv_sizes[3] * conv_sizes[4];
    int channels = conv_sizes[1];
    
    dim3 softmax_block_size(256);
    dim3 softmax_grid_size(batch_size, channels);
    size_t shared_mem_size = 2 * 256 * sizeof(float);
    
    clamp_and_softmax_kernel<<<softmax_grid_size, softmax_block_size, shared_mem_size>>>(
        conv_output.data_ptr<float>(),
        conv_output.data_ptr<float>(),
        clamp_min,
        clamp_max,
        batch_size,
        channels,
        spatial_size
    );
    
    // Step 4: Scale
    int total_elements = batch_size * channels * spatial_size;
    const int scale_block_size = 256;
    const int scale_num_blocks = (total_elements + scale_block_size - 1) / scale_block_size;
    
    scale_kernel<<<scale_num_blocks, scale_block_size>>>(
        conv_output.data_ptr<float>(),
        scale.data_ptr<float>(),
        total_elements,
        channels,
        spatial_size
    );
    
    return conv_output;
}
"""

fused_cpp_source = """
torch::Tensor fused_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    int pool_size,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    float clamp_min,
    float clamp_max
);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_forward"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
        # Initialize conv transpose weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.scale = nn.Parameter(torch.ones(out_channels))
        
        # Initialize the fused operations module
        self.fused_ops = fused_ops

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth, height, width).
        """
        return self.fused_ops.fused_forward(
            x,
            self.weight,
            self.bias,
            self.scale,
            self.pool_kernel_size,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.clamp_min,
            self.clamp_max
        )