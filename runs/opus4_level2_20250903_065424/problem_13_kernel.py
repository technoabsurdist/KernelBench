import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused kernel for mean_pool + bias_add + softmax + tanh + scaling
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void fused_mean_bias_softmax_tanh_scale_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float scaling_factor,
    const int batch_size,
    const int channels,
    const int depth,
    const int height,
    const int width)
{
    const int hw_size = height * width;
    const int dhw_size = depth * height * width;
    
    // Grid-stride loop for batch and spatial dimensions
    const int b = blockIdx.z;
    const int hw_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= batch_size || hw_idx >= hw_size) return;
    
    const int h = hw_idx / width;
    const int w = hw_idx % width;
    
    // Shared memory for channel values
    extern __shared__ float shared_data[];
    float* channel_vals = shared_data;
    
    // First pass: compute mean over depth for each channel and find max
    float max_val = -INFINITY;
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        int base_idx = b * channels * dhw_size + c * dhw_size + h * width + w;
        
        for (int d = 0; d < depth; d++) {
            sum += input[base_idx + d * height * width];
        }
        
        float mean = sum / depth;
        float val_with_bias = mean + bias[c];
        channel_vals[c] = val_with_bias;
        max_val = fmaxf(max_val, val_with_bias);
    }
    
    // Second pass: compute exp and sum for softmax
    float exp_sum = 0.0f;
    for (int c = 0; c < channels; c++) {
        float exp_val = expf(channel_vals[c] - max_val);
        channel_vals[c] = exp_val;
        exp_sum += exp_val;
    }
    
    // Third pass: normalize, apply tanh and scaling, write output
    for (int c = 0; c < channels; c++) {
        float softmax_val = channel_vals[c] / exp_sum;
        float tanh_val = tanhf(softmax_val);
        float scaled_val = tanh_val * scaling_factor;
        
        // Output shape: (B, C, 1, H, W)
        int out_idx = b * channels * hw_size + c * hw_size + hw_idx;
        output[out_idx] = scaled_val;
    }
}

torch::Tensor fused_mean_bias_softmax_tanh_scale_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor)
{
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int depth = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);
    
    // Output shape: (B, C, 1, H, W)
    auto output = torch::zeros({batch_size, channels, 1, height, width}, input.options());
    
    const int threads = 256;
    const int hw_size = height * width;
    const int blocks_x = (hw_size + threads - 1) / threads;
    
    dim3 blocks(blocks_x, 1, batch_size);
    dim3 threads_per_block(threads);
    
    size_t shared_mem_size = channels * sizeof(float);
    
    fused_mean_bias_softmax_tanh_scale_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        batch_size,
        channels,
        depth,
        height,
        width
    );
    
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_mean_bias_softmax_tanh_scale_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_mean_bias_softmax_tanh_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_mean_bias_softmax_tanh_scale_cuda(x, self.bias, self.scaling_factor)
        return x


def get_inputs():
    batch_size = 16
    in_channels = 16  
    depth = 32
    height = width = 128
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]


def get_init_inputs():
    in_channels = 16
    out_channels = 64
    kernel_size = 3
    stride = 1
    padding = 1
    scaling_factor = 2.0
    return [in_channels, out_channels, kernel_size, stride, padding, scaling_factor]