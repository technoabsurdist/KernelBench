import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused min + sum + gelu + bias
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ inline float gelu_activation(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float cdf = 0.5f * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
    return x * cdf;
}

__global__ void fused_min_sum_gelu_bias_kernel(
    const float* input, 
    const float* bias,
    float* output, 
    int batch_size, 
    int channels, 
    int height, 
    int width) {
    
    int batch_idx = blockIdx.x;
    int width_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (batch_idx >= batch_size || width_idx >= width) return;
    
    // First find min across channels for all height positions
    extern __shared__ float shared_mem[];
    float* min_vals = shared_mem; // size: height
    
    for (int h = threadIdx.y; h < height; h += blockDim.y) {
        float min_val = input[batch_idx * channels * height * width + 
                              0 * height * width + h * width + width_idx];
        for (int c = 1; c < channels; c++) {
            float val = input[batch_idx * channels * height * width + 
                            c * height * width + h * width + width_idx];
            min_val = fminf(min_val, val);
        }
        min_vals[h] = min_val;
    }
    __syncthreads();
    
    // Sum across height dimension
    if (threadIdx.y == 0) {
        float sum = 0.0f;
        for (int h = 0; h < height; h++) {
            sum += min_vals[h];
        }
        
        // Apply GELU activation
        sum = gelu_activation(sum);
        
        // Add bias
        sum += bias[0];
        
        // Write output
        output[batch_idx * width + width_idx] = sum;
    }
}

torch::Tensor fused_min_sum_gelu_bias_cuda(torch::Tensor input, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto output = torch::zeros({batch_size, 1, 1, width}, input.options());
    
    dim3 block_dim(32, 8);
    dim3 grid_dim(batch_size, (width + block_dim.x - 1) / block_dim.x);
    
    int shared_mem_size = height * sizeof(float);
    
    fused_min_sum_gelu_bias_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width
    );
    
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_min_sum_gelu_bias_cuda(torch::Tensor input, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_min_sum_gelu_bias_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for fused operations
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_min_sum_gelu_bias_cuda(x.contiguous(), self.bias.view(-1))
        return x

batch_size = 16
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]