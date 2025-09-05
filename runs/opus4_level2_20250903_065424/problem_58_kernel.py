import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused LogSumExp + HardSwish + Bias + Clamp
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_logsumexp_hardswish_bias_clamp_kernel(
    const float* input, 
    float* output, 
    const float* bias,
    int batch_size, 
    int channels, 
    int spatial_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = batch_size * spatial_size;
    
    if (idx < total_spatial) {
        int b = idx / spatial_size;
        int s = idx % spatial_size;
        
        // Compute LogSumExp over channels dimension
        float max_val = -FLT_MAX;
        for (int c = 0; c < channels; c++) {
            int input_idx = b * channels * spatial_size + c * spatial_size + s;
            max_val = fmaxf(max_val, input[input_idx]);
        }
        
        float sum_exp = 0.0f;
        for (int c = 0; c < channels; c++) {
            int input_idx = b * channels * spatial_size + c * spatial_size + s;
            sum_exp += expf(input[input_idx] - max_val);
        }
        
        float logsumexp_val = max_val + logf(sum_exp);
        
        // Apply HardSwish: x * sigmoid(x + 3) / 6
        float sigmoid_val = 1.0f / (1.0f + expf(-(logsumexp_val + 3.0f)));
        float hardswish_val = logsumexp_val * sigmoid_val / 6.0f;
        
        // Subtract bias
        float biased_val = hardswish_val - bias[0];
        
        // Clamp between -1 and 1
        float clamped_val = fminf(fmaxf(biased_val, -1.0f), 1.0f);
        
        // Output has shape (batch_size, 1, D, H, W)
        output[idx] = clamped_val;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto spatial_size = depth * height * width;
    
    auto output = torch::zeros({batch_size, 1, depth, height, width}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * spatial_size + block_size - 1) / block_size;
    
    fused_logsumexp_hardswish_bias_clamp_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size,
        channels,
        spatial_size
    );
    
    return output;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for fused operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_ops_cuda(x, self.bias.view(-1))
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (1, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]