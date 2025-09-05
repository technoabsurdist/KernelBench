import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused kernel for clamp + spatial softmax + scale
fused_clamp_softmax_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void fused_clamp_softmax_scale_kernel(
    const float* input,
    const float* scale,
    float* output,
    float clamp_min,
    float clamp_max,
    int batch_size,
    int channels,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * spatial_size;
    
    if (idx >= total_elements) return;
    
    int b = idx / (channels * spatial_size);
    int c = (idx / spatial_size) % channels;
    int s = idx % spatial_size;
    
    // Compute max for numerical stability (over spatial dimension for this channel)
    float max_val = -FLT_MAX;
    int base_idx = b * channels * spatial_size + c * spatial_size;
    for (int i = 0; i < spatial_size; i++) {
        float val = input[base_idx + i];
        val = fminf(fmaxf(val, clamp_min), clamp_max);
        max_val = fmaxf(max_val, val);
    }
    
    // Compute sum of exp
    float sum_exp = 0.0f;
    for (int i = 0; i < spatial_size; i++) {
        float val = input[base_idx + i];
        val = fminf(fmaxf(val, clamp_min), clamp_max);
        sum_exp += expf(val - max_val);
    }
    
    // Apply softmax and scale
    float clamped_val = fminf(fmaxf(input[idx], clamp_min), clamp_max);
    float softmax_val = expf(clamped_val - max_val) / sum_exp;
    output[idx] = softmax_val * scale[c];
}

torch::Tensor fused_clamp_softmax_scale_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    float clamp_min,
    float clamp_max
) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto spatial_size = depth * height * width;
    
    auto output = torch::zeros_like(input);
    
    const int block_size = 256;
    const int num_elements = batch_size * channels * spatial_size;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    
    fused_clamp_softmax_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        clamp_min,
        clamp_max,
        batch_size,
        channels,
        spatial_size
    );
    
    return output;
}
"""

fused_clamp_softmax_scale_cpp_source = """
torch::Tensor fused_clamp_softmax_scale_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    float clamp_min,
    float clamp_max
);
"""

fused_clamp_softmax_scale = load_inline(
    name="fused_clamp_softmax_scale",
    cpp_sources=fused_clamp_softmax_scale_cpp_source,
    cuda_sources=fused_clamp_softmax_scale_source,
    functions=["fused_clamp_softmax_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(out_channels))
        self.fused_op = fused_clamp_softmax_scale

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        x = self.fused_op.fused_clamp_softmax_scale_cuda(
            x, 
            self.scale, 
            self.clamp_min, 
            self.clamp_max
        )
        return x

batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 32, 64, 64
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max]