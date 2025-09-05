import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused multiply and double global average pooling
fused_mul_gap_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void fused_mul_gap_kernel(
    const float* input,
    float* output,
    float multiplier,
    int batch_size,
    int channels,
    int height,
    int width
) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    
    if (b >= batch_size || c >= channels) return;
    
    // Shared memory for reduction
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int spatial_size = height * width;
    
    // Each thread accumulates multiple elements
    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = ((b * channels + c) * height * width) + i;
        sum += input[idx] * multiplier;
    }
    
    // Store in shared memory
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[b * channels + c] = sdata[0] / spatial_size;
    }
}

torch::Tensor fused_mul_gap_cuda(torch::Tensor input, float multiplier) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto output = torch::zeros({batch_size, channels, 1, 1}, input.options());
    
    dim3 blocks(batch_size, channels);
    int threads = 256;
    int shared_mem = threads * sizeof(float);
    
    fused_mul_gap_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        multiplier,
        batch_size,
        channels,
        height,
        width
    );
    
    return output;
}
"""

fused_mul_gap_cpp_source = """
torch::Tensor fused_mul_gap_cuda(torch::Tensor input, float multiplier);
"""

# Compile the inline CUDA code
fused_mul_gap = load_inline(
    name="fused_mul_gap",
    cpp_sources=fused_mul_gap_cpp_source,
    cuda_sources=fused_mul_gap_source,
    functions=["fused_mul_gap_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier
        self.fused_mul_gap = fused_mul_gap

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_mul_gap.fused_mul_gap_cuda(x, self.multiplier)
        return x

batch_size = 16
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier]