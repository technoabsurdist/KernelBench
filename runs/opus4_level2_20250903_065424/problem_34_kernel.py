import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused LayerNorm + GELU + Scaling kernel
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

__device__ __forceinline__ float gelu_func(float x) {
    const float c = 0.7978845608f; // sqrt(2/pi)
    const float a = 0.044715f;
    float x3 = x * x * x;
    float inner = c * (x + a * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void fused_layernorm_gelu_scale_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float eps,
    float scale,
    int batch_size,
    int channels,
    int spatial_size
) {
    extern __shared__ float shared_data[];
    
    int tid = threadIdx.x;
    int channel_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    
    if (channel_idx >= channels || batch_idx >= batch_size) return;
    
    int base_idx = batch_idx * channels * spatial_size + channel_idx * spatial_size;
    
    // Compute mean
    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        sum += input[base_idx + i];
    }
    
    // Warp reduction for sum
    __shared__ float warp_sums[32];
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    if (tid < 32) {
        sum = (tid < (blockDim.x / WARP_SIZE)) ? warp_sums[tid] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }
    
    __syncthreads();
    float mean = __shfl_sync(0xffffffff, sum, 0) / spatial_size;
    
    // Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        float diff = input[base_idx + i] - mean;
        var_sum += diff * diff;
    }
    
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }
    
    if (lane_id == 0) {
        warp_sums[warp_id] = var_sum;
    }
    __syncthreads();
    
    if (tid < 32) {
        var_sum = (tid < (blockDim.x / WARP_SIZE)) ? warp_sums[tid] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
        }
    }
    
    __syncthreads();
    float variance = __shfl_sync(0xffffffff, var_sum, 0) / spatial_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Apply LayerNorm + GELU + Scaling
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        float val = input[base_idx + i];
        float normalized = (val - mean) * inv_std;
        float scaled = normalized * gamma[channel_idx] + beta[channel_idx];
        float activated = gelu_func(scaled);
        output[base_idx + i] = activated * scale;
    }
}

torch::Tensor fused_layernorm_gelu_scale_cuda(
    torch::Tensor input,
    torch::Tensor gamma, 
    torch::Tensor beta,
    float eps,
    float scale
) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    auto spatial_size = D * H * W;
    
    auto output = torch::zeros_like(input);
    
    dim3 blocks(channels, batch_size);
    dim3 threads(BLOCK_SIZE);
    size_t shared_mem = 32 * sizeof(float);
    
    fused_layernorm_gelu_scale_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        eps,
        scale,
        batch_size,
        channels,
        spatial_size
    );
    
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_layernorm_gelu_scale_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta, 
    float eps,
    float scale
);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_layernorm_gelu_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused LayerNorm + GELU + scaling kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.eps = eps
        self.scaling_factor = scaling_factor
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        self.fused_ops = fused_ops

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_layernorm_gelu_scale_cuda(
            x.contiguous(), 
            self.gamma,
            self.beta,
            self.eps,
            self.scaling_factor
        )
        return x

batch_size = 32
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 4
stride = 2
padding = 1
bias = True
eps = 1e-5
scaling_factor = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias, eps, scaling_factor]