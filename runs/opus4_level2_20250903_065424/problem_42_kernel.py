import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused global avg pool + bias + logsumexp + sum + multiply
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void fused_pool_bias_logsumexp_sum_mul_kernel(
    const float* input, 
    const float* bias,
    float* output, 
    int batch_size,
    int channels,
    int height,
    int width,
    float multiplier) {
    
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    extern __shared__ float shared_data[];
    float* channel_avgs = shared_data;
    
    int tid = threadIdx.x;
    int num_channels = channels;
    
    // Step 1: Compute global average pooling for this batch
    for (int c = tid; c < num_channels; c += blockDim.x) {
        float sum = 0.0f;
        int spatial_size = height * width;
        for (int hw = 0; hw < spatial_size; hw++) {
            sum += input[batch_idx * num_channels * spatial_size + c * spatial_size + hw];
        }
        channel_avgs[c] = sum / spatial_size + bias[c * 1 * 1];  // Add bias here
    }
    __syncthreads();
    
    // Step 2: Compute logsumexp across channels
    if (tid == 0) {
        float max_val = -FLT_MAX;
        for (int c = 0; c < num_channels; c++) {
            max_val = fmaxf(max_val, channel_avgs[c]);
        }
        
        float sum_exp = 0.0f;
        for (int c = 0; c < num_channels; c++) {
            sum_exp += expf(channel_avgs[c] - max_val);
        }
        
        float logsumexp_val = max_val + logf(sum_exp);
        output[batch_idx] = logsumexp_val * multiplier;
    }
}

torch::Tensor fused_pool_bias_logsumexp_sum_mul_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float multiplier) {
    
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto output = torch::zeros({batch_size}, input.options());
    
    const int threads = 256;
    const int blocks = batch_size;
    size_t shared_mem_size = channels * sizeof(float);
    
    fused_pool_bias_logsumexp_sum_mul_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        multiplier
    );
    
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_pool_bias_logsumexp_sum_mul_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float multiplier);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_pool_bias_logsumexp_sum_mul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for fused operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_pool_bias_logsumexp_sum_mul_cuda(x, self.bias, 10.0)
        return x


batch_size = 16
in_channels = 64
out_channels = 128
height = width = 512
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]