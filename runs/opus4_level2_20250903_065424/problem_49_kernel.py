import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Softmax + Sigmoid
fused_softmax_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void fused_softmax_sigmoid_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int channels,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = batch_size * spatial_size;
    
    if (idx < total_spatial) {
        int b = idx / spatial_size;
        int s = idx % spatial_size;
        
        // Find max for numerical stability
        scalar_t max_val = input[b * channels * spatial_size + 0 * spatial_size + s];
        for (int c = 1; c < channels; ++c) {
            scalar_t val = input[b * channels * spatial_size + c * spatial_size + s];
            max_val = fmaxf(max_val, val);
        }
        
        // Compute exp and sum
        scalar_t sum = 0.0;
        for (int c = 0; c < channels; ++c) {
            scalar_t val = input[b * channels * spatial_size + c * spatial_size + s];
            scalar_t exp_val = expf(val - max_val);
            output[b * channels * spatial_size + c * spatial_size + s] = exp_val;
            sum += exp_val;
        }
        
        // Normalize (softmax) and apply sigmoid
        for (int c = 0; c < channels; ++c) {
            int out_idx = b * channels * spatial_size + c * spatial_size + s;
            scalar_t softmax_val = output[out_idx] / sum;
            output[out_idx] = 1.0f / (1.0f + expf(-softmax_val));
        }
    }
}

torch::Tensor fused_softmax_sigmoid_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    auto spatial_size = D * H * W;
    
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (batch_size * spatial_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_softmax_sigmoid_cuda", ([&] {
        fused_softmax_sigmoid_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            spatial_size
        );
    }));
    
    return output;
}
"""

fused_softmax_sigmoid_cpp_source = """
torch::Tensor fused_softmax_sigmoid_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
fused_softmax_sigmoid = load_inline(
    name="fused_softmax_sigmoid",
    cpp_sources=fused_softmax_sigmoid_cpp_source,
    cuda_sources=fused_softmax_sigmoid_source,
    functions=["fused_softmax_sigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with fused Softmax-Sigmoid kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.fused_softmax_sigmoid = fused_softmax_sigmoid

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        x = self.conv_transpose(x)
        x = self.fused_softmax_sigmoid.fused_softmax_sigmoid_cuda(x)
        return x

batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]