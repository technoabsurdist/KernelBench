import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for the fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Kernel to find the minimum value along the channel dimension (dim=1)
__global__ void channel_min_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W) {

    // Grid corresponds to (N, H, W)
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;

    if (n >= N || h >= H || w >= W) {
        return;
    }

    int channel_stride = H * W;
    int batch_stride = C * channel_stride;

    // Pointer to the first element for this (n, h, w) location
    const float* p_in = input + n * batch_stride + h * W + w;
    
    // Initialize min_val with the first channel's value
    float min_val = p_in[0];

    // Iterate over the channels to find the minimum
    for (int c = 1; c < C; ++c) {
        min_val = fminf(min_val, p_in[c * channel_stride]);
    }

    // Output has shape (N, 1, H, W)
    int out_batch_stride = H * W;
    float* p_out = output + n * out_batch_stride + h * W + w;
    *p_out = min_val;
}

// GELU approximation device function
__device__ __forceinline__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * powf(x, 3.0f))));
}

// Fused kernel for sum (dim=2), GELU, and bias addition
__global__ void sum_gelu_add_bias_kernel(
    const float* __restrict__ input, // Shape (N, 1, H, W)
    const float* __restrict__ bias,  // Scalar
    float* __restrict__ output,      // Shape (N, 1, 1, W)
    int N, int H, int W) {

    // Grid corresponds to (N, W)
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (n >= N || w >= W) {
        return;
    }

    // Input has C=1, so batch stride is H * W
    int in_batch_stride = H * W;
    const float* p_in = input + n * in_batch_stride + w;

    // Sum over the H dimension
    float sum_val = 0.0f;
    for (int h = 0; h < H; ++h) {
        sum_val += p_in[h * W];
    }

    // Apply GELU
    float gelu_val = gelu_approx(sum_val);

    // Add bias
    float final_val = gelu_val + bias[0];

    // Output has C=1, H=1, so batch stride is W
    int out_batch_stride = W;
    float* p_out = output + n * out_batch_stride + w;
    *p_out = final_val;
}

// C++ wrapper for the channel_min kernel
torch::Tensor channel_min_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(input.is_contiguous(torch::MemoryFormat::Contiguous), "Input must be contiguous");

    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);

    auto output = torch::empty({N, 1, H, W}, input.options());

    dim3 threads(16, 16); // 256 threads per block
    dim3 blocks(
        (W + threads.x - 1) / threads.x,
        (H + threads.y - 1) / threads.y,
        N
    );

    channel_min_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W
    );
    
    return output;
}

// C++ wrapper for the fused sum_gelu_add_bias kernel
torch::Tensor sum_gelu_add_bias_cuda(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(input.size(1) == 1, "Input channel dimension must be 1");
    TORCH_CHECK(bias.numel() == 1, "Bias must be a single element tensor");
    TORCH_CHECK(input.is_contiguous(torch::MemoryFormat::Contiguous), "Input must be contiguous");


    const auto N = input.size(0);
    const auto H = input.size(2);
    const auto W = input.size(3);

    auto output = torch::empty({N, 1, 1, W}, input.options());

    dim3 threads(256, 1); // 256 threads per block
    dim3 blocks(
        (W + threads.x - 1) / threads.x,
        N
    );

    sum_gelu_add_bias_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, H, W
    );

    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor channel_min_cuda(torch::Tensor input);
torch::Tensor sum_gelu_add_bias_cuda(torch::Tensor input, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["channel_min_cuda", "sum_gelu_add_bias_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    An optimized model that replaces min, sum, GELU, and addition with custom fused CUDA kernels.
    The ConvTranspose2d operation is left unchanged as it is highly optimized in cuDNN.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        # The original bias is broadcasted. A single scalar parameter is sufficient.
        self.bias = nn.Parameter(torch.randn(1))
        self.fused_ops = fused_ops

    def forward(self, x):
        # 1. Standard ConvTranspose2d
        x = self.conv_transpose(x)
        
        # 2. Custom CUDA kernel for minimum along channel dimension
        x = self.fused_ops.channel_min_cuda(x)
        
        # 3. Fused custom CUDA kernel for sum, GELU, and bias addition
        x = self.fused_ops.sum_gelu_add_bias_cuda(x, self.bias)
        
        return x