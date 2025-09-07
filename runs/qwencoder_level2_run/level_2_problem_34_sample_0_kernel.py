import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused layer norm + GELU + scale
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_layernorm_gelu_scale_kernel(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    const int N,
    const int C,
    const float eps,
    const float scale
) {
    // Calculate indices
    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int elem_idx = threadIdx.x;
    
    // Shared memory for reduction
    __shared__ float shared_sum[256];
    __shared__ float shared_sum_sq[256];
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    // Calculate mean
    for (int i = elem_idx; i < N; i += blockDim.x) {
        int idx = batch_idx * C * N + channel_idx * N + i;
        float val = input[idx];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    shared_sum[elem_idx] = local_sum;
    shared_sum_sq[elem_idx] = local_sum_sq;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (elem_idx < stride) {
            shared_sum[elem_idx] += shared_sum[elem_idx + stride];
            shared_sum_sq[elem_idx] += shared_sum_sq[elem_idx + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / N;
    float variance = shared_sum_sq[0] / N - mean * mean;
    float inv_std = rsqrtf(variance + eps);
    
    __syncthreads();
    
    // Apply layer norm, GELU, and scale
    float g = gamma[channel_idx];
    float b = beta[channel_idx];
    
    for (int i = elem_idx; i < N; i += blockDim.x) {
        int idx = batch_idx * C * N + channel_idx * N + i;
        float normalized = (input[idx] - mean) * inv_std;
        float ln_out = normalized * g + b;
        
        // GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float gelu_out = 0.5f * ln_out * (1.0f + tanhf(0.7978845608f * (ln_out + 0.044715f * ln_out * ln_out * ln_out)));
        
        output[idx] = gelu_out * scale;
    }
}

torch::Tensor fused_layernorm_gelu_scale_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps,
    float scale
) {
    CHECK_INPUT(input);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    
    auto shape = input.sizes();
    int batch_size = shape[0];
    int channels = shape[1];
    int elements_per_channel = 1;
    for (int i = 2; i < shape.size(); i++) {
        elements_per_channel *= shape[i];
    }
    
    auto output = torch::zeros_like(input);
    
    dim3 block(256);
    dim3 grid(batch_size, channels);
    
    fused_layernorm_gelu_scale_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        elements_per_channel,
        channels,
        eps,
        scale
    );
    
    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_layernorm_gelu_scale_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps,
    float scale
);
"""

# Compile the inline CUDA code
fused_module = load_inline(
    name="fused_layernorm_gelu_scale",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_layernorm_gelu_scale_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernels for layer norm + GELU + scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        # Layer norm parameters
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        self.eps = eps
        self.scaling_factor = scaling_factor
        self.fused_op = fused_module

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        x = self.conv_transpose(x)
        x = self.fused_op.fused_layernorm_gelu_scale_cuda(x, self.gamma, self.beta, self.eps, self.scaling_factor)
        return x