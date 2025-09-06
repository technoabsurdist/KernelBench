import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: scale -> tanh -> scale -> sigmoid
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For tanhf, expf

__global__ void fused_scale_tanh_scale_sigmoid_kernel(
    const float* input,
    const float* scaling_factor,
    const float* bias,
    float* output,
    long long N, long long C, long long D, long long H, long long W) {

    long long total_elements = N * C * D * H * W;
    long long plane_size = D * H * W;

    // Using a grid-stride loop for robustness
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += gridDim.x * blockDim.x) {

        // Calculate the channel index for broadcasting
        // The scaling_factor and bias are of shape (C), so we need to find which channel this element belongs to.
        long long channel_idx = (idx / plane_size) % C;

        // Get the input value
        float val = input[idx];

        // Apply the fused operations
        // 1. x = x * self.scaling_factor
        val = val * scaling_factor[channel_idx];
        // 2. x = torch.tanh(x)
        val = tanhf(val);
        // 3. x = x * self.bias
        val = val * bias[channel_idx];
        // 4. x = torch.sigmoid(x)
        val = 1.0f / (1.0f + expf(-val));

        // Store the result
        output[idx] = val;
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor x,
    torch::Tensor scaling_factor,
    torch::Tensor bias) {

    TORCH_CHECK(x.is_cuda(), "Input tensor 'x' must be a CUDA tensor");
    TORCH_CHECK(scaling_factor.is_cuda(), "Input tensor 'scaling_factor' must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Input tensor 'bias' must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 5, "Input tensor 'x' must be 5D");
    TORCH_CHECK(x.is_contiguous(), "Input tensor 'x' must be contiguous");
    TORCH_CHECK(scaling_factor.is_contiguous(), "Input tensor 'scaling_factor' must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Input tensor 'bias' must be contiguous");


    auto N = x.size(0);
    auto C = x.size(1);
    auto D = x.size(2);
    auto H = x.size(3);
    auto W = x.size(4);

    auto out = torch::empty_like(x);
    long long total_elements = x.numel();

    if (total_elements == 0) {
        return out;
    }

    const int block_size = 256;
    // A reasonable number of blocks, the grid-stride loop will handle the rest
    const int num_blocks = std::min((int)((total_elements + block_size - 1) / block_size), 4096);

    fused_scale_tanh_scale_sigmoid_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        scaling_factor.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, D, H, W
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor scaling_factor, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_elementwise_op = load_inline(
    name="fused_elementwise_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, then a custom fused operation for the subsequent element-wise layers.
    The fused operation combines: scale -> tanh -> scale -> sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        # The parameters have shape (C, 1, 1, 1) but are contiguous in memory,
        # so they can be treated as 1D arrays of size C in the CUDA kernel.
        x = fused_elementwise_op.fused_op_cuda(x, self.scaling_factor, self.bias)
        return x