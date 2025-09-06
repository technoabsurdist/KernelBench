import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// GELU approximation function
__device__ __forceinline__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * powf(x, 3.0f))));
}

// Fused kernel for ReLU -> LeakyReLU -> GELU -> Sigmoid -> Bias Add
__global__ void fused_activations_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t total_elements,
    int C, int D, int H, int W,
    float leaky_relu_slope) {

    const int64_t DHW = (int64_t)D * H * W;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < total_elements; i += stride) {
        // Get the channel index for broadcasting the bias
        const int c = (i / DHW) % C;

        float val = input[i];

        // 1. ReLU
        val = fmaxf(0.0f, val);

        // 2. LeakyReLU
        val = (val > 0.0f) ? val : val * leaky_relu_slope;

        // 3. GELU
        val = gelu_approx(val);

        // 4. Sigmoid
        val = 1.0f / (1.0f + expf(-val));

        // 5. Bias Add
        val = val + bias[c];

        output[i] = val;
    }
}

// C++ wrapper function to be called from Python
torch::Tensor fused_activations_bias_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float leaky_relu_slope) {

    // Input validation checks
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    TORCH_CHECK(input.dim() == 5, "Input must be a 5D tensor");
    TORCH_CHECK(bias.dim() == 1, "Bias must be a 1D tensor (squeezed)");

    const auto sizes = input.sizes();
    const int N = sizes[0];
    const int C = sizes[1];
    const int D = sizes[2];
    const int H = sizes[3];
    const int W = sizes[4];

    TORCH_CHECK(bias.size(0) == C, "Bias size must match the channel dimension of the input");

    auto output = torch::empty_like(input);
    const int64_t total_elements = input.numel();

    if (total_elements == 0) {
        return output;
    }

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_activations_bias_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        C, D, H, W,
        leaky_relu_slope
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_activations_bias_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float leaky_relu_slope);
"""

# Compile the inline CUDA code. This will be done once when the Python module is imported.
fused_op_module = load_inline(
    name="fused_op_module",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_activations_bias_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, then a single fused kernel for
    ReLU, LeakyReLU, GELU, Sigmoid activations, and bias addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        # The Conv3d layer remains as it is, leveraging cuDNN's optimized implementation.
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # The negative slope for LeakyReLU is hardcoded as in the original model.
        self.leaky_relu_slope = 0.01

    def forward(self, x):
        # 1. Perform 3D convolution using the standard PyTorch operator.
        x = self.conv(x)

        # 2. Call the custom fused CUDA kernel for all subsequent element-wise operations.
        # The bias tensor is squeezed from (C, 1, 1, 1) to (C,) to match the kernel's expectation.
        squeezed_bias = self.bias.squeeze()
        x = fused_op_module.fused_activations_bias_cuda(x, squeezed_bias, self.leaky_relu_slope)
        
        return x