import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Mish activation and Batch Normalization (inference only)
fused_mish_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Device function for Mish activation: x * tanh(softplus(x))
// where softplus(x) = log(1 + exp(x))
__device__ __forceinline__ float mish_activation(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

__global__ void mish_bn_inference_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int total_elements,
    int C,
    int spatial_dim) {

    // Use a grid-stride loop to ensure all elements are processed
    // regardless of the number of blocks launched.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_elements;
         i += blockDim.x * gridDim.x) {

        // Calculate the channel index 'c' for the current element 'i'.
        // This works for NCHW tensor format.
        const int c = (i / spatial_dim) % C;

        // 1. Load Batch Norm parameters for the current channel
        const float mean = running_mean[c];
        const float var = running_var[c];
        const float w = weight[c];
        const float b = bias[c];

        // 2. Compute the inverse standard deviation
        const float inv_std = rsqrtf(var + eps);

        // 3. Load the input value and apply the Mish activation
        const float mish_val = mish_activation(x[i]);

        // 4. Apply the Batch Normalization transformation and store the result
        y[i] = (mish_val - mean) * inv_std * w + b;
    }
}

// CUDA forward pass function (called from C++)
torch::Tensor mish_bn_inference_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {

    // Get tensor dimensions
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);
    const long total_elements = x.numel();
    const int spatial_dim = H * W;

    // Create an output tensor of the same shape as the input
    auto y = torch::empty_like(x);

    // Configure and launch the kernel
    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    mish_bn_inference_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        total_elements,
        C,
        spatial_dim
    );

    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}
"""

fused_mish_bn_cpp_source = """
#include <torch/extension.h>

// Forward declaration of the CUDA function which will be defined in the .cu file
torch::Tensor mish_bn_inference_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps);

// C++ entry point that performs input validation before calling the CUDA function
torch::Tensor mish_bn_inference_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {

    // Basic validation for input tensors
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");

    return mish_bn_inference_forward_cuda(x, weight, bias, running_mean, running_var, eps);
}
"""

# JIT (Just-In-Time) compile the C++/CUDA code into a loadable module
fused_mish_bn_op = load_inline(
    name="fused_mish_bn_op",
    cpp_sources=fused_mish_bn_cpp_source,
    cuda_sources=fused_mish_bn_source,
    functions=["mish_bn_inference_forward"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses the Mish activation and Batch Normalization into a single CUDA kernel for inference.
    For training, it falls back to the standard PyTorch operators to ensure correctness of gradients and running statistics.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        # Store the compiled custom operator as a module attribute
        self.fused_op = fused_mish_bn_op

    def forward(self, x):
        # 1. Convolution (remains a standard PyTorch op)
        x = self.conv(x)

        # 2. Fused Mish + BatchNorm
        if self.training:
            # During training, use the standard, separate PyTorch operations.
            # This is crucial for correct gradient calculation and updates to the running mean/variance.
            # Mish activation: x * tanh(softplus(x))
            x_act = torch.multiply(torch.tanh(F.softplus(x)), x)
            # Standard Batch Normalization
            x = self.bn(x_act)
        else:
            # During inference (i.e., model.eval()), use the custom fused CUDA kernel for a speedup.
            # This kernel combines the Mish activation and the Batch Norm affine transformation.
            x = self.fused_op.mish_bn_inference_forward(
                x,
                self.bn.weight,
                self.bn.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps
            )
        return x