import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation:
# (subtract -> HardSwish -> MaxPool2d -> Mish)
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Device function for HardSwish activation: x * relu6(x + 3) / 6
__device__ __forceinline__ float hardswish_fn(float x) {
    float temp = x + 3.0f;
    temp = fminf(6.0f, fmaxf(0.0f, temp));
    return x * temp / 6.0f;
}

// Device function for Mish activation: x * tanh(softplus(x))
__device__ __forceinline__ float mish_fn(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

__global__ void fused_sub_hswish_pool_mish_kernel(
    const float* input,
    float* output,
    const float subtract_value,
    const int N,
    const int C,
    const int H_in,
    const int W_in,
    const int H_out,
    const int W_out,
    const int pool_kernel_size) {

    const int total_out_elements = N * C * H_out * W_out;
    const int stride = pool_kernel_size; // Assuming stride == kernel_size

    // Use a grid-stride loop for safe and flexible execution
    for (int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
         out_idx < total_out_elements;
         out_idx += blockDim.x * gridDim.x) {

        // De-flatten the output index to get (n, c, h_out, w_out)
        const int w_out = out_idx % W_out;
        const int h_out = (out_idx / W_out) % H_out;
        const int c = (out_idx / (W_out * H_out)) % C;
        const int n = out_idx / (W_out * H_out * C);

        // Calculate the top-left corner of the pooling window in the input tensor
        const int h_in_start = h_out * stride;
        const int w_in_start = w_out * stride;

        float max_val = -1.0e38f; // Negative infinity

        // Iterate over the pooling window
        for (int i = 0; i < pool_kernel_size; ++i) {
            for (int j = 0; j < pool_kernel_size; ++j) {
                const int h_in = h_in_start + i;
                const int w_in = w_in_start + j;

                // Bounds check (optional, assuming valid input dimensions)
                if (h_in < H_in && w_in < W_in) {
                    // Calculate the flattened input index
                    const int in_idx = n * (C * H_in * W_in) + c * (H_in * W_in) + h_in * W_in + w_in;
                    
                    // Load, subtract, and apply HardSwish
                    float val = input[in_idx];
                    val = val - subtract_value;
                    val = hardswish_fn(val);

                    // Update the maximum value
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
        }

        // Apply Mish to the result of the pooling and store it
        output[out_idx] = mish_fn(max_val);
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    const float subtract_value,
    const int pool_kernel_size) {
    
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    input = input.contiguous();

    const int N = input.size(0);
    const int C = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    // Assuming stride = kernel_size and no padding, which is the default for nn.MaxPool2d(k)
    const int stride = pool_kernel_size;
    TORCH_CHECK(H_in % stride == 0 && W_in % stride == 0, "Input dimensions must be divisible by pool_kernel_size");
    const int H_out = H_in / stride;
    const int W_out = W_in / stride;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    const int total_out_elements = N * C * H_out * W_out;
    if (total_out_elements == 0) {
        return output;
    }

    const int block_size = 256;
    const int num_blocks = (total_out_elements + block_size - 1) / block_size;

    fused_sub_hswish_pool_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        subtract_value,
        N, C, H_in, W_in, H_out, W_out,
        pool_kernel_size
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    const float subtract_value,
    const int pool_kernel_size);
"""

# JIT compile the custom CUDA kernel
fused_ops_module = load_inline(
    name="fused_ops_module",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, then a fused operation of 
    (subtract, HardSwish, MaxPool, Mish) using a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size
        
        # The custom fused operator replaces four original operations
        self.fused_ops = fused_ops_module

    def forward(self, x):
        # 1. Apply the standard PyTorch convolution
        x = self.conv(x)
        
        # 2. Apply the custom fused kernel for the remaining operations
        x = self.fused_ops.fused_ops_cuda(x, self.subtract_value, self.pool_kernel_size)
        
        return x