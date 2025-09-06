import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias_add and relu
fused_relu_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel to perform element-wise (x + bias) and then apply ReLU.
// It handles the broadcasting of the bias tensor.
// The input tensor x is of shape (N, C, H, W).
// The bias tensor is of shape (C, 1, 1) and is broadcasted over N, H, W.
__global__ void fused_relu_bias_kernel(
    const float* x, 
    const float* bias, 
    float* out, 
    int N, int C, int H, int W) {
    
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total_elements = (long long)N * C * H * W;

    if (idx < total_elements) {
        // Calculate the channel index 'c' from the flat index 'idx'
        // for NCHW memory layout.
        int h_w = H * W;
        int c = (idx / h_w) % C;

        // Perform the fused operation:
        // 1. Add the corresponding bias value.
        // 2. Apply the ReLU activation function (max(0, value)).
        float bias_val = bias[c];
        float x_val = x[idx];
        out[idx] = fmaxf(0.0f, x_val + bias_val);
    }
}

// C++ wrapper function that launches the CUDA kernel.
// This function is the interface between PyTorch and the custom CUDA code.
torch::Tensor fused_relu_bias_cuda(torch::Tensor x, torch::Tensor bias) {
    // Input validation checks
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Input bias must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input x must be a float32 tensor");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "Input bias must be a float32 tensor");
    TORCH_CHECK(x.dim() == 4, "Input x must be a 4D tensor (NCHW)");

    // Get tensor dimensions
    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);

    // Check bias dimensions for broadcasting. Expected shape is (C, 1, 1).
    TORCH_CHECK(bias.dim() == 3 && bias.size(0) == C && bias.size(1) == 1 && bias.size(2) == 1, 
                "Bias must have shape (C, 1, 1)");

    // Create an output tensor with the same shape and device as the input
    auto out = torch::empty_like(x);
    
    long long total_elements = x.numel();
    if (total_elements == 0) {
        return out;
    }

    // Configure kernel launch parameters
    const int block_size = 1024;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    // Launch the CUDA kernel
    fused_relu_bias_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W
    );
    
    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# C++ source for the function signature
fused_relu_bias_cpp_source = "torch::Tensor fused_relu_bias_cuda(torch::Tensor x, torch::Tensor bias);"

# Compile the inline CUDA code using torch.utils.cpp_extension.load_inline
# This creates a Python module that can be used to call the C++/CUDA function.
fused_relu_bias = load_inline(
    name="fused_relu_bias",
    cpp_sources=fused_relu_bias_cpp_source,
    cuda_sources=fused_relu_bias_source,
    functions=["fused_relu_bias_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that replaces the ReLU and bias addition with a single
    fused custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        # The convolution layer remains a standard PyTorch operator, as it's
        # highly optimized (typically using cuDNN).
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Store the compiled custom operator
        self.fused_op = fused_relu_bias

    def forward(self, x):
        # 1. Perform convolution
        x = self.conv(x)
        
        # 2. Apply the custom fused kernel for bias addition and ReLU
        # This avoids the overhead of two separate kernel launches and reduces
        # memory traffic by reading and writing to global memory only once.
        x = self.fused_op.fused_relu_bias_cuda(x, self.bias)
        
        return x