import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused post-GEMM operations
fused_post_gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_post_gemm_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Load the value from the input tensor
        float val = input[idx];

        // 1. Swish activation: val * sigmoid(val)
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        float swish_val = val * sigmoid_val;

        // 2. Divide by 2.0
        float div_val = swish_val * 0.5f;

        // 3. Clamp between -1.0 and 1.0
        float clamp_val = fminf(fmaxf(div_val, -1.0f), 1.0f);

        // 4. Tanh activation
        float tanh_val = tanhf(clamp_val);

        // Note: The second clamp in the original model is redundant because the output
        // of tanh is already within the range [-1.0, 1.0]. We can omit it for efficiency.

        // Store the result in the output tensor
        output[idx] = tanh_val;
    }
}

torch::Tensor fused_post_gemm_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    auto size = input.numel();
    auto out = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_post_gemm_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    // Check for any CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
"""

fused_post_gemm_cpp_source = (
    "torch::Tensor fused_post_gemm_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for the fused operation
# This fuses: Swish -> Divide -> Clamp -> Tanh
fused_post_gemm = load_inline(
    name="fused_post_gemm",
    cpp_sources=fused_post_gemm_cpp_source,
    cuda_sources=fused_post_gemm_source,
    functions=["fused_post_gemm_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that keeps the PyTorch GEMM (which uses cuBLAS) and fuses
    all subsequent element-wise operations into a single custom CUDA kernel.
    The fused operations are: Swish, Divide, Clamp, and Tanh.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        # Use the highly optimized nn.Linear for the matrix multiplication
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        # Store the compiled fused operator
        self.fused_op = fused_post_gemm

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # 1. Perform GEMM using PyTorch's optimized nn.Linear
        x = self.gemm(x)
        
        # 2. Apply the single fused CUDA kernel for all subsequent operations
        x = self.fused_op.fused_post_gemm_cuda(x)
        
        return x