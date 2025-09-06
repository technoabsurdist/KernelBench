import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: GroupNorm -> Swish -> Multiply -> Swish
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Device function for sigmoid
__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Device function for swish (x * sigmoid(x))
__device__ __forceinline__ float swishf(float x) {
    return x * sigmoidf(x);
}

__global__ void fused_swish_mul_swish_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int batch_size,
    int features) {

    // Using a 2D grid to map to the 2D tensor
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batch_size && col < features) {
        int idx = row * features + col;

        // Load the input value (which is the output of GroupNorm)
        float val = x[idx];

        // --- Start Fused Operations ---

        // 1. First Swish
        val = swishf(val);

        // 2. Element-wise multiply with weight
        // The weight tensor is 1D, so we broadcast it across the batch dimension
        val = val * weight[col];

        // 3. Second Swish
        val = swishf(val);

        // --- End Fused Operations ---

        // Store the final result
        out[idx] = val;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor weight) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Input weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Input weight must be contiguous");
    TORCH_CHECK(x.dim() == 2, "Input x must be 2D");
    TORCH_CHECK(weight.dim() == 1, "Input weight must be 1D");
    TORCH_CHECK(x.size(1) == weight.size(0), "Dimension mismatch: x.size(1) != weight.size(0)");

    const int batch_size = x.size(0);
    const int features = x.size(1);

    auto out = torch::empty_like(x);

    // Kernel launch configuration
    // Use 2D blocks for 2D data
    dim3 block_dim(16, 16);
    dim3 grid_dim(
        (features + block_dim.x - 1) / block_dim.x,
        (batch_size + block_dim.y - 1) / block_dim.y
    );

    fused_swish_mul_swish_kernel<<<grid_dim, block_dim>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        features
    );

    // Check for errors in kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

fused_op_cpp_source = "torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor weight);"

# JIT compile the CUDA kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses the sequence of (Swish, Multiply, Swish)
    into a single custom CUDA kernel. The GEMM and GroupNorm operations
    are kept as standard PyTorch operators due to their high level of
    optimization in cuBLAS and cuDNN.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        # GEMM and GroupNorm are highly optimized; we keep them.
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        
        # The weight for the fused multiplication step
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        
        # The custom fused operator
        self.fused_op = fused_op

    def forward(self, x):
        # (batch_size, in_features) -> (batch_size, out_features)
        x = self.gemm(x)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = self.group_norm(x)
        
        # The following three operations from the original model are now fused:
        # x = x * torch.sigmoid(x)
        # x = x * self.multiply_weight
        # x = x * torch.sigmoid(x)
        
        # Call the custom CUDA kernel for the fused operations
        x = self.fused_op.fused_op_cuda(x, self.multiply_weight)
        
        return x