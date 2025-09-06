import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation
# The original model's forward pass simplifies to:
# 1. The output of the nn.Linear layer is fed into nn.InstanceNorm2d with a 1x1 spatial dimension.
# 2. For a 1x1 spatial input, InstanceNorm2d's output is simply its bias parameter ('beta'), broadcasted.
#    The output of the linear layer is effectively ignored.
# 3. The subsequent operations are: result = (instance_norm_bias + y) * y
# This kernel fuses these three steps (broadcast, add, multiply) into a single operation.
fused_norm_add_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_norm_add_mul_kernel(
    const float* __restrict__ y,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const int batch_size,
    const int out_features) {

    // Using a 2D grid-stride loop for robustness
    for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < batch_size; i += blockDim.y * gridDim.y) {
        for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < out_features; j += blockDim.x * gridDim.x) {
            const int idx = i * out_features + j;
            const float y_val = y[idx];
            const float bias_val = bias[j];
            out[idx] = (bias_val + y_val) * y_val;
        }
    }
}

torch::Tensor fused_norm_add_mul_cuda(torch::Tensor y, torch::Tensor bias) {
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(y.dim() == 2, "y must be a 2D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(y.size(1) == bias.size(0), "y.size(1) must equal bias.size(0)");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");

    auto out = torch::empty_like(y);
    const int batch_size = y.size(0);
    const int out_features = y.size(1);

    const dim3 block_size(32, 8);
    const dim3 num_blocks(
        (out_features + block_size.x - 1) / block_size.x,
        (batch_size + block_size.y - 1) / block_size.y
    );

    fused_norm_add_mul_kernel<<<num_blocks, block_size>>>(
        y.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        out_features
    );
    
    // Check for kernel launch errors, especially in development
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch error in fused_norm_add_mul_kernel: ", cudaGetErrorString(err));
    }

    return out;
}
"""

fused_norm_add_mul_cpp_source = (
    "torch::Tensor fused_norm_add_mul_cuda(torch::Tensor y, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_norm_add_mul_cpp_source,
    cuda_sources=fused_norm_add_mul_source,
    functions=["fused_norm_add_mul_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that replaces the sequence of operations with a single fused CUDA kernel.
    The original model's computation is equivalent to `(instance_norm.bias + y) * y`,
    making the expensive batch matrix multiplication redundant. This new model implements
    this simplified logic in an efficient, fused CUDA kernel.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # We keep the original layers to ensure that state dicts from the original
        # model can be loaded without modification. However, the bmm layer will not
        # be used in the forward pass.
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum, affine=True)
        
        # Store the compiled custom operator
        self.fused_op = fused_op

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
                              This input is ignored in the optimized version,
                              replicating the behavior of the original model where
                              the bmm output is nullified by the InstanceNorm2d.
            y (torch.Tensor): Input tensor of shape (batch_size, out_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # The entire forward pass of the original model is mathematically equivalent to
        # this single fused operation. We pass y and the bias from the instance norm
        # layer to our custom kernel.
        return self.fused_op.fused_norm_add_mul_cuda(y, self.instance_norm.bias)