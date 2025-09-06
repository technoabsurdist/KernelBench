import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused scale and add operation
fused_scale_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel to perform: out = in * (1.0 + scaling_factor)
// This fuses the following operations from the original model:
//   original_x = x.clone()
//   x = x * scaling_factor
//   x = x + original_x
__global__ void fused_scale_add_kernel(const float* in, float* out, float scaling_factor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = in[idx] * (1.0f + scaling_factor);
    }
}

torch::Tensor fused_scale_add_cuda(torch::Tensor in, float scaling_factor) {
    TORCH_CHECK(in.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(in.is_contiguous(), "Input tensor must be contiguous");

    auto out = torch::empty_like(in);
    auto size = in.numel();

    if (size == 0) {
        return out;
    }

    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_scale_add_kernel<<<num_blocks, block_size>>>(
        in.data_ptr<float>(),
        out.data_ptr<float>(),
        scaling_factor,
        size
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# Define the C++ interface
fused_scale_add_cpp_source = """
torch::Tensor fused_scale_add_cuda(torch::Tensor in, float scaling_factor);
"""

# Compile the inline CUDA code using JIT
fused_op_lib = load_inline(
    name="fused_scale_add",
    cpp_sources=fused_scale_add_cpp_source,
    cuda_sources=fused_scale_add_source,
    functions=["fused_scale_add_cuda"],
    verbose=True,
)

# Define a custom autograd function to ensure the backward pass matches the original model
class FusedScaleAddFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor, scaling_factor):
        """
        The forward pass uses the custom CUDA kernel.
        """
        # Save scaling_factor for the backward pass
        ctx.scaling_factor = scaling_factor
        # Call the CUDA kernel
        return fused_op_lib.fused_scale_add_cuda(input_tensor, scaling_factor)

    @staticmethod
    def backward(ctx, grad_output):
        """
        The backward pass mimics the original model's gradient flow.
        Original: z = y * scaling_factor + y.detach()
        Gradient w.r.t y is: dz/dy = scaling_factor
        """
        # Retrieve the saved scaling_factor
        scaling_factor = ctx.scaling_factor
        # The gradient for the input tensor is grad_output scaled by the factor.
        grad_input = grad_output * scaling_factor
        # Gradient for scaling_factor is not needed as it's not a learnable parameter.
        return grad_input, None

class ModelNew(nn.Module):
    """
    An optimized model that fuses scaling and residual addition into a single CUDA kernel.
    The custom kernel avoids an intermediate tensor clone and fuses two element-wise
    operations, reducing memory bandwidth and kernel launch overhead.
    A custom autograd.Function is used to preserve the original model's gradient behavior.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        scaling_factor (float): Scaling factor to apply after matrix multiplication.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        # Get a reference to our custom autograd function's apply method
        self.fused_op = FusedScaleAddFunction.apply

    def forward(self, x):
        """
        Forward pass of the optimized model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # 1. Perform matrix multiplication using the highly optimized PyTorch operator
        x = self.matmul(x)
        
        # 2. Apply the fused scaling and addition using our custom autograd function
        #    which calls the custom CUDA kernel internally.
        x = self.fused_op(x, self.scaling_factor)
        
        return x