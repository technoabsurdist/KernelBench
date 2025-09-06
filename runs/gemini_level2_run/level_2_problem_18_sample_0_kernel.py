import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------------------------------------------------------------------------------
# Custom CUDA Kernel for Fused Linear + Sum with Algorithmic Optimization
# --------------------------------------------------------------------------------
# The original model computes `sum(linear(x), dim=1)`.
# This can be mathematically simplified:
# y_i = sum_j ( (sum_k x_ik * W_jk) + b_j )
#     = (sum_j sum_k x_ik * W_jk) + sum_j b_j
#     = (sum_k x_ik * (sum_j W_jk)) + sum_j b_j
# Let W_col_sum_k = sum_j W_jk and b_sum = sum_j b_j.
# Then y_i = (sum_k x_ik * W_col_sum_k) + b_sum
# This is equivalent to `x @ W_col_sum.T + b_sum`.
# This changes a (batch, in) @ (in, out) matmul to a (batch, in) @ (in, 1) matmul,
# which is a massive speedup.
#
# We implement this optimized algorithm in a single fused CUDA kernel.
# The C++ wrapper will first compute W_col_sum and b_sum using PyTorch's
# optimized reduction kernels, and then launch our custom kernel for the
# matrix-vector multiplication part.
# --------------------------------------------------------------------------------

fused_linear_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// A standard, self-contained block-wide reduction using shared memory.
// Assumes blockDim.x is a power of 2.
__inline__ __device__ void block_reduce_sum(volatile float* sdata, int tid, int block_dim) {
    __syncthreads();
    for (int s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
}

// Kernel to compute (x @ W_col_sum.T) + b_sum
// x: (batch_size, in_features)
// W_col_sum: (in_features)
// out: (batch_size) or (batch_size, 1)
__global__ void fused_linear_sum_kernel(
    const float* __restrict__ x,
    const float* __restrict__ W_col_sum,
    float* __restrict__ out,
    const float b_sum,
    const int batch_size,
    const int in_features)
{
    // Each block computes one row of the output (one element of the final vector)
    int row = blockIdx.x;
    if (row >= batch_size) {
        return;
    }

    // Shared memory for the reduction within the block
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int block_dim = blockDim.x;

    // Each thread computes a partial dot product for its assigned row
    float partial_sum = 0.0f;
    for (int k = tid; k < in_features; k += block_dim) {
        partial_sum += x[row * in_features + k] * W_col_sum[k];
    }
    sdata[tid] = partial_sum;

    // Perform reduction in shared memory
    block_reduce_sum(sdata, tid, block_dim);

    // Thread 0 writes the final result for the row to global memory
    if (tid == 0) {
        out[row] = sdata[0] + b_sum;
    }
}

// C++ wrapper function that is callable from Python
torch::Tensor fused_linear_sum_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias)
{
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Input weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Input bias must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input x must be a float32 tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "Input weight must be a float32 tensor");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "Input bias must be a float32 tensor");

    const auto batch_size = x.size(0);
    const auto in_features = x.size(1);
    const auto out_features = weight.size(0);

    TORCH_CHECK(in_features == weight.size(1), "Dimension mismatch between x and weight");
    TORCH_CHECK(out_features == bias.size(0), "Dimension mismatch between weight and bias");

    // Algorithmic optimization: pre-compute sums using efficient PyTorch ops
    auto W_col_sum = torch::sum(weight, {0});
    auto b_sum_tensor = torch::sum(bias);
    const float b_sum = b_sum_tensor.item<float>();

    // Allocate output tensor
    auto out = torch::zeros({batch_size, 1}, x.options());

    // Kernel launch configuration
    const int block_size = 256; // Must be a power of 2 for the reduction
    const dim3 grid(batch_size);
    const dim3 block(block_size);
    const size_t shared_mem_size = block_size * sizeof(float);

    // Launch kernel
    fused_linear_sum_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        W_col_sum.data_ptr<float>(),
        out.data_ptr<float>(),
        b_sum,
        batch_size,
        in_features
    );

    // Check for errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

fused_linear_sum_cpp_source = (
    "torch::Tensor fused_linear_sum_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);"
)

# JIT compile the custom CUDA kernel
fused_linear_sum = load_inline(
    name="fused_linear_sum",
    cpp_sources=fused_linear_sum_cpp_source,
    cuda_sources=fused_linear_sum_source,
    functions=["fused_linear_sum_cuda"],
    verbose=False,
)


class FusedLinearSumFunction(torch.autograd.Function):
    """
    Custom autograd function to handle the backward pass for our fused operator.
    """
    @staticmethod
    def forward(ctx, x, weight, bias):
        # Call the forward pass CUDA kernel
        output = fused_linear_sum.fused_linear_sum_cuda(x, weight, bias)
        # Save tensors needed for the backward pass
        ctx.save_for_backward(x, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output has shape (batch_size, 1)
        x, weight, bias = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None

        # Calculate gradients based on the chain rule for the simplified operation
        if ctx.needs_input_grad[0]:  # Gradient for input x
            W_col_sum = torch.sum(weight, dim=0, keepdim=True)  # Shape: (1, in_features)
            grad_x = torch.matmul(grad_output, W_col_sum)

        if ctx.needs_input_grad[1]:  # Gradient for weight
            # The gradient for each row of the weight matrix is the same
            grad_w_row = torch.matmul(grad_output.t(), x)  # Shape: (1, in_features)
            grad_weight = grad_w_row.repeat(weight.shape[0], 1)

        if ctx.needs_input_grad[2]:  # Gradient for bias
            # The gradient for each element of the bias vector is the same
            grad_b_scalar = torch.sum(grad_output)
            grad_bias = grad_b_scalar.expand_as(bias)

        return grad_x, grad_weight, grad_bias


class ModelNew(nn.Module):
    """
    Optimized model that replaces the entire sequence of operations with a single,
    algorithmically-optimized, fused CUDA kernel.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # The nn.Linear layer is kept to store and manage the model parameters (weight and bias)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # Apply our custom autograd function which wraps the CUDA kernel
        return FusedLinearSumFunction.apply(x, self.linear.weight, self.linear.bias)