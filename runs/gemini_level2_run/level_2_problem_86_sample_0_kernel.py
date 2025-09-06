import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: Linear -> Divide -> GELU
# This kernel implements a tiled matrix multiplication for performance and fuses the subsequent element-wise operations.
fused_op_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

// The dimension of the square tile processed by each thread block.
// Should be a multiple of the warp size (32) for best performance.
#define TILE_DIM 32

// GELU activation function implementation for CUDA device.
// This is the accurate approximation used in implementations like BERT.
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu_impl(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * powf(x, 3.0f))));
}

// CUDA kernel for Fused Linear + Divide + GELU
// Computes: gelu((x @ weight.T + bias) / divisor)
__global__ void fused_linear_div_gelu_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    const float divisor,
    const int batch_size,
    const int input_size,
    const int output_size
) {
    // Shared memory for tiles of x and weight.T
    __shared__ float x_s[TILE_DIM][TILE_DIM];
    __shared__ float wT_s[TILE_DIM][TILE_DIM];

    // Identify the row and column of the output matrix this thread will compute.
    const int row = blockIdx.y * TILE_DIM + threadIdx.y;
    const int col = blockIdx.x * TILE_DIM + threadIdx.x;

    // Thread indices within the block.
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    // Accumulator for the dot product.
    float acc = 0.0f;

    // Loop over the tiles of the inner dimension (input_size).
    for (int p = 0; p < (input_size + TILE_DIM - 1) / TILE_DIM; ++p) {
        // --- Load one tile of x and one tile of weight.T into shared memory ---

        // Load a tile of x (A matrix)
        const int x_k = p * TILE_DIM + tx;
        if (row < batch_size && x_k < input_size) {
            x_s[ty][tx] = x[row * input_size + x_k];
        } else {
            x_s[ty][tx] = 0.0f;
        }

        // Load a tile of weight.T (B matrix).
        // B[k, col] is equivalent to weight[col, k].
        const int w_k = p * TILE_DIM + ty;
        if (col < output_size && w_k < input_size) {
            wT_s[ty][tx] = weight[col * input_size + w_k];
        } else {
            wT_s[ty][tx] = 0.0f;
        }

        __syncthreads();

        // --- Compute dot product for the current tiles ---
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += x_s[ty][k] * wT_s[k][tx];
        }

        __syncthreads();
    }

    // --- Post-processing (add bias, divide, GELU) and store result ---
    if (row < batch_size && col < output_size) {
        // Add bias
        acc += bias[col];
        // Divide by scalar
        acc /= divisor;
        // Apply GELU
        acc = gelu_impl(acc);
        // Write the final result to the output tensor.
        out[row * output_size + col] = acc;
    }
}

// C++ wrapper function that interfaces with PyTorch.
// This function handles tensor checks, determines kernel launch parameters, and calls the CUDA kernel.
torch::Tensor fused_linear_div_gelu_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double divisor_double
) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor x must be on a CUDA device");
    TORCH_CHECK(weight.is_cuda(), "Input tensor weight must be on a CUDA device");
    TORCH_CHECK(bias.is_cuda(), "Input tensor bias must be on a CUDA device");

    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be a float32 tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be a float32 tensor");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be a float32 tensor");

    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");

    // Ensure tensors are contiguous in memory for predictable access patterns.
    x = x.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    const int batch_size = x.size(0);
    const int input_size = x.size(1);
    const int output_size = weight.size(0);

    TORCH_CHECK(input_size == weight.size(1), "Mismatched dimensions: x.size(1) and weight.size(1)");
    TORCH_CHECK(output_size == bias.size(0), "Mismatched dimensions: weight.size(0) and bias.size(0)");

    // Create the output tensor.
    auto out = torch::empty({batch_size, output_size}, x.options());

    // Configure kernel launch parameters.
    const dim3 block_dim(TILE_DIM, TILE_DIM);
    const dim3 grid_dim(
        (output_size + TILE_DIM - 1) / TILE_DIM,
        (batch_size + TILE_DIM - 1) / TILE_DIM
    );

    const float divisor = static_cast<float>(divisor_double);

    // Launch the CUDA kernel.
    fused_linear_div_gelu_kernel<<<grid_dim, block_dim>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        divisor,
        batch_size,
        input_size,
        output_size
    );

    // Check for any errors during kernel launch.
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# C++ source for function signature, used for linking.
fused_op_cpp_source = """
torch::Tensor fused_linear_div_gelu_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double divisor_double);
"""

# Compile the inline CUDA code using torch.utils.cpp_extension.load_inline
# This creates a Python module containing the 'fused_linear_div_gelu_cuda' function.
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_cuda_source,
    functions=["fused_linear_div_gelu_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    An optimized model that uses a single custom CUDA kernel to fuse the
    matrix multiplication, scalar division, and GELU activation.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        # We still use nn.Linear to conveniently manage the weight and bias parameters.
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Ensure the input tensor is on the same device as the model parameters.
        x = x.to(self.linear.weight.device)
        
        # Call the custom fused CUDA operator.
        return fused_op.fused_linear_div_gelu_cuda(
            x, self.linear.weight, self.linear.bias, self.divisor
        )