import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: GEMM + Multiply + LeakyReLU
fused_gemm_mul_leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused Kernel: GEMM (Y = X @ W.T + B) -> Element-wise Multiply -> LeakyReLU
__global__ void fused_gemm_mul_leaky_relu_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    const float multiplier,
    const float negative_slope,
    const int batch_size,
    const int in_features,
    const int out_features) {

    // Each thread computes one element of the output matrix
    // Using a 2D grid of threads
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        
        // Step 1: Compute dot product for X @ W.T
        // weight is (out_features, in_features), so we access it as weight[col * in_features + k]
        for (int k = 0; k < in_features; ++k) {
            sum += x[row * in_features + k] * weight[col * in_features + k];
        }

        // Add bias
        sum += bias[col];

        // Step 2: Multiply by scalar
        sum *= multiplier;

        // Step 3: Apply LeakyReLU
        out[row * out_features + col] = (sum > 0) ? sum : sum * negative_slope;
    }
}

torch::Tensor fused_gemm_mul_leaky_relu_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    const float multiplier,
    const float negative_slope) {

    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Input weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Input bias must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input x must be 2D");
    TORCH_CHECK(weight.dim() == 2, "Input weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "Input bias must be 1D");
    TORCH_CHECK(x.size(1) == weight.size(1), "Input x and weight must have same in_features dimension");
    TORCH_CHECK(weight.size(0) == bias.size(0), "Weight and bias must have same out_features dimension");

    const int batch_size = x.size(0);
    const int in_features = x.size(1);
    const int out_features = weight.size(0);

    // Create output tensor
    auto out = torch::zeros({batch_size, out_features}, x.options());

    // Define grid and block dimensions
    const dim3 block_size(16, 16);
    const dim3 num_blocks(
        (out_features + block_size.x - 1) / block_size.x,
        (batch_size + block_size.y - 1) / block_size.y
    );

    // Launch the kernel
    fused_gemm_mul_leaky_relu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        multiplier,
        negative_slope,
        batch_size,
        in_features,
        out_features
    );
    
    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return out;
}
"""

fused_gemm_mul_leaky_relu_cpp_source = """
torch::Tensor fused_gemm_mul_leaky_relu_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    const float multiplier,
    const float negative_slope);
"""

# Compile the inline CUDA code
# This is done at the module level so it only compiles once per process
fused_op_module = load_inline(
    name="fused_gemm_mul_leaky_relu",
    cpp_sources=fused_gemm_mul_leaky_relu_cpp_source,
    cuda_sources=fused_gemm_mul_leaky_relu_source,
    functions=["fused_gemm_mul_leaky_relu_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses Gemm, multiplication, and LeakyReLU into a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.multiplier = multiplier
        self.negative_slope = negative_slope

        # Define weight and bias as learnable parameters, similar to nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize parameters using the same method as nn.Linear for fair comparison
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Call the custom fused CUDA kernel
        return fused_op_module.fused_gemm_mul_leaky_relu_cuda(
            x, self.weight, self.bias, self.multiplier, self.negative_slope
        )

batch_size = 1024
in_features  = 8192
out_features = 8192
multiplier = 2.0
negative_slope = 0.1

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]