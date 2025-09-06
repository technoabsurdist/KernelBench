import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: Sigmoid + Scaling + Residual Add
fused_sigmoid_scale_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For expf

// CUDA kernel for Fused Sigmoid + Scaling + Residual Add
__global__ void fused_sigmoid_scale_add_kernel(
    const float* input,
    float* output,
    const float scaling_factor,
    int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        // Compute: sigmoid(val) * scaling_factor + val
        // This is an element-wise operation.
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        output[idx] = sigmoid_val * scaling_factor + val;
    }
}

// C++ wrapper function to be called from Python
torch::Tensor fused_sigmoid_scale_add_cuda(
    torch::Tensor input,
    float scaling_factor) {

    // Ensure input is a contiguous CUDA tensor of type float
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input tensor must be of type float32");

    auto out = torch::empty_like(input);
    auto size = input.numel();

    if (size == 0) {
        return out;
    }

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Launch the kernel
    fused_sigmoid_scale_add_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        scaling_factor,
        size);

    // Check for any CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
"""

# Define the C++ source for the function signature
fused_sigmoid_scale_add_cpp_source = """
torch::Tensor fused_sigmoid_scale_add_cuda(torch::Tensor input, float scaling_factor);
"""

# Compile the inline CUDA code. This is done once when the module is imported.
fused_op = load_inline(
    name="fused_sigmoid_scale_add",
    cpp_sources=fused_sigmoid_scale_add_cpp_source,
    cuda_sources=fused_sigmoid_scale_add_source,
    functions=["fused_sigmoid_scale_add_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd"
    with a custom fused CUDA kernel for the activation and residual part.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        # The GEMM part is kept as is, as it's highly optimized in cuBLAS
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor
        # Store the compiled fused operator
        self.fused_op = fused_op

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        # 1. Perform the GEMM (and bias addition) using the standard PyTorch layer
        gemm_out = self.gemm(x)

        # 2. Apply the fused Sigmoid + Scale + Add operation using our custom kernel
        # The kernel takes the output of the GEMM as input and performs the
        # following operation in a single pass:
        # result = sigmoid(gemm_out) * scaling_factor + gemm_out
        output = self.fused_op.fused_sigmoid_scale_add_cuda(gemm_out, self.scaling_factor)

        return output

batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]