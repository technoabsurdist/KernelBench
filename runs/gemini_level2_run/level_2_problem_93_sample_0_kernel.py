import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// GELU implementation using the error function (erf)
// This matches torch.nn.functional.gelu
__device__ __forceinline__ float gelu_forward(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f)); // 0.7071... is 1/sqrt(2)
}

__global__ void fused_op_kernel(const float* input, float* output, int size, float add_value, float multiply_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 1. Load value
        float val = input[idx];
        
        // 2. Add a value
        val = val + add_value;
        
        // 3. Take the minimum with 0
        val = fminf(val, 0.0f);
        
        // 4. Apply GELU activation
        val = gelu_forward(val);
        
        // 5. Multiply by a value
        val = val * multiply_value;
        
        // 6. Store the result
        output[idx] = val;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor input, float add_value, float multiply_value) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");

    auto output = torch::empty_like(input);
    const int size = input.numel();

    if (size == 0) {
        return output;
    }

    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_op_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        add_value,
        multiply_value
    );
    
    // Check for errors in kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_op_cuda(torch::Tensor input, float add_value, float multiply_value);
"""

# Compile the inline CUDA code for the fused operation
fused_op_module = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    The element-wise operations (add, min, GELU, multiply) are fused into a single custom CUDA kernel for performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        # The ConvTranspose2d layer is kept as a standard PyTorch operator
        # as it is already highly optimized in cuDNN.
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        
        # Store scalar values to be passed to the CUDA kernel
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x):
        # Step 1: Perform the transposed convolution
        x = self.conv_transpose(x)
        
        # Step 2: Call the custom CUDA kernel for the fused element-wise operations.
        # We call .contiguous() to ensure the memory layout is compatible with the kernel's linear indexing.
        # This is a no-op if the tensor is already contiguous.
        return fused_op_module.fused_op_cuda(x.contiguous(), self.add_value, self.multiply_value)