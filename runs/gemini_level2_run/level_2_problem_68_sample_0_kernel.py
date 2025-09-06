import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused element-wise min and subtract
fused_min_sub_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

__global__ void fused_min_sub_kernel(const float* input, float* output, const float constant, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Fused operation: min(input, constant) - constant
        output[idx] = fminf(input[idx], constant) - constant;
    }
}

torch::Tensor fused_min_sub_cuda(torch::Tensor input, torch::Tensor constant_tensor) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(constant_tensor.numel() == 1, "Constant tensor must have a single element");

    auto out = torch::empty_like(input);
    auto size = input.numel();
    
    // .item<float>() moves data to CPU, which is acceptable for a single scalar
    const float constant = constant_tensor.item<float>();

    if (size == 0) {
        return out;
    }

    // Use a block size that is often good for memory-bound kernels
    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_min_sub_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        constant,
        size
    );
    
    C10_CUDA_CHECK(cudaGetLastError());

    return out;
}
"""

fused_min_sub_cpp_source = (
    "torch::Tensor fused_min_sub_cuda(torch::Tensor input, torch::Tensor constant_tensor);"
)

# Compile the inline CUDA code. This is done once when the module is imported.
fused_op = load_inline(
    name="fused_op_min_sub",
    cpp_sources=fused_min_sub_cpp_source,
    cuda_sources=fused_min_sub_source,
    functions=["fused_min_sub_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses the min and subtract operations into a single CUDA kernel.
    The nn.Linear layer is kept as is, since its underlying cuBLAS implementation is
    highly optimized and difficult to outperform. The optimization focuses on reducing
    kernel launch overhead and memory bandwidth by fusing the subsequent element-wise operations.
    """
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))
        # Assign the compiled C++/CUDA function to the model instance
        self.fused_min_sub = fused_op.fused_min_sub_cuda

    def forward(self, x):
        # Step 1: Use the highly optimized PyTorch linear layer
        x = self.linear(x)
        # Step 2: Use the custom fused kernel for the element-wise operations
        x = self.fused_min_sub(x, self.constant)
        return x

batch_size = 128
in_features = 16384
out_features = 16384
constant = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, constant]