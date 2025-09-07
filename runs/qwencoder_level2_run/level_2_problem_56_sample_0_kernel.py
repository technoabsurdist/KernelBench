import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + sigmoid + sum
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void sigmoid_sum_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = input[row * cols + i];
        sum += 1.0f / (1.0f + expf(-val));
    }
    
    // Reduce within block
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[row] = sdata[0];
    }
}

torch::Tensor fused_matmul_sigmoid_sum_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto hidden_size = weight.size(0);
    
    // Perform matrix multiplication: input @ weight.t() + bias
    auto matmul_result = torch::mm(input, weight.t());
    matmul_result = matmul_result + bias;
    
    // Allocate output tensor
    auto output = torch::zeros({batch_size, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Launch kernel for sigmoid and sum
    const int block_size = 1024;
    const int num_blocks = batch_size;
    
    sigmoid_sum_kernel<<<num_blocks, block_size>>>(
        matmul_result.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hidden_size
    );
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_matmul_sigmoid_sum_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_matmul_sigmoid_sum_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for matmul + sigmoid + sum operations.
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))
        self.fused_ops = fused_ops

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        return self.fused_ops.fused_matmul_sigmoid_sum_cuda(x, self.weight, self.bias)

batch_size = 128
input_size = 32768
hidden_size = 32768

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size]