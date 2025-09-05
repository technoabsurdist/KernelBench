import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void fused_div_sum_scale_kernel(
    const float* input,
    float* output,
    int batch_size,
    int hidden_size,
    float scaling_factor
) {
    extern __shared__ float shared_data[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Each block handles one batch element
        float sum = 0.0f;
        
        // Sum reduction within the block
        for (int i = tid; i < hidden_size; i += blockDim.x) {
            sum += input[batch_idx * hidden_size + i] / 2.0f;
        }
        
        // Store partial sum in shared memory
        shared_data[tid] = sum;
        __syncthreads();
        
        // Parallel reduction in shared memory
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_data[tid] += shared_data[tid + stride];
            }
            __syncthreads();
        }
        
        // Thread 0 writes the final result
        if (tid == 0) {
            output[batch_idx] = shared_data[0] * scaling_factor;
        }
    }
}

torch::Tensor fused_div_sum_scale_cuda(
    torch::Tensor input,
    float scaling_factor
) {
    auto batch_size = input.size(0);
    auto hidden_size = input.size(1);
    
    auto output = torch::zeros({batch_size, 1}, input.options());
    
    const int block_size = 256;
    const int num_blocks = batch_size;
    const int shared_mem_size = block_size * sizeof(float);
    
    fused_div_sum_scale_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hidden_size,
        scaling_factor
    );
    
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_div_sum_scale_cuda(torch::Tensor input, float scaling_factor);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_div_sum_scale_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernels.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # Use PyTorch's optimized matmul
        x = torch.matmul(x, self.weight.T)
        # Fused division, sum, and scaling operations
        x = self.fused_ops.fused_div_sum_scale_cuda(x, self.scaling_factor)
        return x

batch_size   = 1024  
input_size   = 8192  
hidden_size  = 8192 
scaling_factor = 1.5

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]