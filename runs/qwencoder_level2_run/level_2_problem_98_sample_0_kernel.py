import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused MatMul + AvgPool + GELU + Scale + Max
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void gelu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        data[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

__global__ void scale_and_max_kernel(const float* input, float* output, float scale_factor, int batch_size, int out_features, int pool_size) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    extern __shared__ float shared_mem[];
    float* max_vals = shared_mem;
    
    max_vals[tid] = -1e38f;
    
    for (int i = tid; i < out_features; i += blockDim.x) {
        float val = input[batch_idx * out_features + i] * scale_factor;
        max_vals[tid] = fmaxf(max_vals[tid], val);
    }
    
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            max_vals[tid] = fmaxf(max_vals[tid], max_vals[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[batch_idx] = max_vals[0];
    }
}

torch::Tensor fused_matmul_avgpool_gelu_scale_max_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int pool_kernel_size,
    float scale_factor) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    // MatMul: output = input @ weight.T + bias
    auto matmul_output = torch::matmul(input, weight.transpose(0, 1)) + bias;
    
    // AvgPool1d
    auto unsqueezed = matmul_output.unsqueeze(1);
    auto pooled = torch::avg_pool1d(unsqueezed, pool_kernel_size, pool_kernel_size);
    auto squeezed = pooled.squeeze(1);
    
    // GELU activation
    auto gelu_output = squeezed.clone();
    int size = gelu_output.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    gelu_kernel<<<num_blocks, block_size>>>(gelu_output.data_ptr<float>(), size);
    
    // Scale and Max reduction
    auto output = torch::zeros({batch_size}, gelu_output.options());
    
    const int threads_per_block = 256;
    const int shared_mem_size = threads_per_block * sizeof(float);
    
    scale_and_max_kernel<<<batch_size, threads_per_block, shared_mem_size>>>(
        gelu_output.data_ptr<float>(),
        output.data_ptr<float>(),
        scale_factor,
        batch_size,
        out_features / pool_kernel_size,
        pool_kernel_size
    );
    
    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_matmul_avgpool_gelu_scale_max_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int pool_kernel_size,
    float scale_factor);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_matmul_avgpool_gelu_scale_max_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused custom CUDA kernels.
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        
        # Linear layer parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # Reference to fused operation
        self.fused_op = fused_ops

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        return self.fused_op.fused_matmul_avgpool_gelu_scale_max_cuda(
            x, self.weight, self.bias, self.pool_kernel_size, self.scale_factor
        )