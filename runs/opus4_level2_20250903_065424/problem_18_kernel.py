import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused kernel for linear layer + row-wise sum reduction
fused_linear_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void fused_linear_sum_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    float sum = 0.0f;
    
    // Each block handles one batch element
    // Threads collaborate to compute the matrix multiplication and sum
    for (int out_idx = threadIdx.x; out_idx < out_features; out_idx += blockDim.x) {
        float val = bias[out_idx];
        for (int in_idx = 0; in_idx < in_features; in_idx++) {
            val += input[batch_idx * in_features + in_idx] * weight[out_idx * in_features + in_idx];
        }
        sum += val;
    }
    
    // Reduce within block
    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();
    
    // Block-wide reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        output[batch_idx] = shared_sum[0];
    }
}

torch::Tensor fused_linear_sum_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, 1}, input.options());
    
    const int threads = 256;
    const int blocks = batch_size;
    
    fused_linear_sum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

fused_linear_sum_cpp_source = """
torch::Tensor fused_linear_sum_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
);
"""

# Compile the fused kernel
fused_linear_sum = load_inline(
    name="fused_linear_sum",
    cpp_sources=fused_linear_sum_cpp_source,
    cuda_sources=fused_linear_sum_source,
    functions=["fused_linear_sum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for linear + sum operations
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.fused_linear_sum = fused_linear_sum

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # Fused linear + sum operation
        x = self.fused_linear_sum.fused_linear_sum_cuda(
            x, 
            self.linear.weight, 
            self.linear.bias
        )
        
        # After sum, x has shape (batch_size, 1)
        # max, mean, and logsumexp on dim=1 with single element are no-ops
        x = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        
        return x

batch_size = 1024
in_features  = 8192  
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]