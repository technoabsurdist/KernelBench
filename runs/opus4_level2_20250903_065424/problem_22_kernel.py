import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused kernel for Linear + Scale + Double + Clamp
fused_linear_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void fused_linear_scale_clamp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int input_size,
    int hidden_size,
    float scale_factor,
    float clamp_min,
    float clamp_max) {
    
    int batch_idx = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && out_idx < hidden_size) {
        float sum = bias[out_idx];
        
        for (int i = 0; i < input_size; i++) {
            sum += input[batch_idx * input_size + i] * weight[out_idx * input_size + i];
        }
        
        // Apply scale_factor * 2 (since x + x = 2*x)
        sum = sum * scale_factor * 2.0f;
        
        // Apply clamp
        sum = fminf(fmaxf(sum, clamp_min), clamp_max);
        
        output[batch_idx * hidden_size + out_idx] = sum;
    }
}

torch::Tensor fused_linear_scale_clamp_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor,
    float clamp_min,
    float clamp_max) {
    
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight.size(0);
    
    auto output = torch::empty({batch_size, hidden_size}, input.options());
    
    const int threads = 256;
    const int blocks_x = (hidden_size + threads - 1) / threads;
    dim3 blocks(blocks_x, batch_size);
    
    fused_linear_scale_clamp_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size,
        scale_factor,
        clamp_min,
        clamp_max
    );
    
    return output;
}
"""

fused_linear_ops_cpp_source = """
torch::Tensor fused_linear_scale_clamp_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor,
    float clamp_min,
    float clamp_max);
"""

# Custom LogSumExp kernel
logsumexp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void logsumexp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int dim_size) {
    
    extern __shared__ float shared_mem[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Find max value for numerical stability
    float max_val = -INFINITY;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        max_val = fmaxf(max_val, input[batch_idx * dim_size + i]);
    }
    
    // Reduce max across block
    shared_mem[tid] = max_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + s]);
        }
        __syncthreads();
    }
    
    max_val = shared_mem[0];
    
    // Compute sum of exp(x - max)
    float sum = 0.0f;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        sum += expf(input[batch_idx * dim_size + i] - max_val);
    }
    
    // Reduce sum across block
    shared_mem[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[batch_idx] = logf(shared_mem[0]) + max_val;
    }
}

torch::Tensor logsumexp_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int dim_size = input.size(1);
    
    auto output = torch::empty({batch_size, 1}, input.options());
    
    const int threads = 256;
    const int blocks = batch_size;
    size_t shared_size = threads * sizeof(float);
    
    logsumexp_kernel<<<blocks, threads, shared_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim_size
    );
    
    return output;
}
"""

logsumexp_cpp_source = "torch::Tensor logsumexp_cuda(torch::Tensor input);"

# Fused Mish activation and multiplication kernel
mish_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

__device__ __inline__ float mish(float x) {
    return x * tanhf(log1pf(expf(x)));
}

__global__ void mish_mul_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = input[idx];
        output[idx] = x * mish(x);
    }
}

torch::Tensor mish_mul_cuda(torch::Tensor input) {
    int size = input.numel();
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    mish_mul_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

mish_mul_cpp_source = "torch::Tensor mish_mul_cuda(torch::Tensor input);"

# Load all custom operators
fused_linear_ops = load_inline(
    name="fused_linear_ops",
    cpp_sources=fused_linear_ops_cpp_source,
    cuda_sources=fused_linear_ops_source,
    functions=["fused_linear_scale_clamp_cuda"],
    verbose=True,
)

logsumexp_op = load_inline(
    name="logsumexp_op",
    cpp_sources=logsumexp_cpp_source,
    cuda_sources=logsumexp_source,
    functions=["logsumexp_cuda"],
    verbose=True,
)

mish_mul_op = load_inline(
    name="mish_mul_op",
    cpp_sources=mish_mul_cpp_source,
    cuda_sources=mish_mul_source,
    functions=["mish_mul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
        # Initialize weights similar to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=torch.nn.init.calculate_gain('linear'))
        
        self.fused_linear_ops = fused_linear_ops
        self.logsumexp_op = logsumexp_op
        self.mish_mul_op = mish_mul_op
    
    def forward(self, x):
        x = x.cuda()
        x = self.fused_linear_ops.fused_linear_scale_clamp_cuda(
            x, self.weight, self.bias, 
            self.scale_factor, self.clamp_min, self.clamp_max
        )
        x = self.logsumexp_op.logsumexp_cuda(x)
        x = self.mish_mul_op.mish_mul_cuda(x)
        return x

batch_size = 1024
input_size = 8192
hidden_size = 8192
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]