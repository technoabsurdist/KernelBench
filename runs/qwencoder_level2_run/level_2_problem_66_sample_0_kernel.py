import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + dropout + softmax
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

__global__ void fused_matmul_dropout_softmax_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    curandState* states,
    int batch_size,
    int in_features,
    int out_features,
    float dropout_p
) {
    int batch_idx = blockIdx.x;
    int feature_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (batch_idx >= batch_size || feature_idx >= out_features) return;
    
    // Initialize random state for this thread
    curandState local_state = states[batch_idx * out_features + feature_idx];
    
    // Compute matmul + bias for this output element
    float sum = 0.0f;
    for (int k = 0; k < in_features; k++) {
        sum += input[batch_idx * in_features + k] * weight[feature_idx * in_features + k];
    }
    sum += bias[feature_idx];
    
    // Apply dropout
    float rand_val = curand_uniform(&local_state);
    float dropped_value = (rand_val > dropout_p) ? (sum / (1.0f - dropout_p)) : 0.0f;
    
    // Store intermediate result
    output[batch_idx * out_features + feature_idx] = dropped_value;
    
    // Update random state
    states[batch_idx * out_features + feature_idx] = local_state;
}

__global__ void softmax_kernel(float* data, int batch_size, int out_features) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    // Find max value for numerical stability
    float max_val = -INFINITY;
    for (int i = 0; i < out_features; i++) {
        float val = data[batch_idx * out_features + i];
        if (val > max_val) max_val = val;
    }
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < out_features; i++) {
        float val = data[batch_idx * out_features + i];
        float exp_val = expf(val - max_val);
        data[batch_idx * out_features + i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    for (int i = 0; i < out_features; i++) {
        data[batch_idx * out_features + i] /= sum;
    }
}

__global__ void setup_kernel(curandState *state, unsigned long seed, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

torch::Tensor fused_matmul_dropout_softmax_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float dropout_p
) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Allocate and initialize random states
    auto state_size = batch_size * out_features;
    curandState* d_state;
    cudaMalloc(&d_state, state_size * sizeof(curandState));
    
    const int block_size = 256;
    const int num_blocks = (state_size + block_size - 1) / block_size;
    setup_kernel<<<num_blocks, block_size>>>(d_state, time(NULL), state_size);
    cudaDeviceSynchronize();
    
    // Launch fused kernel for matmul + dropout
    dim3 grid(batch_size, (out_features + 255) / 256);
    dim3 block(256);
    fused_matmul_dropout_softmax_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        d_state,
        batch_size,
        in_features,
        out_features,
        dropout_p
    );
    
    // Launch softmax kernel
    softmax_kernel<<<batch_size, 1>>>(
        output.data_ptr<float>(),
        batch_size,
        out_features
    );
    
    cudaFree(d_state);
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_matmul_dropout_softmax_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float dropout_p
);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_matmul_dropout_softmax_cuda"],
    verbose=False,
    extra_cflags=["-std=c++14"],
    extra_ldflags=["-lcurand"],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernels for matmul + dropout + softmax.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # Store the fused operation function
        self.fused_op = fused_ops

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return self.fused_op.fused_matmul_dropout_softmax_cuda(
            x, self.weight, self.bias, self.dropout_p
        )