import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Fused Linear+ReLU kernel
fused_linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 16

__global__ void fused_linear_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight, 
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_features, int out_features) {
    
    __shared__ float tile_input[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_weight[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int k = 0; k < (in_features + TILE_SIZE - 1) / TILE_SIZE; k++) {
        // Load input tile
        if (row < batch_size && k * TILE_SIZE + tx < in_features) {
            tile_input[ty][tx] = input[row * in_features + k * TILE_SIZE + tx];
        } else {
            tile_input[ty][tx] = 0.0f;
        }
        
        // Load weight tile (transposed access)
        if (col < out_features && k * TILE_SIZE + ty < in_features) {
            tile_weight[ty][tx] = weight[col * in_features + k * TILE_SIZE + ty];
        } else {
            tile_weight[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tile_input[ty][i] * tile_weight[i][tx];
        }
        
        __syncthreads();
    }
    
    // Write output with bias and ReLU
    if (row < batch_size && col < out_features) {
        float val = sum + bias[col];
        output[row * out_features + col] = fmaxf(0.0f, val);
    }
}

torch::Tensor fused_linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size(
        (out_features + TILE_SIZE - 1) / TILE_SIZE,
        (batch_size + TILE_SIZE - 1) / TILE_SIZE
    );
    
    fused_linear_relu_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_features, out_features
    );
    
    return output;
}
"""

fused_linear_relu_cpp_source = "torch::Tensor fused_linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Optimized Linear kernel (no activation)
linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void linear_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_features, int out_features) {
    
    __shared__ float tile_input[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_weight[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int k = 0; k < (in_features + TILE_SIZE - 1) / TILE_SIZE; k++) {
        // Load input tile
        if (row < batch_size && k * TILE_SIZE + tx < in_features) {
            tile_input[ty][tx] = input[row * in_features + k * TILE_SIZE + tx];
        } else {
            tile_input[ty][tx] = 0.0f;
        }
        
        // Load weight tile (transposed access)
        if (col < out_features && k * TILE_SIZE + ty < in_features) {
            tile_weight[ty][tx] = weight[col * in_features + k * TILE_SIZE + ty];
        } else {
            tile_weight[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tile_input[ty][i] * tile_weight[i][tx];
        }
        
        __syncthreads();
    }
    
    // Write output with bias
    if (row < batch_size && col < out_features) {
        output[row * out_features + col] = sum + bias[col];
    }
}

torch::Tensor linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size(
        (out_features + TILE_SIZE - 1) / TILE_SIZE,
        (batch_size + TILE_SIZE - 1) / TILE_SIZE
    );
    
    linear_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_features, out_features
    );
    
    return output;
}
"""

linear_cpp_source = "torch::Tensor linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Compile the inline CUDA code
fused_linear_relu = load_inline(
    name="fused_linear_relu",
    cpp_sources=fused_linear_relu_cpp_source,
    cuda_sources=fused_linear_relu_source,
    functions=["fused_linear_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

linear_optimized = load_inline(
    name="linear_optimized",
    cpp_sources=linear_cpp_source,
    cuda_sources=linear_source,
    functions=["linear_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(ModelNew, self).__init__()
        
        self.layers = nn.ModuleList()
        current_input_size = input_size
        
        # Create linear layers
        for hidden_size in hidden_layer_sizes:
            self.layers.append(nn.Linear(current_input_size, hidden_size))
            current_input_size = hidden_size
        
        self.layers.append(nn.Linear(current_input_size, output_size))
        
        self.num_hidden = len(hidden_layer_sizes)
        self.fused_linear_relu = fused_linear_relu
        self.linear_optimized = linear_optimized
    
    def forward(self, x):
        # Apply fused linear+relu for hidden layers
        for i in range(self.num_hidden):
            layer = self.layers[i]
            x = self.fused_linear_relu.fused_linear_relu_cuda(
                x.contiguous(), 
                layer.weight.contiguous(),
                layer.bias.contiguous()
            )
        
        # Apply final linear layer without activation
        final_layer = self.layers[-1]
        x = self.linear_optimized.linear_cuda(
            x.contiguous(),
            final_layer.weight.contiguous(), 
            final_layer.bias.contiguous()
        )
        
        return x

def get_inputs():
    batch_size = 128
    input_size = 16384
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    input_size = 16384
    hidden_layer_sizes = [32768, 32768]
    output_size = 16384
    return [input_size, hidden_layer_sizes, output_size]