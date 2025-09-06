import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for the fused operator
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_DIM 32

__global__ void fused_op_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ linear_weight, 
    const float* __restrict__ linear_bias,
    const float* __restrict__ bn_weight, 
    const float* __restrict__ bn_bias, 
    const float* __restrict__ bn_mean, 
    const float* __restrict__ bn_var,
    const float* __restrict__ custom_bias, 
    float* __restrict__ out,
    int batch_size, 
    int in_features, 
    int out_features,
    float bn_eps, 
    float divide_value
) {
    // Shared memory for tiles of x and linear_weight
    __shared__ float x_tile[TILE_DIM][TILE_DIM];
    __shared__ float w_tile[TILE_DIM][TILE_DIM];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row and column for the output element this thread computes
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float acc = 0.0f;

    // Loop over tiles of the input matrices to perform matrix multiplication
    for (int i = 0; i < (in_features + TILE_DIM - 1) / TILE_DIM; ++i) {
        int k_base = i * TILE_DIM;

        // Load a tile of x into shared memory
        int x_k = k_base + tx;
        if (row < batch_size && x_k < in_features) {
            x_tile[ty][tx] = x[row * in_features + x_k];
        } else {
            x_tile[ty][tx] = 0.0f;
        }

        // Load a tile of linear_weight into shared memory
        int w_row_for_load = blockIdx.x * TILE_DIM + ty;
        int w_k_for_load = k_base + tx;
        if (w_row_for_load < out_features && w_k_for_load < in_features) {
            w_tile[ty][tx] = linear_weight[w_row_for_load * in_features + w_k_for_load];
        } else {
            w_tile[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute dot product for the current tiles
        if (row < batch_size && col < out_features) {
            for (int k = 0; k < TILE_DIM; ++k) {
                // acc += x[row][k_base+k] * W[col][k_base+k]
                // x[row][k_base+k] is in x_tile[ty][k]
                // W[col][k_base+k] is in w_tile[tx][k]
                acc += x_tile[ty][k] * w_tile[tx][k];
            }
        }
        __syncthreads();
    }

    // After matmul, apply the rest of the fused operations
    if (row < batch_size && col < out_features) {
        // 1. Add linear bias
        acc += linear_bias[col];

        // 2. Batch Normalization (inference mode)
        float inv_std = rsqrtf(bn_var[col] + bn_eps);
        acc = bn_weight[col] * (acc - bn_mean[col]) * inv_std + bn_bias[col];

        // 3. Custom Bias Addition
        acc += custom_bias[0];

        // 4. Division
        acc /= divide_value;

        // 5. Swish Activation
        float sigmoid_val = 1.0f / (1.0f + expf(-acc));
        out[row * out_features + col] = acc * sigmoid_val;
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor x,
    torch::Tensor linear_weight,
    torch::Tensor linear_bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    torch::Tensor custom_bias,
    float bn_eps,
    float divide_value
) {
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    // Add checks for all other tensors as well for a robust implementation

    const auto batch_size = x.size(0);
    const auto in_features = x.size(1);
    const auto out_features = linear_weight.size(0);

    auto out = torch::empty({batch_size, out_features}, x.options());

    dim3 block_dim(TILE_DIM, TILE_DIM);
    dim3 grid_dim(
        (out_features + TILE_DIM - 1) / TILE_DIM,
        (batch_size + TILE_DIM - 1) / TILE_DIM
    );

    fused_op_kernel<<<grid_dim, block_dim>>>(
        x.data_ptr<float>(),
        linear_weight.data_ptr<float>(),
        linear_bias.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        custom_bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        bn_eps,
        divide_value
    );
    
    return out;
}
"""

# C++ source for function signature declaration
fused_op_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor x,
    torch::Tensor linear_weight,
    torch::Tensor linear_bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    torch::Tensor custom_bias,
    float bn_eps,
    float divide_value
);
"""

# JIT compile the CUDA and C++ code
fused_op_module = load_inline(
    name="fused_op_module",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses matrix multiplication, batch normalization, bias addition, 
    division, and Swish activation into a single custom CUDA kernel.
    
    NOTE: This custom kernel is designed for inference (`eval()` mode) only.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        # Store original modules to hold parameters and state
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Store scalar values needed by the kernel
        self.divide_value = divide_value
        self.bn_eps = bn_eps
        
        # Store the compiled custom operator
        self.fused_op = fused_op_module

    def forward(self, x):
        if self.training:
            # Fallback to the original PyTorch implementation for training
            # to ensure correct gradient computation and BatchNorm state updates.
            x = self.matmul(x)
            x = self.bn(x)
            x = x + self.bias
            x = x / self.divide_value
            x = x * torch.sigmoid(x)
            return x
        else:
            # Use the high-performance fused CUDA kernel for inference
            return self.fused_op.fused_op_cuda(
                x,
                self.matmul.weight,
                self.matmul.bias,
                self.bn.weight,
                self.bn.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bias,
                self.bn_eps,
                self.divide_value
            )