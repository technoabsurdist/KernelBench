import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// Device function for LeakyReLU
__device__ __forceinline__ float leaky_relu(float x, float negative_slope) {
    return (x > 0.0f) ? x : negative_slope * x;
}

// Device function for GELU
__device__ __forceinline__ float gelu(float x) {
    return x * 0.5f * (1.0f + erff(x * 0.70710678118f)); // 0.7071... is 1/sqrt(2)
}

// Kernel to perform fused LogSumExp and activations
__global__ void fused_logsumexp_activations_kernel(const float* input, float* output, int rows, int cols) {
    // Each block processes one row
    int row_idx = blockIdx.x;
    if (row_idx >= rows) {
        return;
    }

    const float* input_row = input + row_idx * cols;
    
    extern __shared__ float sdata[];

    // Step 1: Find the maximum value in the row for numerical stability
    float thread_max = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        thread_max = fmaxf(thread_max, input_row[i]);
    }
    sdata[threadIdx.x] = thread_max;
    __syncthreads();

    // Parallel reduction to find the max in the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }
    const float max_val = sdata[0];
    __syncthreads();

    // Step 2: Calculate sum(exp(x - max_val))
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        thread_sum += expf(input_row[i] - max_val);
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // Parallel reduction to find the sum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Step 3: Final calculation and apply activations
    if (threadIdx.x == 0) {
        const float sum_val = sdata[0];
        float result = max_val + logf(sum_val);

        // Apply the chain of activations
        const float negative_slope = 0.01f;
        // LeakyReLU
        result = leaky_relu(result, negative_slope);
        // LeakyReLU
        result = leaky_relu(result, negative_slope);
        // GELU
        result = gelu(result);
        // GELU
        result = gelu(result);

        output[row_idx] = result;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D");

    const int rows = input.size(0);
    const int cols = input.size(1);

    auto output = torch::zeros({rows, 1}, input.options());

    const int block_size = 256;
    const int num_blocks = rows;
    const int shared_mem_size = block_size * sizeof(float);

    fused_logsumexp_activations_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );
    
    return output;
}
"""

fused_op_cpp_source = "torch::Tensor fused_op_cuda(torch::Tensor input);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), followed by a fused kernel for 
    LogSumExp, LeakyReLU, LeakyReLU, GELU, and GELU activations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        # The linear layer is kept as it is highly optimized in PyTorch (cuBLAS)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # The subsequent operations are fused into a single custom CUDA kernel
        self.fused_logsumexp_activations = fused_op.fused_op_cuda

    def forward(self, x):
        # Gemm
        x = self.linear(x)
        # Fused LogSumExp and activations
        x = self.fused_logsumexp_activations(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]