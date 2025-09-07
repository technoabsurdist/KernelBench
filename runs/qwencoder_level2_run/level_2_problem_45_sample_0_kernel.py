import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Gemm + Sigmoid + Gemm + LogSumExp
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

__global__ void sigmoid_activation_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        data[idx] = 1.0f / (1.0f + expf(-val));
    }
}

__global__ void logsumexp_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Shared memory for reduction
    extern __shared__ float shared_data[];
    
    float max_val = -INFINITY;
    // Find maximum value in the row
    for (int i = tid; i < cols; i += block_size) {
        float val = input[row * cols + i];
        max_val = fmaxf(max_val, val);
    }
    shared_data[tid] = max_val;
    __syncthreads();
    
    // Reduce to find maximum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    
    float row_max = shared_data[0];
    if (row_max == -INFINITY) {
        if (tid == 0) output[row] = -INFINITY;
        return;
    }
    
    // Compute sum of exponentials
    float sum_exp = 0.0f;
    for (int i = tid; i < cols; i += block_size) {
        sum_exp += expf(input[row * cols + i] - row_max);
    }
    shared_data[tid] = sum_exp;
    __syncthreads();
    
    // Reduce to compute sum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    // Compute final logsumexp
    if (tid == 0) {
        output[row] = logf(shared_data[0]) + row_max;
    }
}

torch::Tensor fused_gemm_sigmoid_gemm_logsumexp(
    torch::Tensor input,
    torch::Tensor weight1,
    torch::Tensor bias1,
    torch::Tensor weight2,
    torch::Tensor bias2) {
    
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto hidden_size = weight1.size(0);
    auto output_size = weight2.size(0);
    
    // Create output tensor
    auto output = torch::zeros({batch_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Get cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Allocate intermediate tensors
    auto hidden = torch::zeros({batch_size, hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto activated = torch::zeros({batch_size, hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto result = torch::zeros({batch_size, output_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Gemm 1: input @ weight1.T + bias1
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                hidden_size, batch_size, input_size,
                &alpha,
                weight1.data_ptr<float>(), input_size,
                input.data_ptr<float>(), input_size,
                &beta,
                hidden.data_ptr<float>(), hidden_size);
    
    // Add bias1
    float* hidden_ptr = hidden.data_ptr<float>();
    float* bias1_ptr = bias1.data_ptr<float>();
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            hidden_ptr[i * hidden_size + j] += bias1_ptr[j];
        }
    }
    
    // Sigmoid activation
    int hidden_elements = batch_size * hidden_size;
    int block_size = 256;
    int num_blocks = (hidden_elements + block_size - 1) / block_size;
    sigmoid_activation_kernel<<<num_blocks, block_size>>>(hidden.data_ptr<float>(), hidden_elements);
    
    // Gemm 2: activated @ weight2.T + bias2
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                output_size, batch_size, hidden_size,
                &alpha,
                weight2.data_ptr<float>(), hidden_size,
                hidden.data_ptr<float>(), hidden_size,
                &beta,
                result.data_ptr<float>(), output_size);
    
    // Add bias2
    float* result_ptr = result.data_ptr<float>();
    float* bias2_ptr = bias2.data_ptr<float>();
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < output_size; j++) {
            result_ptr[i * output_size + j] += bias2_ptr[j];
        }
    }
    
    // LogSumExp
    int lse_block_size = 256;
    size_t shared_mem_size = lse_block_size * sizeof(float);
    logsumexp_kernel<<<batch_size, lse_block_size, shared_mem_size>>>(
        result.data_ptr<float>(), output.data_ptr<float>(), batch_size, output_size);
    
    // Cleanup
    cublasDestroy(handle);
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_gemm_sigmoid_gemm_logsumexp(
    torch::Tensor input,
    torch::Tensor weight1,
    torch::Tensor bias1,
    torch::Tensor weight2,
    torch::Tensor bias2);
"""

# Compile the inline CUDA code
fused_module = load_inline(
    name="fused_gemm_sigmoid_gemm_logsumexp_module",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_gemm_sigmoid_gemm_logsumexp"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Gemm + Sigmoid + Gemm + LogSumExp
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weight1 = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias1 = nn.Parameter(torch.randn(hidden_size))
        self.weight2 = nn.Parameter(torch.randn(output_size, hidden_size))
        self.bias2 = nn.Parameter(torch.randn(output_size))
        
        # Load the fused CUDA kernel
        self.fused_op = fused_module

    def forward(self, x):
        return self.fused_op.fused_gemm_sigmoid_gemm_logsumexp(
            x, self.weight1, self.bias1, self.weight2, self.bias2)