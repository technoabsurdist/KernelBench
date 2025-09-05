import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Linear + ReLU
fused_linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void add_bias_relu_kernel(float* output, const float* bias, int m, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < m && col < n) {
        int idx = row * n + col;
        float val = output[idx] + bias[col];
        output[idx] = fmaxf(0.0f, val);
    }
}

__global__ void add_bias_kernel(float* output, const float* bias, int m, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < m && col < n) {
        int idx = row * n + col;
        output[idx] = output[idx] + bias[col];
    }
}

torch::Tensor fused_linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto m = input.size(0);
    auto k = input.size(1);
    auto n = weight.size(0);
    
    auto output = torch::zeros({m, n}, input.options());
    
    // Use cuBLAS for matrix multiplication (input @ weight.T)
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Perform C = A * B^T where A is input, B is weight
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                weight.data_ptr<float>(), k,
                input.data_ptr<float>(), k,
                &beta,
                output.data_ptr<float>(), n);
    
    cublasDestroy(handle);
    
    // Add bias and apply ReLU in a single kernel
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    
    add_bias_relu_kernel<<<grid, block>>>(
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        m, n
    );
    
    return output;
}

torch::Tensor linear_no_activation_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto m = input.size(0);
    auto k = input.size(1);
    auto n = weight.size(0);
    
    auto output = torch::zeros({m, n}, input.options());
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                weight.data_ptr<float>(), k,
                input.data_ptr<float>(), k,
                &beta,
                output.data_ptr<float>(), n);
    
    cublasDestroy(handle);
    
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    
    add_bias_kernel<<<grid, block>>>(
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        m, n
    );
    
    return output;
}
"""

fused_linear_relu_cpp_source = """
torch::Tensor fused_linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor linear_no_activation_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_linear_relu_cpp_source,
    cuda_sources=fused_linear_relu_source,
    functions=["fused_linear_relu_cuda", "linear_no_activation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(ModelNew, self).__init__()
        
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        
        current_input_size = input_size
        
        for hidden_size in hidden_layer_sizes:
            self.weights.append(nn.Parameter(torch.randn(hidden_size, current_input_size)))
            self.biases.append(nn.Parameter(torch.randn(hidden_size)))
            current_input_size = hidden_size
        
        self.weights.append(nn.Parameter(torch.randn(output_size, current_input_size)))
        self.biases.append(nn.Parameter(torch.randn(output_size)))
        
        self.fused_ops = fused_ops
        
        # Initialize weights similar to nn.Linear
        for w, b in zip(self.weights, self.biases):
            nn.init.kaiming_uniform_(w, a=0)
            fan_in = w.size(1)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(b, -bound, bound)
    
    def forward(self, x):
        # Apply fused linear+relu for all hidden layers
        for i in range(len(self.weights) - 1):
            x = self.fused_ops.fused_linear_relu_cuda(x, self.weights[i], self.biases[i])
        
        # Apply final linear layer without activation
        x = self.fused_ops.linear_no_activation_cuda(x, self.weights[-1], self.biases[-1])
        
        return x


def get_inputs():
    batch_size = 1024
    input_size = 8192
    return [torch.rand(batch_size, input_size).cuda()]


def get_init_inputs():
    input_size = 8192
    hidden_layer_sizes = [1024] * 16
    output_size = 8192
    return [input_size, hidden_layer_sizes, output_size]