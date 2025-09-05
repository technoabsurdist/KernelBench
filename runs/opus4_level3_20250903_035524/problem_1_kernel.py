import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Linear+ReLU
fused_linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void add_bias_relu_kernel(float* output, const float* bias, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = m * n;
    
    if (idx < total_size) {
        int col = idx % n;
        float val = output[idx] + bias[col];
        output[idx] = fmaxf(val, 0.0f);
    }
}

__global__ void add_bias_kernel(float* output, const float* bias, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = m * n;
    
    if (idx < total_size) {
        int col = idx % n;
        output[idx] = output[idx] + bias[col];
    }
}

torch::Tensor fused_linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto m = input.size(0);
    auto k = input.size(1);
    auto n = weight.size(0);
    
    auto output = torch::zeros({m, n}, input.options());
    
    // Use cuBLAS for matrix multiplication
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Perform C = A * B^T
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                weight.data_ptr<float>(), k,
                input.data_ptr<float>(), k,
                &beta,
                output.data_ptr<float>(), n);
    
    cublasDestroy(handle);
    
    // Add bias and apply ReLU
    const int block_size = 256;
    const int num_blocks = (m * n + block_size - 1) / block_size;
    add_bias_relu_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        m, n);
    
    return output;
}

torch::Tensor linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto m = input.size(0);
    auto k = input.size(1);
    auto n = weight.size(0);
    
    auto output = torch::zeros({m, n}, input.options());
    
    // Use cuBLAS for matrix multiplication
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Perform C = A * B^T
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                weight.data_ptr<float>(), k,
                input.data_ptr<float>(), k,
                &beta,
                output.data_ptr<float>(), n);
    
    cublasDestroy(handle);
    
    // Add bias
    const int block_size = 256;
    const int num_blocks = (m * n + block_size - 1) / block_size;
    add_bias_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        m, n);
    
    return output;
}
"""

fused_linear_relu_cpp_source = """
torch::Tensor fused_linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_linear_relu_cpp_source,
    cuda_sources=fused_linear_relu_source,
    functions=["fused_linear_relu_cuda", "linear_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()
        
        self.weights = []
        self.biases = []
        
        current_input_size = input_size
        
        for layer_size in layer_sizes:
            weight = nn.Parameter(torch.randn(layer_size, current_input_size))
            bias = nn.Parameter(torch.zeros(layer_size))
            nn.init.kaiming_uniform_(weight, a=0)
            self.weights.append(weight)
            self.biases.append(bias)
            current_input_size = layer_size
        
        # Final layer
        weight = nn.Parameter(torch.randn(output_size, current_input_size))
        bias = nn.Parameter(torch.zeros(output_size))
        nn.init.kaiming_uniform_(weight, a=0)
        self.weights.append(weight)
        self.biases.append(bias)
        
        # Register parameters
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            self.register_parameter(f'weight_{i}', w)
            self.register_parameter(f'bias_{i}', b)
        
        self.fused_ops = fused_ops
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Apply fused linear+relu for all but the last layer
        for i in range(len(self.weights) - 1):
            x = self.fused_ops.fused_linear_relu_cuda(x, self.weights[i], self.biases[i])
        
        # Final layer without ReLU
        x = self.fused_ops.linear_cuda(x, self.weights[-1], self.biases[-1])
        
        return x


def get_inputs():
    batch_size = 128
    input_size = 16384
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    input_size = 16384
    layer_sizes = [16384, 16384]
    output_size = 8192
    return [input_size, layer_sizes, output_size]