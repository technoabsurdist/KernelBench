import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + relu
fused_matmul_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_matmul_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int input_size,
    int output_size
) {
    int batch_idx = blockIdx.x;
    int output_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && output_idx < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += input[batch_idx * input_size + i] * weight[output_idx * input_size + i];
        }
        sum += bias[output_idx];
        // Apply ReLU
        output[batch_idx * output_size + output_idx] = fmaxf(0.0f, sum);
    }
}

torch::Tensor fused_matmul_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto output_size = weight.size(0);
    
    auto output = torch::zeros({batch_size, output_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int threads_per_block = 256;
    const dim3 blocks(batch_size, (output_size + threads_per_block - 1) / threads_per_block);
    
    fused_matmul_relu_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        output_size
    );
    
    return output;
}
"""

fused_matmul_relu_cpp_source = """
torch::Tensor fused_matmul_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code for fused matmul + relu
fused_matmul_relu = load_inline(
    name="fused_matmul_relu",
    cpp_sources=fused_matmul_relu_cpp_source,
    cuda_sources=fused_matmul_relu_source,
    functions=["fused_matmul_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for final matmul (without ReLU)
final_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void final_matmul_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int input_size,
    int output_size
) {
    int batch_idx = blockIdx.x;
    int output_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && output_idx < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += input[batch_idx * input_size + i] * weight[output_idx * input_size + i];
        }
        sum += bias[output_idx];
        output[batch_idx * output_size + output_idx] = sum;
    }
}

torch::Tensor final_matmul_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto output_size = weight.size(0);
    
    auto output = torch::zeros({batch_size, output_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int threads_per_block = 256;
    const dim3 blocks(batch_size, (output_size + threads_per_block - 1) / threads_per_block);
    
    final_matmul_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        output_size
    );
    
    return output;
}
"""

final_matmul_cpp_source = """
torch::Tensor final_matmul_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code for final matmul
final_matmul = load_inline(
    name="final_matmul",
    cpp_sources=final_matmul_cpp_source,
    cuda_sources=final_matmul_source,
    functions=["final_matmul_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()
        
        self.fused_matmul_relu = fused_matmul_relu
        self.final_matmul = final_matmul
        
        # Create layers with parameters
        self.layers = nn.ModuleList()
        current_input_size = input_size
        
        for i, layer_size in enumerate(layer_sizes):
            linear = nn.Linear(current_input_size, layer_size)
            self.layers.append(linear)
            current_input_size = layer_size
        
        # Final layer
        self.final_layer = nn.Linear(current_input_size, output_size)
        
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Process through hidden layers with fused matmul + relu
        for layer in self.layers:
            x = self.fused_matmul_relu.fused_matmul_relu_cuda(x, layer.weight, layer.bias)
        
        # Final layer without ReLU
        x = self.final_matmul.final_matmul_cuda(x, self.final_layer.weight, self.final_layer.bias)
        
        return x