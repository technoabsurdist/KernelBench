import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Linear + ReLU
fused_linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_linear_relu_kernel(
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

torch::Tensor fused_linear_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int output_size = weight.size(0);
    
    auto output = torch::zeros({batch_size, output_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    dim3 block_size(256);
    dim3 num_blocks(batch_size, (output_size + block_size.x - 1) / block_size.x);
    
    fused_linear_relu_kernel<<<num_blocks, block_size>>>(
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

fused_linear_relu_cpp_source = """
torch::Tensor fused_linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code for fused Linear + ReLU
fused_linear_relu = load_inline(
    name="fused_linear_relu",
    cpp_sources=fused_linear_relu_cpp_source,
    cuda_sources=fused_linear_relu_source,
    functions=["fused_linear_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for the final linear layer
final_linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void final_linear_kernel(
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

torch::Tensor final_linear_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int output_size = weight.size(0);
    
    auto output = torch::zeros({batch_size, output_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    dim3 block_size(256);
    dim3 num_blocks(batch_size, (output_size + block_size.x - 1) / block_size.x);
    
    final_linear_kernel<<<num_blocks, block_size>>>(
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

final_linear_cpp_source = """
torch::Tensor final_linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code for final linear layer
final_linear = load_inline(
    name="final_linear",
    cpp_sources=final_linear_cpp_source,
    cuda_sources=final_linear_source,
    functions=["final_linear_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()
        
        self.fused_linear_relu = fused_linear_relu
        self.final_linear = final_linear
        
        # Create parameters for the first layer
        self.first_weight = nn.Parameter(torch.randn(hidden_layer_sizes[0], input_size))
        self.first_bias = nn.Parameter(torch.randn(hidden_layer_sizes[0]))
        
        # Create parameters for hidden layers
        self.hidden_weights = nn.ParameterList()
        self.hidden_biases = nn.ParameterList()
        
        for i in range(len(hidden_layer_sizes) - 1):
            weight = nn.Parameter(torch.randn(hidden_layer_sizes[i+1], hidden_layer_sizes[i]))
            bias = nn.Parameter(torch.randn(hidden_layer_sizes[i+1]))
            self.hidden_weights.append(weight)
            self.hidden_biases.append(bias)
        
        # Create parameters for the final layer
        self.final_weight = nn.Parameter(torch.randn(output_size, hidden_layer_sizes[-1]))
        self.final_bias = nn.Parameter(torch.randn(output_size))
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # First layer with ReLU
        x = self.fused_linear_relu.fused_linear_relu_cuda(x, self.first_weight, self.first_bias)
        
        # Hidden layers with ReLU
        for weight, bias in zip(self.hidden_weights, self.hidden_biases):
            x = self.fused_linear_relu.fused_linear_relu_cuda(x, weight, bias)
        
        # Final layer without ReLU
        x = self.final_linear.final_linear_cuda(x, self.final_weight, self.final_bias)
        
        return x

# Test code
batch_size = 128
input_size = 16384
hidden_layer_sizes = [32768, 32768]
output_size = 16384

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]