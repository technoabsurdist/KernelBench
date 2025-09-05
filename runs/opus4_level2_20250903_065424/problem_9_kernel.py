import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear + elementwise ops + relu
fused_linear_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_bias_subtract_multiply_relu_kernel(
    float* output, 
    const float* bias,
    float subtract_val,
    float multiply_val,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = output[idx] + bias[idx % blockDim.y];
        val = (val - subtract_val) * multiply_val;
        output[idx] = fmaxf(0.0f, val);
    }
}

torch::Tensor fused_linear_ops_cuda(
    torch::Tensor input,
    torch::Tensor weight, 
    torch::Tensor bias,
    float subtract_val,
    float multiply_val) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    // Perform matrix multiplication using cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Compute output = input @ weight.T
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, batch_size, in_features,
                &alpha,
                weight.data_ptr<float>(), in_features,
                input.data_ptr<float>(), in_features,
                &beta,
                output.data_ptr<float>(), out_features);
    
    cublasDestroy(handle);
    
    // Fused bias addition, subtraction, multiplication and ReLU
    const int threads = 256;
    const int blocks = (batch_size * out_features + threads - 1) / threads;
    
    dim3 block_dim(threads, out_features);
    fused_bias_subtract_multiply_relu_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        subtract_val,
        multiply_val,
        batch_size * out_features
    );
    
    return output;
}
"""

fused_linear_ops_cpp_source = """
torch::Tensor fused_linear_ops_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_val,
    float multiply_val);
"""

# Compile the inline CUDA code
fused_linear_ops = load_inline(
    name="fused_linear_ops",
    cpp_sources=fused_linear_ops_cpp_source,
    cuda_sources=fused_linear_ops_source,
    functions=["fused_linear_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernels for linear layer + elementwise ops + ReLU
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self.fused_linear_ops = fused_linear_ops
        
        # Initialize weights similar to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=torch.nn.init.calculate_gain('relu'))
        nn.init.uniform_(self.bias, -1.0 / (in_features ** 0.5), 1.0 / (in_features ** 0.5))

    def forward(self, x):
        return self.fused_linear_ops.fused_linear_ops_cuda(
            x, 
            self.weight, 
            self.bias,
            self.subtract_value,
            self.multiply_value
        )

batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]