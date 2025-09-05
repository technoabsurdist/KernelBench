import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused concatenate + linear + tanh
fused_rnn_cell_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void tanh_kernel(float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(output[idx]);
    }
}

torch::Tensor fused_rnn_cell_cuda(
    torch::Tensor input, 
    torch::Tensor hidden,
    torch::Tensor weight_i2h,
    torch::Tensor bias_i2h) {
    
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = hidden.size(1);
    int combined_size = input_size + hidden_size;
    
    // Allocate output tensor
    auto output = torch::zeros({batch_size, hidden_size}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Create handle for cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Prepare for GEMM: output = [input, hidden] @ weight_i2h.T + bias_i2h
    // We'll do this in two parts to avoid explicit concatenation
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Part 1: Compute input @ weight_i2h[:input_size].T
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                hidden_size, batch_size, input_size,
                &alpha,
                weight_i2h.data_ptr<float>(), combined_size,
                input.data_ptr<float>(), input_size,
                &beta,
                output.data_ptr<float>(), hidden_size);
    
    // Part 2: Add hidden @ weight_i2h[input_size:].T
    beta = 1.0f;  // Add to existing result
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                hidden_size, batch_size, hidden_size,
                &alpha,
                weight_i2h.data_ptr<float>() + input_size, combined_size,
                hidden.data_ptr<float>(), hidden_size,
                &beta,
                output.data_ptr<float>(), hidden_size);
    
    // Add bias
    cublasSger(handle,
               hidden_size, batch_size,
               &alpha,
               bias_i2h.data_ptr<float>(), 1,
               torch::ones({batch_size}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).data_ptr<float>(), 1,
               output.data_ptr<float>(), hidden_size);
    
    // Apply tanh activation
    const int block_size = 256;
    const int num_blocks = (batch_size * hidden_size + block_size - 1) / block_size;
    tanh_kernel<<<num_blocks, block_size>>>(output.data_ptr<float>(), batch_size * hidden_size);
    
    cublasDestroy(handle);
    
    return output;
}
"""

fused_rnn_cell_cpp_source = """
torch::Tensor fused_rnn_cell_cuda(
    torch::Tensor input, 
    torch::Tensor hidden,
    torch::Tensor weight_i2h,
    torch::Tensor bias_i2h);
"""

# Define optimized GEMM kernel for output layer
optimized_gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

torch::Tensor optimized_gemm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    int batch_size = input.size(0);
    int input_dim = input.size(1);
    int output_dim = weight.size(0);
    
    auto output = torch::zeros({batch_size, output_dim}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Compute output = input @ weight.T + bias
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                output_dim, batch_size, input_dim,
                &alpha,
                weight.data_ptr<float>(), input_dim,
                input.data_ptr<float>(), input_dim,
                &beta,
                output.data_ptr<float>(), output_dim);
    
    // Add bias using broadcasting
    cublasSger(handle,
               output_dim, batch_size,
               &alpha,
               bias.data_ptr<float>(), 1,
               torch::ones({batch_size}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).data_ptr<float>(), 1,
               output.data_ptr<float>(), output_dim);
    
    cublasDestroy(handle);
    
    return output;
}
"""

optimized_gemm_cpp_source = """
torch::Tensor optimized_gemm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_rnn_cell = load_inline(
    name="fused_rnn_cell",
    cpp_sources=fused_rnn_cell_cpp_source,
    cuda_sources=fused_rnn_cell_source,
    functions=["fused_rnn_cell_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

optimized_gemm = load_inline(
    name="optimized_gemm",
    cpp_sources=optimized_gemm_cpp_source,
    cuda_sources=optimized_gemm_source,
    functions=["optimized_gemm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden = torch.randn((batch_size, hidden_size))
        
        # Define the RNN cell components
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        
        # Store custom kernels
        self.fused_rnn_cell = fused_rnn_cell
        self.optimized_gemm = optimized_gemm
    
    def forward(self, x: torch.Tensor, initial_hidden=None) -> torch.Tensor:
        if initial_hidden is not None:
            self.hidden.copy_(initial_hidden)
        self.hidden = self.hidden.to(x.device)
        
        # Use fused kernel for concatenate + linear + tanh
        self.hidden = self.fused_rnn_cell.fused_rnn_cell_cuda(
            x, self.hidden, self.i2h.weight, self.i2h.bias
        )
        
        # Use optimized GEMM for output layer
        output = self.optimized_gemm.optimized_gemm_cuda(
            self.hidden, self.h2o.weight, self.h2o.bias
        )
        
        return output

batch_size = 256
input_size = 16384
hidden_size = 16384
output_size = 8192
sequence_length = 256

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda(), torch.rand(batch_size, hidden_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, output_size]