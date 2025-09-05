import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused Linear + Sigmoid kernel
fused_linear_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void add_bias_sigmoid_kernel(float* output, const float* bias, int M, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = M * N;
    
    if (tid < total_elements) {
        int col = tid % N;
        float val = output[tid] + bias[col];
        output[tid] = 1.0f / (1.0f + expf(-val));
    }
}

torch::Tensor fused_linear_sigmoid_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto M = input.size(0);
    auto K = input.size(1);
    auto N = weight.size(0);
    
    auto output = torch::empty({M, N}, input.options());
    
    // Perform GEMM using cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f, beta = 0.0f;
    
    // Note: cublas uses column-major, so we compute output = input * weight^T
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                weight.data_ptr<float>(), K,
                input.data_ptr<float>(), K,
                &beta,
                output.data_ptr<float>(), N);
    
    cublasDestroy(handle);
    
    // Add bias and apply sigmoid
    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;
    add_bias_sigmoid_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(), 
        bias.data_ptr<float>(),
        M, N
    );
    
    return output;
}
"""

fused_linear_sigmoid_cpp_source = """
torch::Tensor fused_linear_sigmoid_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Fused Linear + LogSumExp kernel
fused_linear_logsumexp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <float.h>

__global__ void fused_bias_logsumexp_kernel(const float* input, const float* bias, float* output, int M, int N) {
    extern __shared__ float shared_mem[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= M) return;
    
    float max_val = -FLT_MAX;
    
    // Find max value for numerical stability
    for (int i = tid; i < N; i += blockDim.x) {
        float val = input[row * N + i] + bias[i];
        max_val = fmaxf(max_val, val);
    }
    
    // Reduce max within block
    shared_mem[tid] = max_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + s]);
        }
        __syncthreads();
    }
    
    max_val = shared_mem[0];
    
    // Compute sum of exp(x - max)
    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = input[row * N + i] + bias[i];
        sum += expf(val - max_val);
    }
    
    // Reduce sum within block
    shared_mem[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[row] = max_val + logf(shared_mem[0]);
    }
}

torch::Tensor fused_linear_logsumexp_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto M = input.size(0);
    auto K = input.size(1);
    auto N = weight.size(0);
    
    auto temp = torch::empty({M, N}, input.options());
    
    // Perform GEMM using cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f, beta = 0.0f;
    
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                weight.data_ptr<float>(), K,
                input.data_ptr<float>(), K,
                &beta,
                temp.data_ptr<float>(), N);
    
    cublasDestroy(handle);
    
    // Compute LogSumExp with bias
    auto output = torch::empty({M}, input.options());
    
    int threads = 256;
    int blocks = M;
    size_t shared_mem_size = threads * sizeof(float);
    
    fused_bias_logsumexp_kernel<<<blocks, threads, shared_mem_size>>>(
        temp.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N
    );
    
    return output;
}
"""

fused_linear_logsumexp_cpp_source = """
torch::Tensor fused_linear_logsumexp_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA kernels
fused_linear_sigmoid = load_inline(
    name="fused_linear_sigmoid",
    cpp_sources=fused_linear_sigmoid_cpp_source,
    cuda_sources=fused_linear_sigmoid_source,
    functions=["fused_linear_sigmoid_cuda"],
    extra_cuda_cflags=["-lcublas"],
    verbose=True,
)

fused_linear_logsumexp = load_inline(
    name="fused_linear_logsumexp",
    cpp_sources=fused_linear_logsumexp_cpp_source,
    cuda_sources=fused_linear_logsumexp_source,
    functions=["fused_linear_logsumexp_cuda"],
    extra_cuda_cflags=["-lcublas"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.weight1 = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias1 = nn.Parameter(torch.randn(hidden_size))
        self.weight2 = nn.Parameter(torch.randn(output_size, hidden_size))
        self.bias2 = nn.Parameter(torch.randn(output_size))
        
        # Initialize weights similar to nn.Linear
        nn.init.kaiming_uniform_(self.weight1, a=torch.nn.init.calculate_gain('linear'))
        nn.init.kaiming_uniform_(self.weight2, a=torch.nn.init.calculate_gain('linear'))
        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)
        
        self.fused_linear_sigmoid = fused_linear_sigmoid
        self.fused_linear_logsumexp = fused_linear_logsumexp

    def forward(self, x):
        x = x.cuda()
        x = self.fused_linear_sigmoid.fused_linear_sigmoid_cuda(x, self.weight1, self.bias1)
        x = self.fused_linear_logsumexp.fused_linear_logsumexp_cuda(x, self.weight2, self.bias2)
        return x

batch_size = 16384
input_size = 2048
hidden_size = 4096
output_size = 1024

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, output_size]