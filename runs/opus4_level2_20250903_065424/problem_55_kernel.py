import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused max pooling, sum, and scaling
fused_pool_sum_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void fused_pool_sum_scale_kernel(
    const float* input, 
    float* output,
    int batch_size,
    int in_features,
    int kernel_size,
    float scale_factor
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int pooled_size = in_features / kernel_size;
    
    // Shared memory for partial sums
    extern __shared__ float sdata[];
    
    float thread_sum = 0.0f;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Each thread handles multiple elements
    for (int i = tid; i < pooled_size; i += num_threads) {
        float max_val = -1e38f;
        
        // Compute max pooling for this position
        for (int k = 0; k < kernel_size; k++) {
            int input_idx = batch_idx * in_features + i * kernel_size + k;
            if (input_idx < batch_idx * in_features + in_features) {
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }
        
        thread_sum += max_val;
    }
    
    // Store partial sum in shared memory
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[batch_idx] = sdata[0] * scale_factor;
    }
}

torch::Tensor fused_pool_sum_scale_cuda(
    torch::Tensor input,
    int kernel_size,
    float scale_factor
) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    
    auto output = torch::zeros({batch_size}, input.options());
    
    const int threads = 256;
    const int blocks = batch_size;
    size_t shared_mem_size = threads * sizeof(float);
    
    fused_pool_sum_scale_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        kernel_size,
        scale_factor
    );
    
    return output;
}
"""

fused_pool_sum_scale_cpp_source = """
torch::Tensor fused_pool_sum_scale_cuda(
    torch::Tensor input,
    int kernel_size,
    float scale_factor
);
"""

# Custom CUDA kernel for optimized matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

torch::Tensor matmul_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    // Using cuBLAS for optimized matrix multiplication
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Perform matrix multiplication: output = input @ weight.T
    cublasSgemm(handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                out_features,
                batch_size,
                in_features,
                &alpha,
                weight.data_ptr<float>(),
                in_features,
                input.data_ptr<float>(),
                in_features,
                &beta,
                output.data_ptr<float>(),
                out_features);
    
    // Add bias
    const int threads = 256;
    const int blocks = (batch_size * out_features + threads - 1) / threads;
    
    auto add_bias = [=] __device__ (int idx) {
        if (idx < batch_size * out_features) {
            int feature_idx = idx % out_features;
            output.data_ptr<float>()[idx] += bias.data_ptr<float>()[feature_idx];
        }
    };
    
    // Simple kernel to add bias
    static auto add_bias_kernel = [] __global__ (float* output, const float* bias, int batch_size, int out_features, int total_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < total_size) {
            int feature_idx = idx % out_features;
            output[idx] += bias[feature_idx];
        }
    };
    
    add_bias_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size,
        out_features,
        batch_size * out_features
    );
    
    cublasDestroy(handle);
    
    return output;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_pool_sum_scale = load_inline(
    name="fused_pool_sum_scale",
    cpp_sources=fused_pool_sum_scale_cpp_source,
    cuda_sources=fused_pool_sum_scale_source,
    functions=["fused_pool_sum_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

matmul_op = load_inline(
    name="matmul_op",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.fused_pool_sum_scale = fused_pool_sum_scale
        self.matmul_op = matmul_op
        
    def forward(self, x):
        x = x.cuda()
        self.weight = self.weight.cuda()
        self.bias = self.bias.cuda()
        
        # Custom matmul with bias
        x = self.matmul_op.matmul_cuda(x, self.weight, self.bias)
        
        # Fused max pooling, sum, and scaling
        x = self.fused_pool_sum_scale.fused_pool_sum_scale_cuda(x, self.kernel_size, self.scale_factor)
        
        return x

batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]