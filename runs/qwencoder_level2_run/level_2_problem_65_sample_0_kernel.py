import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv + avg_pool + sigmoid + sum
fused_conv_pool_sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void sigmoid_activation(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        data[idx] = 1.0f / (1.0f + expf(-val));
    }
}

__global__ void avg_pool2d_and_sigmoid_and_sum(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int pool_size,
    int output_height,
    int output_width
) {
    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int thread_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || channel_idx >= channels) return;
    
    // Shared memory for reduction
    extern __shared__ float sdata[];
    
    float sum = 0.0f;
    
    // Each thread processes multiple pooling windows
    for (int ph = thread_idx; ph < output_height; ph += blockDim.x) {
        for (int pw = 0; pw < output_width; pw++) {
            float pool_sum = 0.0f;
            int count = 0;
            
            for (int ih = ph * pool_size; ih < (ph + 1) * pool_size && ih < input_height; ih++) {
                for (int iw = pw * pool_size; iw < (pw + 1) * pool_size && iw < input_width; iw++) {
                    int input_idx = batch_idx * (channels * input_height * input_width) +
                                   channel_idx * (input_height * input_width) +
                                   ih * input_width + iw;
                    pool_sum += input[input_idx];
                    count++;
                }
            }
            
            if (count > 0) {
                float avg_val = pool_sum / count;
                // Apply sigmoid
                float sigmoid_val = 1.0f / (1.0f + expf(-avg_val));
                sum += sigmoid_val;
            }
        }
    }
    
    sdata[thread_idx] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_idx < s) {
            sdata[thread_idx] += sdata[thread_idx + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (thread_idx == 0) {
        output[batch_idx] = sdata[0];
    }
}

torch::Tensor fused_conv_pool_sigmoid_sum_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int pool_size
) {
    // Get dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    // Convolution output dimensions
    int conv_out_height = input_height - kernel_size + 1;
    int conv_out_width = input_width - kernel_size + 1;
    
    // Pooling output dimensions
    int pool_out_height = conv_out_height / pool_size;
    int pool_out_width = conv_out_width / pool_size;
    
    // Allocate output tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto conv_output = torch::zeros({batch_size, out_channels, conv_out_height, conv_out_width}, options);
    auto final_output = torch::zeros({batch_size}, options);
    
    // Perform convolution using cuBLAS
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    
    // Im2col implementation for convolution
    int M = out_channels;
    int N = batch_size * conv_out_height * conv_out_width;
    int K = in_channels * kernel_size * kernel_size;
    
    // For simplicity, we'll use PyTorch's convolution here and then apply our custom kernel
    // In a full implementation, we would implement the entire convolution in CUDA
    auto conv_result = torch::conv2d(input, weight, bias);
    
    // Launch kernel for pooling, sigmoid and sum
    dim3 grid(batch_size, out_channels);
    dim3 block(256);
    int shared_mem_size = block.x * sizeof(float);
    
    avg_pool2d_and_sigmoid_and_sum<<<grid, block, shared_mem_size>>>(
        conv_result.data_ptr<float>(),
        final_output.data_ptr<float>(),
        batch_size,
        out_channels,
        conv_out_height,
        conv_out_width,
        pool_size,
        pool_out_height,
        pool_out_width
    );
    
    cublasDestroy(cublas_handle);
    return final_output;
}
"""

fused_conv_pool_sigmoid_sum_cpp_source = """
torch::Tensor fused_conv_pool_sigmoid_sum_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int pool_size
);
"""

# Compile the inline CUDA code
fused_conv_pool_sigmoid_sum = load_inline(
    name="fused_conv_pool_sigmoid_sum",
    cpp_sources=fused_conv_pool_sigmoid_sum_cpp_source,
    cuda_sources=fused_conv_pool_sigmoid_sum_source,
    functions=["fused_conv_pool_sigmoid_sum_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.fused_op = fused_conv_pool_sigmoid_sum

    def forward(self, x):
        # Use custom CUDA kernel for fused operations
        return self.fused_op.fused_conv_pool_sigmoid_sum_cuda(
            x, 
            self.conv.weight, 
            self.conv.bias, 
            self.pool_kernel_size
        )