import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for MSE loss
mse_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

__global__ void mse_loss_kernel(const float* predictions, const float* targets, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = predictions[idx] - targets[idx];
        output[idx] = diff * diff;
    }
}

__global__ void mse_loss_reduction_kernel(const float* input, float* output, int size) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

torch::Tensor mse_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    auto squared_diffs = torch::zeros_like(predictions);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    // Compute squared differences
    mse_loss_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(), 
        targets.data_ptr<float>(), 
        squared_diffs.data_ptr<float>(), 
        size
    );
    
    // Reduction to compute mean
    auto result = torch::zeros({}, predictions.options());
    mse_loss_reduction_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        squared_diffs.data_ptr<float>(),
        result.data_ptr<float>(),
        size
    );
    
    // Divide by size to get mean
    result /= static_cast<float>(size);
    
    return result;
}
"""

mse_loss_cpp_source = (
    "torch::Tensor mse_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

# Compile the inline CUDA code for MSE loss
mse_loss = load_inline(
    name="mse_loss",
    cpp_sources=mse_loss_cpp_source,
    cuda_sources=mse_loss_source,
    functions=["mse_loss_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks using custom CUDA kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.mse_loss = mse_loss

    def forward(self, predictions, targets):
        return self.mse_loss.mse_loss_cuda(predictions, targets)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    scale = torch.rand(())
    return [torch.rand(batch_size, *input_shape)*scale, torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return []