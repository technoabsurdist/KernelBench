import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for MSE loss
mse_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mse_loss_kernel(const float* predictions, const float* targets, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = predictions[idx] - targets[idx];
        output[idx] = diff * diff;
    }
}

__global__ void mse_loss_backward_kernel(const float* predictions, const float* targets, float* grad_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_output[idx] = 2.0f * (predictions[idx] - targets[idx]) / (float)size;
    }
}

torch::Tensor mse_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    auto squared_diffs = torch::zeros_like(predictions);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    mse_loss_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(), 
        targets.data_ptr<float>(), 
        squared_diffs.data_ptr<float>(), 
        size
    );
    
    return squared_diffs.mean();
}

torch::Tensor mse_loss_backward_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    auto grad_output = torch::zeros_like(predictions);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    mse_loss_backward_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(), 
        targets.data_ptr<float>(), 
        grad_output.data_ptr<float>(), 
        size
    );
    
    return grad_output;
}
"""

mse_loss_cpp_source = """
torch::Tensor mse_loss_cuda(torch::Tensor predictions, torch::Tensor targets);
torch::Tensor mse_loss_backward_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

# Compile the inline CUDA code for MSE loss
mse_loss = load_inline(
    name="mse_loss",
    cpp_sources=mse_loss_cpp_source,
    cuda_sources=mse_loss_source,
    functions=["mse_loss_cuda", "mse_loss_backward_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks using custom CUDA kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.mse_loss_cuda = mse_loss

    def forward(self, predictions, targets):
        return self.mse_loss_cuda.mse_loss_cuda(predictions, targets)