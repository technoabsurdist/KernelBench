import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Smooth L1 Loss
smooth_l1_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void smooth_l1_loss_kernel(
    const float* predictions,
    const float* targets,
    float* output,
    const float beta,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = fabsf(predictions[idx] - targets[idx]);
        if (diff < beta) {
            output[idx] = 0.5f * diff * diff / beta;
        } else {
            output[idx] = diff - 0.5f * beta;
        }
    }
}

torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets, float beta=1.0) {
    auto size = predictions.numel();
    auto output = torch::zeros_like(predictions);
    
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    
    smooth_l1_loss_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        beta,
        size
    );
    
    return output;
}
"""

smooth_l1_loss_cpp_source = """
torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets, float beta);
"""

# Compile the inline CUDA code for Smooth L1 Loss
smooth_l1_loss = load_inline(
    name="smooth_l1_loss",
    cpp_sources=smooth_l1_loss_cpp_source,
    cuda_sources=smooth_l1_loss_source,
    functions=["smooth_l1_loss_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.smooth_l1_loss_func = smooth_l1_loss

    def forward(self, predictions, targets):
        # Compute element-wise loss using custom CUDA kernel
        loss_elements = self.smooth_l1_loss_func.smooth_l1_loss_cuda(predictions, targets)
        # Return mean of all elements (same as PyTorch's reduction='mean')
        return loss_elements.mean()

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    scale = torch.rand(()).cuda()
    return [torch.rand(batch_size, *input_shape).cuda()*scale, torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return []