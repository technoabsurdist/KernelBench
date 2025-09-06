import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Smooth L1 Loss
smooth_l1_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

__global__ void smooth_l1_loss_kernel(
    const float* predictions,
    const float* targets,
    float* output,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = predictions[idx] - targets[idx];
        float abs_diff = fabsf(diff);
        if (abs_diff < 1.0f) {
            output[idx] = 0.5f * diff * diff;
        } else {
            output[idx] = abs_diff - 0.5f;
        }
    }
}

__global__ void smooth_l1_loss_backward_kernel(
    const float* predictions,
    const float* targets,
    float* grad_output,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = predictions[idx] - targets[idx];
        float abs_diff = fabsf(diff);
        if (abs_diff < 1.0f) {
            grad_output[idx] = diff;
        } else {
            grad_output[idx] = (diff > 0) ? 1.0f : -1.0f;
        }
    }
}

torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    auto output = torch::zeros_like(predictions);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    smooth_l1_loss_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output.mean();
}

torch::Tensor smooth_l1_loss_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor predictions,
    torch::Tensor targets
) {
    auto size = predictions.numel();
    auto grad_input = torch::zeros_like(predictions);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    smooth_l1_loss_backward_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        size
    );
    
    // Apply chain rule with upstream gradient
    grad_input *= grad_output / size;
    
    return grad_input;
}
"""

smooth_l1_loss_cpp_source = """
torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets);
torch::Tensor smooth_l1_loss_backward_cuda(torch::Tensor grad_output, torch::Tensor predictions, torch::Tensor targets);
"""

# Compile the inline CUDA code for Smooth L1 Loss
smooth_l1_loss_module = load_inline(
    name="smooth_l1_loss",
    cpp_sources=smooth_l1_loss_cpp_source,
    cuda_sources=smooth_l1_loss_source,
    functions=["smooth_l1_loss_cuda", "smooth_l1_loss_backward_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class SmoothL1LossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predictions, targets):
        ctx.save_for_backward(predictions, targets)
        return smooth_l1_loss_module.smooth_l1_loss_cuda(predictions, targets)
    
    @staticmethod
    def backward(ctx, grad_output):
        predictions, targets = ctx.saved_tensors
        grad_predictions = smooth_l1_loss_module.smooth_l1_loss_backward_cuda(grad_output, predictions, targets)
        return grad_predictions, None

class ModelNew(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks using custom CUDA kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.smooth_l1_loss_fn = SmoothL1LossFunction.apply

    def forward(self, predictions, targets):
        return self.smooth_l1_loss_fn(predictions, targets)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    scale = torch.rand(())
    return [torch.rand(batch_size, *input_shape)*scale, torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return []