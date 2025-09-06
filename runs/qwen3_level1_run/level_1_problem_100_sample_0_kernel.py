import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for hinge loss
hinge_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void hinge_loss_kernel(
    const float* predictions,
    const float* targets,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = 1.0f - predictions[idx] * targets[idx];
        output[idx] = fmaxf(0.0f, val);
    }
}

__global__ void hinge_loss_backward_kernel(
    const float* predictions,
    const float* targets,
    const float* grad_output,
    float* grad_predictions,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float margin = 1.0f - predictions[idx] * targets[idx];
        if (margin > 0) {
            grad_predictions[idx] = -targets[idx] * grad_output[idx];
        } else {
            grad_predictions[idx] = 0;
        }
    }
}

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    auto output = torch::zeros_like(predictions);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hinge_loss_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    return output;
}

torch::Tensor hinge_loss_backward_cuda(
    torch::Tensor predictions,
    torch::Tensor targets,
    torch::Tensor grad_output
) {
    auto size = predictions.numel();
    auto grad_predictions = torch::zeros_like(predictions);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hinge_loss_backward_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_predictions.data_ptr<float>(),
        size
    );

    return grad_predictions;
}
"""

hinge_loss_cpp_source = """
torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);
torch::Tensor hinge_loss_backward_cuda(torch::Tensor predictions, torch::Tensor targets, torch::Tensor grad_output);
"""

# Compile the inline CUDA code for hinge loss
hinge_loss_module = load_inline(
    name="hinge_loss",
    cpp_sources=hinge_loss_cpp_source,
    cuda_sources=hinge_loss_source,
    functions=["hinge_loss_cuda", "hinge_loss_backward_cuda"],
    verbose=False,
)

class HingeLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predictions, targets):
        ctx.save_for_backward(predictions, targets)
        loss_values = hinge_loss_module.hinge_loss_cuda(predictions, targets)
        return torch.mean(loss_values)
    
    @staticmethod
    def backward(ctx, grad_output):
        predictions, targets = ctx.saved_tensors
        grad_predictions = hinge_loss_module.hinge_loss_backward_cuda(predictions, targets, grad_output.expand_as(predictions) / predictions.numel())
        return grad_predictions, None

class ModelNew(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks with custom CUDA implementation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hinge_loss = HingeLossFunction.apply

    def forward(self, predictions, targets):
        return self.hinge_loss(predictions, targets)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda(), torch.randint(0, 2, (batch_size,)).float().cuda() * 2 - 1]

def get_init_inputs():
    return []