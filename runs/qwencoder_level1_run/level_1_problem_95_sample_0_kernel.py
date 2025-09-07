import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for online softmax and cross entropy loss
cross_entropy_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void cross_entropy_kernel(
    const float* predictions,
    const long* targets,
    float* losses,
    float* softmax_buffer,
    int batch_size,
    int num_classes
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* x = predictions + batch_idx * num_classes;
    float* softmax_x = softmax_buffer + batch_idx * num_classes;
    
    // Step 1: Find max for numerical stability
    float max_val = x[0];
    for (int i = 1; i < num_classes; ++i) {
        max_val = fmaxf(max_val, x[i]);
    }
    
    // Step 2: Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < num_classes; ++i) {
        softmax_x[i] = expf(x[i] - max_val);
        sum += softmax_x[i];
    }
    
    // Step 3: Normalize
    for (int i = 0; i < num_classes; ++i) {
        softmax_x[i] /= sum;
    }
    
    // Step 4: Compute cross entropy loss
    int target_class = targets[batch_idx];
    losses[batch_idx] = -logf(softmax_x[target_class] + 1e-8f);
}

torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.size(0);
    auto num_classes = predictions.size(1);
    
    // Allocate temporary buffer for softmax results
    auto softmax_buffer = torch::zeros_like(predictions);
    auto losses = torch::zeros({batch_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int threads_per_block = 1;
    const int num_blocks = batch_size;
    
    cross_entropy_kernel<<<num_blocks, threads_per_block>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<long>(),
        losses.data_ptr<float>(),
        softmax_buffer.data_ptr<float>(),
        batch_size,
        num_classes
    );
    
    // Return mean loss
    return losses.mean();
}
"""

cross_entropy_cpp_source = (
    "torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

# Compile the inline CUDA code for cross entropy
cross_entropy = load_inline(
    name="cross_entropy",
    cpp_sources=cross_entropy_cpp_source,
    cuda_sources=cross_entropy_source,
    functions=["cross_entropy_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks
    using a custom CUDA kernel for improved performance.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cross_entropy_fn = cross_entropy

    def forward(self, predictions, targets):
        return self.cross_entropy_fn.cross_entropy_cuda(predictions, targets)

batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda(), torch.randint(0, num_classes, (batch_size,)).cuda()]

def get_init_inputs():
    return []