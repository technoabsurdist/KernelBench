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
    float* loss,
    float* probabilities,
    int batch_size,
    int num_classes
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int idx = batch_idx * num_classes;
    
    // Find maximum for numerical stability
    float max_val = predictions[idx];
    for (int i = 1; i < num_classes; i++) {
        float val = predictions[idx + i];
        max_val = fmaxf(max_val, val);
    }
    
    // Compute softmax and sum
    float sum_exp = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        float exp_val = expf(predictions[idx + i] - max_val);
        sum_exp += exp_val;
        if (probabilities != nullptr) {
            probabilities[idx + i] = exp_val;
        }
    }
    
    // Normalize probabilities
    if (probabilities != nullptr) {
        for (int i = 0; i < num_classes; i++) {
            probabilities[idx + i] /= sum_exp;
        }
    }
    
    // Compute cross entropy loss for this batch element
    int target_class = targets[batch_idx];
    float prob_target = expf(predictions[idx + target_class] - max_val) / sum_exp;
    loss[batch_idx] = -logf(prob_target);
}

torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.size(0);
    auto num_classes = predictions.size(1);
    
    auto loss = torch::zeros({batch_size}, torch::TensorOptions().device(predictions.device()).dtype(torch::kFloat32));
    
    const int threads_per_block = 1;
    const int num_blocks = batch_size;
    
    cross_entropy_kernel<<<num_blocks, threads_per_block>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<long>(),
        loss.data_ptr<float>(),
        nullptr, // Don't store probabilities
        batch_size,
        num_classes
    );
    
    // Return mean loss
    return loss.mean();
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

batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)
dim = 1

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

def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda(), torch.randint(0, num_classes, (batch_size,)).cuda()]

def get_init_inputs():
    return []