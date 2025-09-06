import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for KL divergence
kl_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void kl_div_kernel(
    const float* log_predictions,
    const float* targets,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (targets[idx] > 1e-8f) {
            output[idx] = targets[idx] * (logf(targets[idx]) - log_predictions[idx]);
        } else {
            output[idx] = 0.0f;
        }
    }
}

torch::Tensor kl_div_cuda(torch::Tensor log_predictions, torch::Tensor targets) {
    auto size = log_predictions.numel();
    auto output = torch::zeros_like(log_predictions);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    kl_div_kernel<<<num_blocks, block_size>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    // Sum over all elements and divide by batch size for batchmean reduction
    float sum = output.sum().item<float>();
    int batch_size = log_predictions.size(0);
    return torch::tensor(sum / batch_size, torch::dtype(log_predictions.dtype()).device(log_predictions.device()));
}
"""

kl_div_cpp_source = (
    "torch::Tensor kl_div_cuda(torch::Tensor log_predictions, torch::Tensor targets);"
)

# Compile the inline CUDA code for KL divergence
kl_div = load_inline(
    name="kl_div",
    cpp_sources=kl_div_cpp_source,
    cuda_sources=kl_div_source,
    functions=["kl_div_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions
    using custom CUDA kernels for improved performance.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.kl_div_func = kl_div

    def forward(self, predictions, targets):
        log_predictions = torch.log(predictions)
        return self.kl_div_func.kl_div_cuda(log_predictions, targets)

batch_size = 8192 * 2
input_shape = (8192 * 2,)
dim = 1

def get_inputs():
    scale = torch.rand(())
    return [(torch.rand(batch_size, *input_shape)*scale).softmax(dim=-1), torch.rand(batch_size, *input_shape).softmax(dim=-1)]

def get_init_inputs():
    return []