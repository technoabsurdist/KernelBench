import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Triplet Margin Loss
triplet_margin_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void triplet_margin_loss_kernel(
    const float* anchor,
    const float* positive,
    const float* negative,
    float* output,
    const int batch_size,
    const int dim,
    const float margin
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float dist_pos = 0.0f;
        float dist_neg = 0.0f;
        
        // Compute squared Euclidean distance between anchor and positive
        for (int i = 0; i < dim; ++i) {
            float diff = anchor[idx * dim + i] - positive[idx * dim + i];
            dist_pos += diff * diff;
        }
        
        // Compute squared Euclidean distance between anchor and negative
        for (int i = 0; i < dim; ++i) {
            float diff = anchor[idx * dim + i] - negative[idx * dim + i];
            dist_neg += diff * diff;
        }
        
        // Compute loss: max(dist_pos - dist_neg + margin, 0)
        float loss = fmaxf(dist_pos - dist_neg + margin, 0.0f);
        output[idx] = loss;
    }
}

torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin
) {
    const int batch_size = anchor.size(0);
    const int dim = anchor.size(1);
    auto output = torch::zeros({batch_size}, anchor.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;
    
    triplet_margin_loss_kernel<<<num_blocks, block_size>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim,
        margin
    );
    
    // Return mean of losses
    return output.mean();
}
"""

triplet_margin_loss_cpp_source = """
torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin
);
"""

# Compile the inline CUDA code for Triplet Margin Loss
triplet_margin_loss = load_inline(
    name="triplet_margin_loss",
    cpp_sources=triplet_margin_loss_cpp_source,
    cuda_sources=triplet_margin_loss_source,
    functions=["triplet_margin_loss_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks using custom CUDA kernel.

    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin
        self.triplet_loss = triplet_margin_loss

    def forward(self, anchor, positive, negative):
        return self.triplet_loss.triplet_margin_loss_cuda(anchor, positive, negative, self.margin)

batch_size = 32768
input_shape = (8192,)
dim = 1

def get_inputs():
    scale = torch.rand(())
    return [torch.rand(batch_size, *input_shape)*scale, torch.rand(batch_size, *input_shape), torch.rand(batch_size, *input_shape)]
    
def get_init_inputs():
    return [1.0]  # Default margin