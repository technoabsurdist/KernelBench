import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused LogSoftmax and NLLLoss (CrossEntropy)
cross_entropy_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel to compute cross entropy loss.
// Each block computes the loss for one sample in the batch.
// The kernel calculates the per-sample loss, and the final reduction (mean) is done in the C++ wrapper.
__global__ void cross_entropy_forward_kernel(
    const float* predictions,
    const long* targets,
    float* losses,
    int batch_size,
    int num_classes) {

    // Use dynamic shared memory for reductions
    extern __shared__ float sdata[];

    // Each block processes one sample from the batch
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) {
        return;
    }

    const float* pred_row = predictions + batch_idx * num_classes;
    long target_idx = targets[batch_idx];

    // Step 1: Find the maximum value in the prediction row for numerical stability.
    // This is a parallel reduction within the block.
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        max_val = max(max_val, pred_row[i]);
    }

    // Block-wide reduction for max_val
    sdata[threadIdx.x] = max_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = max(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }
    // The max value for the row is now in sdata[0]
    max_val = sdata[0];
    __syncthreads();

    // Step 2: Calculate the sum of exponentials (denominator of softmax).
    // This is also a parallel reduction.
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        sum_exp += expf(pred_row[i] - max_val);
    }

    // Block-wide reduction for sum_exp
    sdata[threadIdx.x] = sum_exp;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    // The sum of exponentials is now in sdata[0]
    sum_exp = sdata[0];

    // Step 3: Calculate the final loss for the sample.
    // log(softmax(x)_i) = log(exp(x_i) / sum(exp(x_j)))
    //                   = x_i - log(sum(exp(x_j)))
    // With the max trick for stability:
    //                   = (x_i - max_val) - log(sum(exp(x_j - max_val)))
    // NLLLoss is the negative of this value.
    if (threadIdx.x == 0) {
        float log_sum_exp = logf(sum_exp);
        float pred_target_val = pred_row[target_idx];
        losses[batch_idx] = -( (pred_target_val - max_val) - log_sum_exp );
    }
}

// C++ wrapper function to launch the CUDA kernel and handle tensors.
torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets) {
    // Input validation
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.is_contiguous(), "predictions must be contiguous");
    TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be a float32 tensor");
    TORCH_CHECK(targets.scalar_type() == torch::kInt64, "targets must be a int64 tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.size(0) == targets.size(0), "Input batch sizes must match");

    const auto batch_size = predictions.size(0);
    const auto num_classes = predictions.size(1);

    // Create an output tensor to store the loss for each sample
    auto losses = torch::empty({batch_size}, predictions.options());

    // Kernel launch configuration
    // Use a large block size for efficient reduction
    const int block_size = 1024;
    const int num_blocks = batch_size;
    // Allocate shared memory dynamically for the reduction
    const int shared_mem_size = block_size * sizeof(float);

    cross_entropy_forward_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<long>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );
    
    // Check for any CUDA errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Return the mean of the per-sample losses
    return losses.mean();
}
"""

cross_entropy_cpp_source = (
    "torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

# Compile the inline CUDA code
custom_cross_entropy = load_inline(
    name="custom_cross_entropy",
    cpp_sources=cross_entropy_cpp_source,
    cuda_sources=cross_entropy_source,
    functions=["cross_entropy_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks
    using a custom fused CUDA kernel for improved performance.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.custom_ce_loss = custom_cross_entropy.cross_entropy_cuda

    def forward(self, predictions, targets):
        return self.custom_ce_loss(predictions, targets)