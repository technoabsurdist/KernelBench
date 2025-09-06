import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Triplet Margin Loss
triplet_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel using shared memory for parallel reduction to compute Triplet Margin Loss
__global__ void triplet_margin_loss_kernel(const float* anchor,
                                           const float* positive,
                                           const float* negative,
                                           float* loss_output,
                                           const float margin,
                                           const int batch_size,
                                           const int feature_dim) {
    // Each block computes the loss for one sample in the batch
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) {
        return;
    }

    // Dynamically allocated shared memory for reduction
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // --- Calculate squared L2 distance for (anchor, positive) ---
    float sum_pos = 0.0f;
    // Each thread computes a partial sum over the feature dimension
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        float diff = anchor[batch_idx * feature_dim + i] - positive[batch_idx * feature_dim + i];
        sum_pos += diff * diff;
    }
    sdata[tid] = sum_pos;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // After reduction, thread 0 has the final squared sum for the positive pair
    float dist_pos_sq = sdata[0];
    __syncthreads(); // Sync before reusing shared memory for the negative pair

    // --- Calculate squared L2 distance for (anchor, negative) ---
    float sum_neg = 0.0f;
    // Each thread computes a partial sum
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        float diff = anchor[batch_idx * feature_dim + i] - negative[batch_idx * feature_dim + i];
        sum_neg += diff * diff;
    }
    sdata[tid] = sum_neg;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // After reduction, thread 0 has the final squared sum for the negative pair
    float dist_neg_sq = sdata[0];

    // --- Final loss calculation by thread 0 ---
    if (tid == 0) {
        float dist_pos = sqrtf(dist_pos_sq);
        float dist_neg = sqrtf(dist_neg_sq);
        float loss = dist_pos - dist_neg + margin;
        loss_output[batch_idx] = fmaxf(loss, 0.0f);
    }
}

// C++ wrapper function to launch the CUDA kernel
torch::Tensor triplet_margin_loss_cuda(torch::Tensor anchor,
                                       torch::Tensor positive,
                                       torch::Tensor negative,
                                       float margin) {
    // Input validation
    TORCH_CHECK(anchor.is_cuda(), "Input tensors must be on a CUDA device");
    TORCH_CHECK(anchor.is_contiguous(), "Input anchor tensor must be contiguous");
    TORCH_CHECK(positive.is_contiguous(), "Input positive tensor must be contiguous");
    TORCH_CHECK(negative.is_contiguous(), "Input negative tensor must be contiguous");
    TORCH_CHECK(anchor.dtype() == torch::kFloat32, "Input tensors must be of type float32");
    TORCH_CHECK(anchor.dim() == 2, "Input tensors should be 2D");

    const auto batch_size = anchor.size(0);
    const auto feature_dim = anchor.size(1);

    // Output tensor for per-sample losses
    auto per_sample_loss = torch::empty({batch_size}, anchor.options());

    // Kernel launch configuration
    // Use a power of 2 for block_size for efficient reduction
    const int block_size = 1024;
    const int grid_size = batch_size;
    
    // Shared memory size needed for the reduction
    const size_t shared_mem_size = block_size * sizeof(float);

    triplet_margin_loss_kernel<<<grid_size, block_size, shared_mem_size>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        per_sample_loss.data_ptr<float>(),
        margin,
        batch_size,
        feature_dim
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    // The original loss uses mean reduction by default
    return per_sample_loss.mean();
}
"""

triplet_loss_cpp_source = """
torch::Tensor triplet_margin_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, float margin);
"""

# Compile the inline CUDA code
triplet_loss_op = load_inline(
    name="triplet_loss_op",
    cpp_sources=triplet_loss_cpp_source,
    cuda_sources=triplet_loss_source,
    functions=["triplet_margin_loss_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    An optimized model that computes Triplet Margin Loss using a custom fused CUDA kernel.

    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin
        self.fused_triplet_loss = triplet_loss_op.triplet_margin_loss_cuda

    def forward(self, anchor, positive, negative):
        return self.fused_triplet_loss(anchor, positive, negative, self.margin)

batch_size = 32768
input_shape = (8192,)
dim = 1

def get_inputs():
    # Ensure inputs are contiguous and on CUDA device for the custom kernel
    scale = torch.rand(())
    return [
        (torch.rand(batch_size, *input_shape) * scale).cuda().contiguous(),
        torch.rand(batch_size, *input_shape).cuda().contiguous(),
        torch.rand(batch_size, *input_shape).cuda().contiguous()
    ]
    
def get_init_inputs():
    return [1.0]  # Default margin