import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused Hinge Loss operation
# This kernel fuses the element-wise multiplication, subtraction, clamp, and mean operations
# into a single multi-pass reduction kernel to avoid allocating large intermediate tensors.
fused_hinge_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm> // For std::min

// Kernel to reduce an array 'in' of size 'n' and write partial sums to 'out'.
// Each block writes one value to 'out'.
__global__ void reduce_sum_kernel(const float* in, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads one element from global to shared mem
    float my_val = (i < n) ? in[i] : 0.0f;
    sdata[tid] = my_val;
    __syncthreads();

    // Do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

// Fused Hinge Loss + Reduction (Pass 1)
// Computes sum(max(0, 1 - pred * target)) for chunks of the input
// and stores partial sums in 'partial_sums_out'.
__global__ void fused_hinge_loss_reduce_part1(
    const float* predictions,
    const float* targets,
    float* partial_sums_out,
    long long total_size,
    int M // columns of predictions
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float thread_sum = 0.0f;
    // Grid-stride loop
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size; i += blockDim.x * gridDim.x) {
        int col = i % M;
        float pred_val = predictions[i];
        float target_val = targets[col];
        float hinge_val = 1.0f - pred_val * target_val;
        if (hinge_val > 0.0f) {
            thread_sum += hinge_val;
        }
    }
    sdata[tid] = thread_sum;
    __syncthreads();

    // In-block reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // First thread writes the block's partial sum to global memory
    if (tid == 0) {
        partial_sums_out[blockIdx.x] = sdata[0];
    }
}

torch::Tensor fused_hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.is_contiguous(), "predictions must be contiguous");
    TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be a float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kFloat32, "targets must be a float32 tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 1, "targets must be 1D");
    TORCH_CHECK(predictions.size(1) == targets.size(0), "predictions.size(1) must equal targets.size(0)");

    const long long total_size = predictions.numel();
    if (total_size == 0) {
        return torch::tensor(0.0, predictions.options());
    }
    const int M = predictions.size(1);

    // --- Pass 1: Main computation and first reduction ---
    const int block_size = 256;
    // Cap the number of blocks to avoid creating a huge intermediate tensor and to stay within CUDA grid limits.
    const int num_blocks_pass1 = std::min((long long)65535, (total_size + block_size - 1) / block_size);
    auto partial_sums = torch::zeros({num_blocks_pass1}, predictions.options());

    fused_hinge_loss_reduce_part1<<<num_blocks_pass1, block_size, block_size * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        total_size,
        M
    );

    // --- Subsequent reduction passes ---
    torch::Tensor input_to_reduce = partial_sums;
    long long n = num_blocks_pass1;

    while (n > 1) {
        int num_blocks_reduce = (n + block_size - 1) / block_size;
        auto next_partial_sums = torch::zeros({num_blocks_reduce}, predictions.options());

        reduce_sum_kernel<<<num_blocks_reduce, block_size, block_size * sizeof(float)>>>(
            input_to_reduce.data_ptr<float>(),
            next_partial_sums.data_ptr<float>(),
            n
        );
        input_to_reduce = next_partial_sums;
        n = num_blocks_reduce;
    }

    // At this point, input_to_reduce is a tensor with a single element (the total sum).
    // Now, calculate the mean.
    return input_to_reduce / total_size;
}
"""

fused_hinge_loss_cpp_source = (
    "torch::Tensor fused_hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

# Compile the inline CUDA code
fused_hinge_loss = load_inline(
    name="fused_hinge_loss",
    cpp_sources=fused_hinge_loss_cpp_source,
    cuda_sources=fused_hinge_loss_source,
    functions=["fused_hinge_loss_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    An optimized model that computes Hinge Loss for binary classification tasks
    using a custom fused CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.fused_op = fused_hinge_loss.fused_hinge_loss_cuda

    def forward(self, predictions, targets):
        return self.fused_op(predictions, targets)