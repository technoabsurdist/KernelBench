import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for RMS Normalization
rms_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void rms_norm_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const int num_features,
    const int feature_size,
    const float eps
) {
    const int batch_idx = blockIdx.x;
    const int feature_idx = blockIdx.y;
    const int thread_idx = threadIdx.x;
    
    const int block_size = blockDim.x;
    const int block_offset = batch_idx * num_features * feature_size + feature_idx * feature_size;
    
    // Shared memory for reduction
    extern __shared__ float shared_mem[];
    float* shared_data = shared_mem;
    
    // Compute sum of squares
    float local_sum = 0.0f;
    for (int i = thread_idx; i < feature_size; i += block_size) {
        float val = x[block_offset + i];
        local_sum += val * val;
    }
    
    shared_data[thread_idx] = local_sum;
    __syncthreads();
    
    // Reduce within block
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (thread_idx < stride) {
            shared_data[thread_idx] += shared_data[thread_idx + stride];
        }
        __syncthreads();
    }
    
    // Compute RMS
    float rms = sqrtf(shared_data[0] / feature_size + eps);
    
    // Normalize
    for (int i = thread_idx; i < feature_size; i += block_size) {
        out[block_offset + i] = x[block_offset + i] / rms;
    }
}

torch::Tensor rms_norm_cuda(torch::Tensor x, float eps) {
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    
    auto sizes = x.sizes();
    auto batch_size = sizes[0];
    auto num_features = sizes[1];
    auto feature_size = 1;
    for (int i = 2; i < sizes.size(); i++) {
        feature_size *= sizes[i];
    }
    
    auto out = torch::empty_like(x);
    
    const int threads_per_block = 256;
    const dim3 blocks(batch_size, num_features);
    const int shared_mem_size = threads_per_block * sizeof(float);
    
    rms_norm_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        num_features,
        feature_size,
        eps
    );
    
    return out;
}
"""

rms_norm_cpp_source = """
torch::Tensor rms_norm_cuda(torch::Tensor x, float eps);
"""

# Compile the inline CUDA code for RMS normalization
rms_norm = load_inline(
    name="rms_norm",
    cpp_sources=rms_norm_cpp_source,
    cuda_sources=rms_norm_source,
    functions=["rms_norm_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs RMS Normalization with custom CUDA kernel.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.rms_norm = rms_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        return self.rms_norm.rms_norm_cuda(x, self.eps)

batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [features]