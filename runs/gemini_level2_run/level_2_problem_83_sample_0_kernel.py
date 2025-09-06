import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for fusing min, clamp, and dropout
fused_min_clamp_dropout_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <limits>

// CUDA kernel to fuse min, clamp, and dropout operations
__global__ void fused_min_clamp_dropout_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int size,
    const float min_value,
    const float max_value,
    const float p,
    const float scale,
    const bool is_training,
    const uint64_t seed,
    const uint64_t offset) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        curandState_t state;
        // Seed the random number generator.
        // Using idx + offset ensures different random numbers for each element and each call.
        curand_init(seed, idx, offset, &state);

        float val = input[idx];

        // 1. torch.min(x, min_value)
        val = fminf(val, min_value);

        // 2. torch.clamp(x, min=min_value, max=max_value)
        val = fmaxf(min_value, fminf(val, max_value));

        // 3. Dropout
        if (is_training) {
            float rand_val = curand_uniform(&state);
            if (rand_val < p) {
                output[idx] = 0.0f;
            } else {
                output[idx] = val * scale;
            }
        } else {
            output[idx] = val;
        }
    }
}

// C++ wrapper function to be called from Python
torch::Tensor fused_min_clamp_dropout_cuda(
    torch::Tensor input,
    const float min_value,
    const float max_value,
    const float p,
    const bool is_training) {

    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input tensor must be of type float32");

    auto out = torch::empty_like(input);
    const int size = input.numel();
    if (size == 0) {
        return out;
    }

    // Dropout is a no-op if p=0 or not in training mode
    if (p == 0.0f || !is_training) {
        // We can still run the min+clamp part of the kernel
        // Or, for simplicity, we can just call the PyTorch ops if dropout is off.
        // Let's run the kernel anyway for consistency.
    }

    const float scale = (p < 1.0f) ? (1.0f / (1.0f - p)) : 0.0f;
    
    // Use torch's default generator for a high-quality seed and offset
    auto gen = at::cuda::get_generator_or_default<at::CUDAGeneratorImpl>(
        c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
    std::pair<uint64_t, uint64_t> seed_offset = at::cuda::philox::unpack(gen->philox_engine_inputs(10));

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_min_clamp_dropout_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        size,
        min_value,
        max_value,
        p,
        scale,
        is_training,
        seed_offset.first,
        seed_offset.second
    );
    
    AT_CUDA_CHECK(cudaGetLastError());

    return out;
}
"""

fused_min_clamp_dropout_cpp_source = (
    "torch::Tensor fused_min_clamp_dropout_cuda(torch::Tensor input, const float min_value, const float max_value, const float p, const bool is_training);"
)

# JIT compile the custom CUDA kernel
fused_op = load_inline(
    name="fused_min_clamp_dropout",
    cpp_sources=fused_min_clamp_dropout_cpp_source,
    cuda_sources=fused_min_clamp_dropout_source,
    functions=["fused_min_clamp_dropout_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, 
    and then a fused (min, clamp, dropout) operation using a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        
        # Parameters for the fused operation are stored
        self.min_value = min_value
        self.max_value = max_value
        self.dropout_p = dropout_p

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        
        # The original sequence of min, clamp, and dropout is replaced by a single call
        # to our custom fused CUDA kernel.
        # The `self.training` attribute is passed to control dropout behavior.
        x = fused_op.fused_min_clamp_dropout_cuda(
            x, self.min_value, self.max_value, self.dropout_p, self.training
        )
        return x