import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused CUDA kernel for the sequence: clamp -> multiply -> max_reduction
fused_clamp_mul_max_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat> // For FLT_MAX

// Device function for clamping a float value
__device__ inline float clamp(float val, float min_val, float max_val) {
    return fmaxf(min_val, fminf(val, max_val));
}

// CUDA kernel to perform fused clamp, multiply, and max reduction
__global__ void clamp_mul_max_kernel(
    const float* input,      // Input tensor of shape (N, C, D, H, W)
    const float* multiplier, // Multiplier tensor of shape (C, 1, 1, 1)
    float* output,           // Output tensor of shape (N, D, H, W)
    const float clamp_min,
    const float clamp_max,
    const int C,
    const long long spatial_dims, // D * H * W
    const long long num_output_elements // N * D * H * W
) {
    // Calculate strides for efficient indexing
    const long long C_stride = spatial_dims;
    const long long N_stride = (long long)C * spatial_dims;

    // Use a grid-stride loop to ensure all output elements are processed
    for (long long out_idx = blockIdx.x * blockDim.x + threadIdx.x; 
         out_idx < num_output_elements; 
         out_idx += gridDim.x * blockDim.x) {
        
        // De-flatten the output index to get batch and spatial indices
        const long long spatial_idx = out_idx % spatial_dims;
        const long long n = out_idx / spatial_dims;

        float max_val = -FLT_MAX;

        // Each thread performs a sequential reduction over the C dimension for its assigned output element
        for (int c = 0; c < C; ++c) {
            // Calculate the linear index for the input tensor
            const long long in_idx = n * N_stride + c * C_stride + spatial_idx;
            
            // Load value from global memory
            float val = input[in_idx];
            
            // Perform clamp and multiply operations in registers
            val = clamp(val, clamp_min, clamp_max);
            val *= multiplier[c]; // Multiplier is broadcasted, so we only need the channel index
            
            // Update the maximum value
            max_val = fmaxf(max_val, val);
        }

        // Write the final result to global memory
        output[out_idx] = max_val;
    }
}

// C++ wrapper function that will be called from Python
torch::Tensor clamp_mul_max_cuda(
    torch::Tensor input, 
    torch::Tensor multiplier, 
    float clamp_min, 
    float clamp_max
) {
    // Input validation checks
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(multiplier.is_cuda(), "Multiplier tensor must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input tensor must be of type float32");
    TORCH_CHECK(multiplier.scalar_type() == torch::kFloat32, "Multiplier tensor must be of type float32");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(multiplier.is_contiguous(), "Multiplier tensor must be contiguous");
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5D (N, C, D, H, W)");
    
    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto D = input.size(2);
    const auto H = input.size(3);
    const auto W = input.size(4);

    TORCH_CHECK(multiplier.size(0) == C, "Multiplier must have size C in the first dimension");

    // Create the output tensor with the reduced shape
    auto output = torch::empty({N, D, H, W}, input.options());

    const long long spatial_dims = (long long)D * H * W;
    const long long num_output_elements = (long long)N * spatial_dims;
    
    if (num_output_elements == 0) {
        return output;
    }

    // Configure kernel launch parameters
    const int block_size = 256;
    const int num_blocks = (num_output_elements + block_size - 1) / block_size;

    // Launch the CUDA kernel
    clamp_mul_max_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        output.data_ptr<float>(),
        clamp_min,
        clamp_max,
        C,
        spatial_dims,
        num_output_elements
    );
    
    // Check for any CUDA errors during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

# C++ source for the function binding
fused_clamp_mul_max_cpp_source = """
torch::Tensor clamp_mul_max_cuda(
    torch::Tensor input, 
    torch::Tensor multiplier, 
    float clamp_min, 
    float clamp_max
);
"""

# Use JIT compilation to build the custom CUDA operator
fused_op = load_inline(
    name="fused_clamp_mul_max",
    cpp_sources=fused_clamp_mul_max_cpp_source,
    cuda_sources=fused_clamp_mul_max_source,
    functions=["clamp_mul_max_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    An optimized version of the model that fuses the last three operations
    (clamp, multiply, max_reduction) into a single custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        # Standard PyTorch operators that are highly optimized
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        
        # Learnable parameter
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        
        # Store constants
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        # 1. Convolution (using PyTorch's cuDNN-backed implementation)
        x = self.conv(x)
        
        # 2. First multiplication
        x = x * self.multiplier
        
        # 3. Instance Normalization (using PyTorch's implementation)
        x = self.instance_norm(x)
        
        # 4. Fused operation: clamp -> multiply -> max_reduction
        # This replaces the following three PyTorch operations:
        # x = torch.clamp(x, self.clamp_min, self.clamp_max)
        # x = x * self.multiplier
        # x = torch.max(x, dim=1)[0]
        x = fused_op.clamp_mul_max_cuda(x, self.multiplier, self.clamp_min, self.clamp_max)
        
        return x