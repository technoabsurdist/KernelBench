import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: min(dim=1) -> tanh -> tanh
fused_min_tanh_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // for fminf, tanhf

// Kernel to perform min reduction along channels, followed by two tanh applications
__global__ void fused_min_tanh_tanh_kernel(const float* input, float* output, int N, int C, int H, int W) {
    // Calculate the global thread indices for width, height, and batch
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;

    // Check if the thread is within the bounds of the output tensor's spatial dimensions
    if (w < W && h < H) {
        // Calculate the base index for the current (n, h, w) position in the input tensor.
        // This points to the first channel (c=0) for this spatial location.
        // The input tensor is in NCHW format.
        long long input_base_idx = (long long)n * C * H * W + (long long)h * W + w;
        
        // Initialize min_val with the value from the first channel
        float min_val = input[input_base_idx];

        // Iterate over the remaining channels to find the minimum value.
        // The stride between channels for a fixed (h, w) is H * W.
        for (int c = 1; c < C; ++c) {
            long long current_idx = input_base_idx + (long long)c * H * W;
            min_val = fminf(min_val, input[current_idx]);
        }

        // Apply tanh twice to the minimum value
        float result = tanhf(tanhf(min_val));

        // Calculate the output index. The output has shape (N, 1, H, W).
        long long output_idx = (long long)n * H * W + (long long)h * W + w;
        
        // Write the final result to the output tensor
        output[output_idx] = result;
    }
}

// C++ wrapper function to launch the CUDA kernel
torch::Tensor fused_min_tanh_tanh_cuda(torch::Tensor input) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input tensor must be 4-dimensional (NCHW)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    
    // Ensure the input tensor is contiguous in memory for correct indexing
    input = input.contiguous();

    // Get dimensions from the input tensor
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    // Create the output tensor with shape (N, 1, H, W), matching the input device and dtype
    auto output = torch::empty({N, 1, H, W}, input.options());

    // Define block and grid dimensions for the kernel launch
    // Use 16x16 threads per block, a common choice for 2D problems
    const dim3 block_dim(16, 16, 1);
    // Calculate grid dimensions needed to cover the entire HxW plane for each item in the batch
    const dim3 grid_dim(
        (W + block_dim.x - 1) / block_dim.x,
        (H + block_dim.y - 1) / block_dim.y,
        N
    );

    // Launch the CUDA kernel
    fused_min_tanh_tanh_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W
    );
    
    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

# C++ source for the function signature
fused_min_tanh_tanh_cpp_source = "torch::Tensor fused_min_tanh_tanh_cuda(torch::Tensor input);"

# Compile the inline CUDA/C++ code
fused_op = load_inline(
    name="fused_min_tanh_tanh",
    cpp_sources=fused_min_tanh_tanh_cpp_source,
    cuda_sources=fused_min_tanh_tanh_source,
    functions=["fused_min_tanh_tanh_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, then a custom fused operation that combines
    minimum reduction along the channel dimension with two subsequent Tanh activations.
    This fusion reduces memory bandwidth and kernel launch overhead.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        # The convolution layer remains a standard PyTorch/cuDNN operator,
        # as it is highly optimized.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Store the compiled custom fused operator
        self.fused_op = fused_op.fused_min_tanh_tanh_cuda

    def forward(self, x):
        # 1. Apply the standard PyTorch convolution
        x = self.conv(x)
        # 2. Apply the custom fused kernel for min(dim=1) -> tanh -> tanh
        x = self.fused_op(x)
        return x