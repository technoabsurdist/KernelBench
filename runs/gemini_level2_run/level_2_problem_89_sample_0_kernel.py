import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation:
# Softmax (dim=1) -> Subtract -> Swish -> Max (dim=1)
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat> // For FLT_MAX

// Device function for swish activation: x * sigmoid(x)
__device__ inline float swish(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void fused_softmax_sub_swish_max_kernel(
    const float* input,      // Input tensor data (N, C, D, H, W)
    const float* subtract,   // Subtract tensor data (C)
    float* output,           // Output tensor data (N, D, H, W)
    int N, int C, int D, int H, int W) {

    // Calculate the global thread index, where each thread handles one spatial location (n,d,h,w)
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial_dims = N * D * H * W;

    // Boundary check to prevent out-of-bounds access
    if (spatial_idx >= total_spatial_dims) {
        return;
    }

    // Decompose the 1D spatial_idx into 4D (n, d, h, w) coordinates
    int w = spatial_idx % W;
    int h = (spatial_idx / W) % H;
    int d = (spatial_idx / (W * H)) % D;
    int n = spatial_idx / (W * H * D);

    // Calculate strides for the NCDHW memory layout
    const int C_stride = D * H * W;
    const int N_stride = C * C_stride;

    // Get a pointer to the first channel's data for the current spatial location
    const float* current_pixel_channels = input + n * N_stride + d * H * W + h * W + w;

    // --- Fused Operation Start ---

    // 1. Softmax (stable version) across the channel dimension (C)
    // 1a. Find the maximum value among channels for numerical stability
    float max_val = -FLT_MAX;
    for (int c = 0; c < C; ++c) {
        max_val = fmaxf(max_val, current_pixel_channels[c * C_stride]);
    }

    // 1b. Compute the sum of exponentials of (value - max_val)
    float sum_exp = 0.0f;
    for (int c = 0; c < C; ++c) {
        sum_exp += expf(current_pixel_channels[c * C_stride] - max_val);
    }
    
    // Use the inverse of the sum to replace division with multiplication inside the main loop
    const float inv_sum_exp = 1.0f / sum_exp;

    // 2, 3, 4. Fused loop for Softmax -> Subtract -> Swish -> Max Reduction
    float final_max_val = -FLT_MAX;
    for (int c = 0; c < C; ++c) {
        // Complete the softmax calculation for the current channel
        float sm_val = expf(current_pixel_channels[c * C_stride] - max_val) * inv_sum_exp;
        
        // Perform the channel-wise subtraction
        float sub_val = sm_val - subtract[c];
        
        // Apply the Swish activation function
        float swish_val = swish(sub_val);
        
        // Update the running maximum value
        final_max_val = fmaxf(final_max_val, swish_val);
    }

    // 5. Write the final result to the output tensor
    int output_idx = n * (D * H * W) + d * (H * W) + h * W + w;
    output[output_idx] = final_max_val;
}

// C++ wrapper function that PyTorch will call
torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor subtract) {
    // Input validation checks
    TORCH_CHECK(x.is_cuda(), "Input tensor 'x' must be a CUDA tensor");
    TORCH_CHECK(subtract.is_cuda(), "Input tensor 'subtract' must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 5, "Input tensor 'x' must be 5D (NCDHW)");
    TORCH_CHECK(subtract.dim() == 1, "Input tensor 'subtract' must be 1D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor 'x' must be of type float32");
    TORCH_CHECK(subtract.scalar_type() == torch::kFloat32, "Input tensor 'subtract' must be of type float32");
    TORCH_CHECK(x.size(1) == subtract.size(0), "x.size(1) (channels) must match subtract.size(0)");
    
    // Ensure contiguous memory layout for predictable strided access in the kernel
    x = x.contiguous();
    subtract = subtract.contiguous();

    // Get tensor dimensions
    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto D = x.size(2);
    const auto H = x.size(3);
    const auto W = x.size(4);

    // Create the output tensor with shape (N, D, H, W)
    auto output = torch::empty({N, D, H, W}, x.options());

    // Kernel launch configuration
    const int total_spatial_dims = N * D * H * W;
    if (total_spatial_dims == 0) {
        return output; // Handle empty input
    }
    const int block_size = 256;
    const int num_blocks = (total_spatial_dims + block_size - 1) / block_size;

    // Launch the CUDA kernel
    fused_softmax_sub_swish_max_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        subtract.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W
    );
    
    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

# C++ source for the function signature
fused_kernel_cpp_source = "torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor subtract);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_op_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    An optimized model that fuses the last four operations into a single CUDA kernel:
        - ConvTranspose3d
        - MaxPool3d
        - FUSED [Softmax -> Subtract -> Swish -> Max]
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        # These layers are highly optimized in cuDNN, so we leave them as is.
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        
        # This parameter will be passed to our custom kernel.
        self.subtract = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        # 1. Apply standard PyTorch ConvTranspose3d
        x = self.conv_transpose(x)
        
        # 2. Apply standard PyTorch MaxPool3d
        x = self.max_pool(x)
        
        # 3. Apply the custom fused CUDA kernel for the remaining operations
        x = fused_op.fused_op_cuda(x, self.subtract)
        
        return x