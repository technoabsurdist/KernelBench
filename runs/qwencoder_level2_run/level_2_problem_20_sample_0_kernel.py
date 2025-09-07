import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_conv3d_transpose_elementwise_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA kernel for element-wise operations: (x + bias) + x * x + x
__global__ void fused_elementwise_kernel(
    const float* input,
    const float* bias,
    float* output,
    int num_elements,
    int bias_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int bias_idx = idx % bias_size;
        float x = input[idx];
        float biased = x + bias[bias_idx];
        float added = biased + x;
        float multiplied = added * x;
        output[idx] = multiplied + x;
    }
}

// Simplified convolution transpose forward (basic implementation for demonstration)
// Note: In practice, you would use cuDNN for performance
__global__ void conv3d_transpose_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding
) {
    // This is a simplified placeholder - real implementation would be much more complex
    // For optimization purposes, we'll focus on the elementwise fusion which has more impact
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    if (idx < total_elements) {
        output[idx] = input[idx % (batch_size * in_channels * input_depth * input_height * input_width)] + bias[idx % out_channels];
    }
}

torch::Tensor fused_conv3d_transpose_elementwise_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding
) {
    // Perform conv transpose
    auto conv_output = torch::conv_transpose3d(input, weight, torch::Tensor(), stride, padding, output_padding);
    
    // Get dimensions
    int num_elements = conv_output.numel();
    int bias_size = bias.size(0);
    
    // Allocate output tensor
    auto output = torch::zeros_like(conv_output);
    
    // Launch fused elementwise kernel
    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    
    fused_elementwise_kernel<<<num_blocks, block_size>>>(
        conv_output.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        bias_size
    );
    
    return output;
}
"""

fused_conv3d_transpose_elementwise_cpp_source = """
torch::Tensor fused_conv3d_transpose_elementwise_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding
);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_conv3d_transpose_elementwise",
    cpp_sources=fused_conv3d_transpose_elementwise_cpp_source,
    cuda_sources=fused_conv3d_transpose_elementwise_source,
    functions=["fused_conv3d_transpose_elementwise_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        
        # Initialize convolution transpose weights
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        # Use our custom fused operation
        return fused_ops.fused_conv3d_transpose_elementwise_cuda(
            x, self.weight, self.bias, self.stride, self.padding, self.output_padding
        )