import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused logsumexp + relu
logsumexp_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>

__global__ void logsumexp_relu_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_size = batch_size * spatial_size;
    
    if (idx < total_output_size) {
        int batch_idx = idx / spatial_size;
        int spatial_idx = idx % spatial_size;
        
        // Find max value across channels for numerical stability
        float max_val = -FLT_MAX;
        for (int c = 0; c < channels; c++) {
            int input_idx = batch_idx * channels * spatial_size + c * spatial_size + spatial_idx;
            max_val = fmaxf(max_val, input[input_idx]);
        }
        
        // Compute sum of exp(x - max_val)
        float sum_exp = 0.0f;
        for (int c = 0; c < channels; c++) {
            int input_idx = batch_idx * channels * spatial_size + c * spatial_size + spatial_idx;
            sum_exp += expf(input[input_idx] - max_val);
        }
        
        // Compute log(sum_exp) + max_val and apply ReLU
        float result = logf(sum_exp) + max_val;
        output[idx] = fmaxf(0.0f, result);
    }
}

torch::Tensor logsumexp_relu_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto spatial_size = depth * height * width;
    
    auto output = torch::zeros({batch_size, 1, depth, height, width}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * spatial_size + block_size - 1) / block_size;
    
    logsumexp_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        spatial_size
    );
    
    return output;
}
"""

logsumexp_relu_cpp_source = "torch::Tensor logsumexp_relu_cuda(torch::Tensor input);"

# Compile the inline CUDA code
logsumexp_relu = load_inline(
    name="logsumexp_relu",
    cpp_sources=logsumexp_relu_cpp_source,
    cuda_sources=logsumexp_relu_source,
    functions=["logsumexp_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, max pooling, and fused log sum exp + ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.logsumexp_relu = logsumexp_relu

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width')
        """
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.logsumexp_relu.logsumexp_relu_cuda(x)
        return x

batch_size = 4
in_channels = 32
out_channels = 64
depth, height, width = 32, 128, 128
kernel_size = 3
stride = 1
padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]