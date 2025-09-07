import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations: scale + avg_pool + bias + scale
fused_conv3d_operations_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_operations_kernel(
    const float* input,
    const float scale1,
    const float* bias,
    const float scale2,
    float* output,
    const int batch_size,
    const int channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * output_depth * output_height * output_width;
    
    if (idx < total_elements) {
        int w = idx % output_width;
        int h = (idx / output_width) % output_height;
        int d = (idx / (output_width * output_height)) % output_depth;
        int c = (idx / (output_width * output_height * output_depth)) % channels;
        
        // Average pooling indices in input tensor
        int input_w_start = w * 2;
        int input_h_start = h * 2;
        int input_d_start = d * 2;
        
        float sum = 0.0f;
        int count = 0;
        
        // 2x2x2 average pooling
        for (int pd = 0; pd < 2 && (input_d_start + pd) < input_depth; pd++) {
            for (int ph = 0; ph < 2 && (input_h_start + ph) < input_height; ph++) {
                for (int pw = 0; pw < 2 && (input_w_start + pw) < input_width; pw++) {
                    int input_idx = ((c * input_depth + (input_d_start + pd)) * input_height + (input_h_start + ph)) * input_width + (input_w_start + pw);
                    sum += input[input_idx];
                    count++;
                }
            }
        }
        
        float avg_val = (count > 0) ? (sum / count) : 0.0f;
        float scaled_val = avg_val * scale1;
        float biased_val = scaled_val + bias[c];
        output[idx] = biased_val * scale2;
    }
}

torch::Tensor fused_conv3d_operations_cuda(
    torch::Tensor input,
    float scale1,
    torch::Tensor bias,
    float scale2
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto input_depth = input.size(2);
    auto input_height = input.size(3);
    auto input_width = input.size(4);
    
    // Calculate output dimensions after 2x2x2 average pooling
    auto output_depth = (input_depth + 1) / 2;
    auto output_height = (input_height + 1) / 2;
    auto output_width = (input_width + 1) / 2;
    
    auto output = torch::zeros({batch_size, channels, output_depth, output_height, output_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    const int total_elements = batch_size * channels * output_depth * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_operations_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        scale1,
        bias.data_ptr<float>(),
        scale2,
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width
    );
    
    return output;
}
"""

fused_conv3d_operations_cpp_source = """
torch::Tensor fused_conv3d_operations_cuda(
    torch::Tensor input,
    float scale1,
    torch::Tensor bias,
    float scale2
);
"""

# Compile the inline CUDA code for fused operations
fused_conv3d_operations = load_inline(
    name="fused_conv3d_operations",
    cpp_sources=fused_conv3d_operations_cpp_source,
    cuda_sources=fused_conv3d_operations_source,
    functions=["fused_conv3d_operations_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        self.fused_ops = fused_conv3d_operations

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fuse scale1, avg_pool, bias add, and scale2 operations
        x = self.fused_ops.fused_conv3d_operations_cuda(x, self.scale1.item(), self.bias.squeeze(), self.scale2.item())
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale1 = 0.5
scale2 = 1.0
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape]