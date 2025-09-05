import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused subtract-tanh-subtract-avgpool
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_subtract_tanh_subtract_avgpool_kernel(
    const float* input,
    float* output,
    float subtract1_value,
    float subtract2_value,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_size,
    int out_height,
    int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_elements = batch_size * channels * out_height * out_width;
    
    if (idx < total_out_elements) {
        int w_out = idx % out_width;
        int h_out = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % channels;
        int b = idx / (channels * out_height * out_width);
        
        float sum = 0.0f;
        int count = 0;
        
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = h_out * kernel_size + kh;
                int w_in = w_out * kernel_size + kw;
                
                if (h_in < height && w_in < width) {
                    int in_idx = b * channels * height * width + 
                                 c * height * width + 
                                 h_in * width + w_in;
                    
                    float val = input[in_idx];
                    val = val - subtract1_value;
                    val = tanhf(val);
                    val = val - subtract2_value;
                    sum += val;
                    count++;
                }
            }
        }
        
        output[idx] = sum / count;
    }
}

torch::Tensor fused_subtract_tanh_subtract_avgpool_cuda(
    torch::Tensor input,
    float subtract1_value,
    float subtract2_value,
    int kernel_size
) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto out_height = height / kernel_size;
    auto out_width = width / kernel_size;
    
    auto output = torch::zeros({batch_size, channels, out_height, out_width}, input.options());
    
    const int block_size = 256;
    const int num_elements = batch_size * channels * out_height * out_width;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    
    fused_subtract_tanh_subtract_avgpool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        subtract1_value,
        subtract2_value,
        batch_size,
        channels,
        height,
        width,
        kernel_size,
        out_height,
        out_width
    );
    
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_subtract_tanh_subtract_avgpool_cuda(
    torch::Tensor input,
    float subtract1_value,
    float subtract2_value,
    int kernel_size
);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_subtract_tanh_subtract_avgpool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for fused operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_subtract_tanh_subtract_avgpool_cuda(
            x, 
            self.subtract1_value, 
            self.subtract2_value,
            self.kernel_size_pool
        )
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]