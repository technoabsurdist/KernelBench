import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_scale_avgpool_bias_scale_kernel(
    const float* input,
    const float* bias,
    float* output,
    float scale1,
    float scale2,
    int batch_size,
    int channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = batch_size * channels * out_depth * out_height * out_width;
    
    if (idx < total_out) {
        int w_out = idx % out_width;
        int h_out = (idx / out_width) % out_height;
        int d_out = (idx / (out_width * out_height)) % out_depth;
        int c = (idx / (out_width * out_height * out_depth)) % channels;
        int b = idx / (out_width * out_height * out_depth * channels);
        
        // Average pooling with kernel_size=2
        int d_start = d_out * 2;
        int h_start = h_out * 2;
        int w_start = w_out * 2;
        
        float sum = 0.0f;
        int count = 0;
        
        for (int dd = 0; dd < 2; ++dd) {
            for (int hh = 0; hh < 2; ++hh) {
                for (int ww = 0; ww < 2; ++ww) {
                    int d_in = d_start + dd;
                    int h_in = h_start + hh;
                    int w_in = w_start + ww;
                    
                    if (d_in < in_depth && h_in < in_height && w_in < in_width) {
                        int in_idx = ((b * channels + c) * in_depth + d_in) * in_height * in_width + h_in * in_width + w_in;
                        sum += input[in_idx] * scale1;
                        count++;
                    }
                }
            }
        }
        
        float avg = (count > 0) ? (sum / count) : 0.0f;
        output[idx] = (avg + bias[c]) * scale2;
    }
}

torch::Tensor fused_scale_avgpool_bias_scale_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float scale1,
    float scale2)
{
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);
    
    auto out_depth = (in_depth + 1) / 2;
    auto out_height = (in_height + 1) / 2;
    auto out_width = (in_width + 1) / 2;
    
    auto output = torch::zeros({batch_size, channels, out_depth, out_height, out_width}, input.options());
    
    int total_out = batch_size * channels * out_depth * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_out + block_size - 1) / block_size;
    
    fused_scale_avgpool_bias_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scale1,
        scale2,
        batch_size,
        channels,
        in_depth,
        in_height,
        in_width,
        out_depth,
        out_height,
        out_width);
    
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_scale_avgpool_bias_scale_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float scale1,
    float scale2);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_scale_avgpool_bias_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA operations
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.bias = nn.Parameter(torch.randn(bias_shape).squeeze(-1).squeeze(-1).squeeze(-1))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_scale_avgpool_bias_scale_cuda(
            x, 
            self.bias,
            self.scale1.item(),
            self.scale2.item()
        )
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
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape]