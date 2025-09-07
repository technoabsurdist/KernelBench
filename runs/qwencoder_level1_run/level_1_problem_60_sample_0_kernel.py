import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void conv3d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * output_d * output_h * output_w;
    
    if (out_idx >= total_outputs) return;
    
    int w_out = out_idx % output_w;
    out_idx /= output_w;
    int h_out = out_idx % output_h;
    out_idx /= output_h;
    int d_out = out_idx % output_d;
    out_idx /= output_d;
    int out_ch = out_idx % out_channels;
    int batch = out_idx / out_channels;
    
    float sum = 0.0f;
    
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int d_in = d_out * stride_d - pad_d + kd;
                    int h_in = h_out * stride_h - pad_h + kh;
                    int w_in = w_out * stride_w - pad_w + kw;
                    
                    if (d_in >= 0 && d_in < input_d &&
                        h_in >= 0 && h_in < input_h &&
                        w_in >= 0 && w_in < input_w) {
                        
                        int input_idx = batch * (in_channels * input_d * input_h * input_w) +
                                       in_ch * (input_d * input_h * input_w) +
                                       d_in * (input_h * input_w) +
                                       h_in * input_w +
                                       w_in;
                                       
                        int weight_idx = out_ch * (in_channels * kernel_d * kernel_h * kernel_w) +
                                        in_ch * (kernel_d * kernel_h * kernel_w) +
                                        kd * (kernel_h * kernel_w) +
                                        kh * kernel_w +
                                        kw;
                                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    output[out_idx] = sum;
}

torch::Tensor conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding
) {
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_d = input_sizes[2];
    int input_h = input_sizes[3];
    int input_w = input_sizes[4];
    
    int out_channels = weight_sizes[0];
    int kernel_d = weight_sizes[2];
    int kernel_h = weight_sizes[3];
    int kernel_w = weight_sizes[4];
    
    int stride_d = stride[0];
    int stride_h = stride[1];
    int stride_w = stride[2];
    
    int pad_d = padding[0];
    int pad_h = padding[1];
    int pad_w = padding[2];
    
    int output_d = (input_d + 2 * pad_d - kernel_d) / stride_d + 1;
    int output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_d, output_h, output_w}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * output_d * output_h * output_w + block_size - 1) / block_size;
    
    conv3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w
    );
    
    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding
);
"""

# Compile the inline CUDA code for 3D convolution
conv3d = load_inline(
    name="conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with a square input and an asymmetric kernel,
    optimized with custom CUDA kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        self.bias = bias
        
        # Initialize weight parameter
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias_param = None
            
        self.conv3d_op = conv3d
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution with custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, width, height, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, width_out, height_out, depth_out).
        """
        output = self.conv3d_op.conv3d_cuda(x, self.weight, self.stride, self.padding)
        
        if self.bias_param is not None:
            # Add bias if needed
            for i in range(output.dim() - 1):
                if i < 1: continue
                self.bias_param = self.bias_param.unsqueeze(-1)
            output = output + self.bias_param
        
        return output