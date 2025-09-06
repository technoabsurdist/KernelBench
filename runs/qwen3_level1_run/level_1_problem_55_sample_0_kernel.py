import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D convolution
conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void im2col_kernel(const float* data_im, float* data_col,
                              int channels, int height, int width,
                              int kernel_h, int kernel_w,
                              int pad_h, int pad_w,
                              int stride_h, int stride_w,
                              int dilation_h, int dilation_w,
                              int height_col, int width_col) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = channels * height_col * width_col;
    
    if (index >= total_threads) return;
    
    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    int channel_in = index / height_col;
    
    int channel_out = channel_in * kernel_h * kernel_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    
    float* data_col_ptr = data_col + (channel_out * height_col + h_out) * width_col + w_out;
    const float* data_im_ptr = data_im + (channel_in * height + h_in) * width + w_in;
    
    for (int i = 0; i < kernel_h; ++i) {
        for (int j = 0; j < kernel_w; ++j) {
            int h = h_in + i * dilation_h;
            int w = w_in + j * dilation_w;
            
            *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                            data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
            data_col_ptr += height_col * width_col;
        }
    }
}

__global__ void conv2d_kernel(const float* input, const float* weight, float* output,
                              int batch_size, int in_channels, int out_channels,
                              int height_col, int width_col, int kernel_size_sq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * height_col * width_col;
    
    if (idx >= total_elements) return;
    
    int w_out = idx % width_col;
    int h_out = (idx / width_col) % height_col;
    int out_ch = (idx / (width_col * height_col)) % out_channels;
    int batch = idx / (width_col * height_col * out_channels);
    
    float sum = 0.0f;
    
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int k = 0; k < kernel_size_sq; ++k) {
            int input_idx = ((batch * in_channels + in_ch) * kernel_size_sq + k) * height_col * width_col + 
                           h_out * width_col + w_out;
            int weight_idx = (out_ch * in_channels + in_ch) * kernel_size_sq + k;
            sum += input[input_idx] * weight[weight_idx];
        }
    }
    
    output[idx] = sum;
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                          int stride_h, int stride_w, int pad_h, int pad_w,
                          int dilation_h, int dilation_w) {
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int height = input_sizes[2];
    int width = input_sizes[3];
    int out_channels = weight_sizes[0];
    int kernel_h = weight_sizes[2];
    int kernel_w = weight_sizes[3];
    
    int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    
    // Im2col transformation
    int channels_col = in_channels * kernel_h * kernel_w;
    auto col_buffer = torch::zeros({batch_size, channels_col, height_out, width_out}, 
                                   torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    int num_blocks = (channels_col * height_out * width_out + block_size - 1) / block_size;
    
    for (int b = 0; b < batch_size; ++b) {
        im2col_kernel<<<num_blocks, block_size>>>(
            input[b].data_ptr<float>(),
            col_buffer[b].data_ptr<float>(),
            in_channels, height, width,
            kernel_h, kernel_w,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            height_out, width_out
        );
    }
    
    // Convolution using matrix multiplication
    auto output = torch::zeros({batch_size, out_channels, height_out, width_out},
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_elements = batch_size * out_channels * height_out * width_out;
    num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv2d_kernel<<<num_blocks, block_size>>>(
        col_buffer.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        height_out, width_out, kernel_h * kernel_w
    );
    
    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                          int stride_h, int stride_w, int pad_h, int pad_w,
                          int dilation_h, int dilation_w);
"""

# Compile the inline CUDA code for 2D convolution
conv2d_module = load_inline(
    name="conv2d_module",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with an asymmetric input and a square kernel,
    optimized with custom CUDA kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Ensure groups is 1 for this implementation
        if groups != 1:
            raise NotImplementedError("Grouped convolutions are not supported in this custom implementation")
        
        # Create weight parameter
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        
        # Create bias parameter if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
            
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Call custom CUDA convolution
        output = conv2d_module.conv2d_cuda(
            x, self.weight,
            self.stride, self.stride,
            self.padding, self.padding,
            self.dilation, self.dilation
        )
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
            
        return output