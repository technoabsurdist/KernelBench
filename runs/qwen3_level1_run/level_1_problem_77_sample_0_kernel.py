import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void col2im_3d_kernel(
    const float* data_col,
    const int channels,
    const int depth,
    const int height,
    const int width,
    const int ksize_d,
    const int ksize_h,
    const int ksize_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int dilation_d,
    const int dilation_h,
    const int dilation_w,
    float* data_im) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = channels * depth * height * width;
    
    if (index >= total_threads) return;
    
    int w = index % width;
    index /= width;
    int h = index % height;
    index /= height;
    int d = index % depth;
    int c = index / depth;
    
    float val = 0;
    for (int kd = 0; kd < ksize_d; ++kd) {
        for (int kh = 0; kh < ksize_h; ++kh) {
            for (int kw = 0; kw < ksize_w; ++kw) {
                int d_out = d + pad_d - kd * dilation_d;
                int h_out = h + pad_h - kh * dilation_h;
                int w_out = w + pad_w - kw * dilation_w;
                
                if (d_out % stride_d == 0 && h_out % stride_h == 0 && w_out % stride_w == 0) {
                    d_out /= stride_d;
                    h_out /= stride_h;
                    w_out /= stride_w;
                    
                    if (d_out >= 0 && h_out >= 0 && w_out >= 0) {
                        int col_idx = (((c * ksize_d + kd) * ksize_h + kh) * ksize_w + kw) * 
                                      ((depth + 2 * pad_d - (dilation_d * (ksize_d - 1) + 1)) / stride_d + 1) *
                                      ((height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1) *
                                      ((width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1);
                        
                        col_idx += d_out * ((height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1) * 
                                   ((width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1);
                        col_idx += h_out * ((width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1);
                        col_idx += w_out;
                        
                        val += data_col[col_idx];
                    }
                }
            }
        }
    }
    data_im[((c * depth + d) * height + h) * width + w] = val;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool with_bias) {
    
    // Get dimensions
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);
    
    auto out_channels = weight.size(0);
    auto ksize_d = kernel_size;
    auto ksize_h = kernel_size;
    auto ksize_w = kernel_size;
    
    // Calculate output dimensions
    auto out_depth = (in_depth - 1) * stride + (ksize_d - 1) * dilation + 1 - 2 * padding;
    auto out_height = (in_height - 1) * stride + (ksize_h - 1) * dilation + 1 - 2 * padding;
    auto out_width = (in_width - 1) * stride + (ksize_w - 1) * dilation + 1 - 2 * padding;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    // Get raw pointers
    float* input_ptr = input.data_ptr<float>();
    float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* bias_ptr = with_bias ? bias.data_ptr<float>() : nullptr;
    
    // For each batch
    for (int b = 0; b < batch_size; b++) {
        // Calculate col dimension
        auto col_depth = (out_depth + 2 * padding - (dilation * (ksize_d - 1) + 1)) / stride + 1;
        auto col_height = (out_height + 2 * padding - (dilation * (ksize_h - 1) + 1)) / stride + 1;
        auto col_width = (out_width + 2 * padding - (dilation * (ksize_w - 1) + 1)) / stride + 1;
        
        auto col_size = out_channels * col_depth * col_height * col_width;
        auto col = torch::zeros({col_size}, torch::TensorOptions().dtype(input.dtype()).device(input.device()));
        float* col_ptr = col.data_ptr<float>();
        
        // Im2col operation (simplified)
        for (int c = 0; c < in_channels; c++) {
            for (int kd = 0; kd < ksize_d; kd++) {
                for (int kh = 0; kh < ksize_h; kh++) {
                    for (int kw = 0; kw < ksize_w; kw++) {
                        for (int d = 0; d < in_depth; d++) {
                            for (int h = 0; h < in_height; h++) {
                                for (int w = 0; w < in_width; w++) {
                                    int input_idx = (((b * in_channels + c) * in_depth + d) * in_height + h) * in_width + w;
                                    int weight_idx = (((c * ksize_d + kd) * ksize_h + kh) * ksize_w + kw) * out_channels;
                                    int col_idx = ((((b * out_channels) * ksize_d + kd) * ksize_h + kh) * ksize_w + kw) * 
                                                  (in_depth * in_height * in_width) + (d * in_height + h) * in_width + w;
                                    
                                    if (col_idx < col_size) {
                                        for (int oc = 0; oc < out_channels; oc++) {
                                            col_ptr[col_idx * out_channels + oc] += 
                                                input_ptr[input_idx] * weight_ptr[weight_idx + oc];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Col2im operation
        const int block_size = 256;
        const int num_blocks = (out_channels * out_depth * out_height * out_width + block_size - 1) / block_size;
        
        col2im_3d_kernel<<<num_blocks, block_size>>>(
            col_ptr,
            out_channels,
            out_depth,
            out_height,
            out_width,
            ksize_d,
            ksize_h,
            ksize_w,
            padding,
            padding,
            padding,
            stride,
            stride,
            stride,
            dilation,
            dilation,
            dilation,
            output_ptr + b * out_channels * out_depth * out_height * out_width
        );
        
        // Add bias if needed
        if (with_bias) {
            for (int oc = 0; oc < out_channels; oc++) {
                float bias_val = bias_ptr[oc];
                for (int d = 0; d < out_depth; d++) {
                    for (int h = 0; h < out_height; h++) {
                        for (int w = 0; w < out_width; w++) {
                            int out_idx = (((b * out_channels + oc) * out_depth + d) * out_height + h) * out_width + w;
                            output_ptr[out_idx] += bias_val;
                        }
                    }
                }
                cudaDeviceSynchronize();
            }
        }
    }
    
    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool with_bias);
"""

# Compile the inline CUDA code for 3D transposed convolution
conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Performs a 3D transposed convolution operation with square input and square kernel,
    and supports padding, dilation, and stride, using custom CUDA implementation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.with_bias = bias
        
        # Initialize weight parameter
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        
        # Initialize bias parameter if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.conv_transpose3d_op = conv_transpose3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return self.conv_transpose3d_op.conv_transpose3d_cuda(
            x, 
            self.weight, 
            self.bias if self.bias is not None else torch.empty(0),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.with_bias
        )