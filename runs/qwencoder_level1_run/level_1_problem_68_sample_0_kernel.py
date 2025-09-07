import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_channels,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int opad_d,
    int opad_h,
    int opad_w
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (out_idx >= total_out_elements) return;
    
    int tmp = out_idx;
    int w_idx = tmp % out_width;
    tmp /= out_width;
    int h_idx = tmp % out_height;
    tmp /= out_height;
    int d_idx = tmp % out_depth;
    tmp /= out_depth;
    int c_out = tmp % out_channels;
    int batch_idx = tmp / out_channels;
    
    float sum = 0.0f;
    
    // Calculate input indices that would contribute to this output position
    for (int kd = 0; kd < kernel_d; kd++) {
        int in_d = d_idx + pad_d - kd;
        if (in_d % stride_d != 0) continue;
        in_d /= stride_d;
        if (in_d < 0 || in_d >= in_depth) continue;
        
        for (int kh = 0; kh < kernel_h; kh++) {
            int in_h = h_idx + pad_h - kh;
            if (in_h % stride_h != 0) continue;
            in_h /= stride_h;
            if (in_h < 0 || in_h >= in_height) continue;
            
            for (int kw = 0; kw < kernel_w; kw++) {
                int in_w = w_idx + pad_w - kw;
                if (in_w % stride_w != 0) continue;
                in_w /= stride_w;
                if (in_w < 0 || in_w >= in_width) continue;
                
                for (int c_in = 0; c_in < in_channels; c_in++) {
                    int input_idx = batch_idx * (in_channels * in_depth * in_height * in_width) +
                                   c_in * (in_depth * in_height * in_width) +
                                   in_d * (in_height * in_width) +
                                   in_h * in_width +
                                   in_w;
                                   
                    int weight_idx = c_in * (out_channels * kernel_d * kernel_h * kernel_w) +
                                    c_out * (kernel_d * kernel_h * kernel_w) +
                                    kd * (kernel_h * kernel_w) +
                                    kh * kernel_w +
                                    kw;
                                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    output[out_idx] = sum;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef output_padding
) {
    // Set the CUDA device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int in_depth = input_sizes[2];
    int in_height = input_sizes[3];
    int in_width = input_sizes[4];
    
    int out_channels = weight_sizes[1];
    int kernel_d = weight_sizes[2];
    int kernel_h = weight_sizes[3];
    int kernel_w = weight_sizes[4];
    
    int stride_d = stride[0];
    int stride_h = stride[1];
    int stride_w = stride[2];
    
    int pad_d = padding[0];
    int pad_h = padding[1];
    int pad_w = padding[2];
    
    int opad_d = output_padding[0];
    int opad_h = output_padding[1];
    int opad_w = output_padding[2];
    
    // Calculate output dimensions
    int out_depth = (in_depth - 1) * stride_d - 2 * pad_d + kernel_d + opad_d;
    int out_height = (in_height - 1) * stride_h - 2 * pad_h + kernel_h + opad_h;
    int out_width = (in_width - 1) * stride_w - 2 * pad_w + kernel_w + opad_w;
    
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    int total_out_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_out_elements + block_size - 1) / block_size;
    
    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_depth,
        in_height,
        in_width,
        out_channels,
        out_depth,
        out_height,
        out_width,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        opad_d,
        opad_h,
        opad_w
    );
    
    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef output_padding
);
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
    Performs a transposed 3D convolution with a square input and an asymmetric kernel using custom CUDA implementation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_depth, kernel_width, kernel_height), 
                             where kernel_width == kernel_height.
        stride (tuple, optional): Stride of the convolution. Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input. Defaults to (0, 0, 0).
        output_padding (tuple, optional): Additional size added to one side of the output shape. Defaults to (0, 0, 0).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        if groups != 1:
            raise NotImplementedError("Only groups=1 is supported in this custom implementation")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weight parameter
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, *kernel_size))
        
        # Initialize bias parameter if needed
        if bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias_param = None
            
        self.conv_transpose3d_cuda = conv_transpose3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using custom CUDA implementation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        output = self.conv_transpose3d_cuda.conv_transpose3d_cuda(
            x, 
            self.weight, 
            self.stride, 
            self.padding, 
            self.output_padding
        )
        
        # Add bias if needed
        if self.bias and self.bias_param is not None:
            # Reshape bias to broadcast correctly
            bias_view = self.bias_param.view(1, self.out_channels, 1, 1, 1)
            output = output + bias_view
            
        return output