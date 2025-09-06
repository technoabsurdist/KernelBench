import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Max Pooling 3D
maxpool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void maxpool3d_kernel(
    const float* input,
    float* output,
    const int batch_size,
    const int channels,
    const int input_d,
    const int input_h,
    const int input_w,
    const int output_d,
    const int output_h,
    const int output_w,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int dilation_d,
    const int dilation_h,
    const int dilation_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * output_d * output_h * output_w;
    
    if (idx >= total_elements) return;
    
    int w_out = idx % output_w;
    int h_out = (idx / output_w) % output_h;
    int d_out = (idx / (output_w * output_h)) % output_d;
    int c = (idx / (output_w * output_h * output_d)) % channels;
    int b = idx / (output_w * output_h * output_d * channels);
    
    int input_w_start = w_out * stride_w - padding_w;
    int input_h_start = h_out * stride_h - padding_h;
    int input_d_start = d_out * stride_d - padding_d;
    
    float max_val = -FLT_MAX;
    
    for (int kd = 0; kd < kernel_d; kd++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int input_d_coord = input_d_start + kd * dilation_d;
                int input_h_coord = input_h_start + kh * dilation_h;
                int input_w_coord = input_w_start + kw * dilation_w;
                
                if (input_d_coord >= 0 && input_d_coord < input_d &&
                    input_h_coord >= 0 && input_h_coord < input_h &&
                    input_w_coord >= 0 && input_w_coord < input_w) {
                    
                    int input_idx = b * (channels * input_d * input_h * input_w) +
                                    c * (input_d * input_h * input_w) +
                                    input_d_coord * (input_h * input_w) +
                                    input_h_coord * input_w +
                                    input_w_coord;
                    
                    float val = input[input_idx];
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
        }
    }
    
    output[idx] = max_val;
}

torch::Tensor maxpool3d_cuda(
    torch::Tensor input,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int channels = input_sizes[1];
    int input_d = input_sizes[2];
    int input_h = input_sizes[3];
    int input_w = input_sizes[4];
    
    // Calculate output dimensions
    int output_d = (input_d + 2 * padding_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    int output_h = (input_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int output_w = (input_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    auto output = torch::zeros({batch_size, channels, output_d, output_h, output_w}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    int total_elements = batch_size * channels * output_d * output_h * output_w;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    maxpool3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_d,
        input_h,
        input_w,
        output_d,
        output_h,
        output_w,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        dilation_d,
        dilation_h,
        dilation_w
    );
    
    return output;
}
"""

maxpool3d_cpp_source = """
torch::Tensor maxpool3d_cuda(
    torch::Tensor input,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w
);
"""

# Compile the inline CUDA code for Max Pooling 3D
maxpool3d = load_inline(
    name="maxpool3d",
    cpp_sources=maxpool3d_cpp_source,
    cuda_sources=maxpool3d_source,
    functions=["maxpool3d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 3D with custom CUDA kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        """
        Initializes the Max Pooling 3D layer with custom CUDA implementation.

        Args:
            kernel_size (int): Size of the kernel for the max pooling operation.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which means stride is equal to kernel_size.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return indices of the maximum values. Defaults to False.
            ceil_mode (bool, optional): When True, the output size is ceil(input_size / stride) instead of floor. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        
        # Handle tuple inputs for kernel_size, stride, padding, dilation
        if isinstance(self.kernel_size, int):
            self.kernel_d = self.kernel_h = self.kernel_w = self.kernel_size
        else:
            self.kernel_d, self.kernel_h, self.kernel_w = self.kernel_size
            
        if isinstance(self.stride, int):
            self.stride_d = self.stride_h = self.stride_w = self.stride
        else:
            self.stride_d, self.stride_h, self.stride_w = self.stride
            
        if isinstance(self.padding, int):
            self.padding_d = self.padding_h = self.padding_w = self.padding
        else:
            self.padding_d, self.padding_h, self.padding_w = self.padding
            
        if isinstance(self.dilation, int):
            self.dilation_d = self.dilation_h = self.dilation_w = self.dilation
        else:
            self.dilation_d, self.dilation_h, self.dilation_w = self.dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 3D applied.
        """
        if self.return_indices or self.ceil_mode:
            # Fall back to PyTorch implementation for unsupported features
            return nn.functional.max_pool3d(
                x, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                padding=self.padding, 
                dilation=self.dilation, 
                return_indices=self.return_indices, 
                ceil_mode=self.ceil_mode
            )
        
        return maxpool3d.maxpool3d_cuda(
            x,
            self.kernel_d, self.kernel_h, self.kernel_w,
            self.stride_d, self.stride_h, self.stride_w,
            self.padding_d, self.padding_h, self.padding_w,
            self.dilation_d, self.dilation_h, self.dilation_w
        )