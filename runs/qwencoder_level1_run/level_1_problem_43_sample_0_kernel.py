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
    const int input_d1, const int input_d2, const int input_d3,
    const int output_d1, const int output_d2, const int output_d3,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * output_d1 * output_d2 * output_d3;
    
    if (idx >= total_elements) return;
    
    int d3_idx = idx % output_d3;
    int d2_idx = (idx / output_d3) % output_d2;
    int d1_idx = (idx / (output_d3 * output_d2)) % output_d1;
    int c_idx = (idx / (output_d3 * output_d2 * output_d1)) % channels;
    int b_idx = idx / (output_d3 * output_d2 * output_d1 * channels);
    
    int input_d1_start = d1_idx * stride - padding;
    int input_d2_start = d2_idx * stride - padding;
    int input_d3_start = d3_idx * stride - padding;
    
    float max_val = -1e38f; // Negative infinity
    
    for (int kd1 = 0; kd1 < kernel_size; ++kd1) {
        for (int kd2 = 0; kd2 < kernel_size; ++kd2) {
            for (int kd3 = 0; kd3 < kernel_size; ++kd3) {
                int input_idx_d1 = input_d1_start + kd1 * dilation;
                int input_idx_d2 = input_d2_start + kd2 * dilation;
                int input_idx_d3 = input_d3_start + kd3 * dilation;
                
                if (input_idx_d1 >= 0 && input_idx_d1 < input_d1 &&
                    input_idx_d2 >= 0 && input_idx_d2 < input_d2 &&
                    input_idx_d3 >= 0 && input_idx_d3 < input_d3) {
                    
                    int input_idx = b_idx * (channels * input_d1 * input_d2 * input_d3) +
                                    c_idx * (input_d1 * input_d2 * input_d3) +
                                    input_idx_d1 * (input_d2 * input_d3) +
                                    input_idx_d2 * input_d3 +
                                    input_idx_d3;
                    
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
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Set the CUDA device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int channels = input_sizes[1];
    int input_d1 = input_sizes[2];
    int input_d2 = input_sizes[3];
    int input_d3 = input_sizes[4];
    
    // Calculate output dimensions
    int output_d1 = (input_d1 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_d2 = (input_d2 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_d3 = (input_d3 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, channels, output_d1, output_d2, output_d3}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    int total_elements = batch_size * channels * output_d1 * output_d2 * output_d3;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    maxpool3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_d1, input_d2, input_d3,
        output_d1, output_d2, output_d3,
        kernel_size,
        stride,
        padding,
        dilation
    );
    
    return output;
}
"""

maxpool3d_cpp_source = """
torch::Tensor maxpool3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);
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
        Initializes the Max Pooling 3D layer.

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
        
        # Store the custom CUDA function
        self.maxpool3d_cuda_fn = maxpool3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 3D applied.
        """
        # Use the custom CUDA implementation
        return self.maxpool3d_cuda_fn.maxpool3d_cuda(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation
        )

batch_size = 16
channels = 32
dim1 = 128
dim2 = 128
dim3 = 128
kernel_size = 3
stride = 2
padding = 1
dilation = 3

def get_inputs():
    x = torch.rand(batch_size, channels, dim1, dim2, dim3).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]