import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Max Pooling 1D
maxpool1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void maxpool1d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int num_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int output_idx = threadIdx.x + blockIdx.z * blockDim.x;
    
    if (batch_idx >= batch_size || channel_idx >= num_channels || output_idx >= output_length)
        return;
        
    int input_start = output_idx * stride - padding;
    float max_val = -FLT_MAX;
    
    for (int k = 0; k < kernel_size; ++k) {
        int input_idx = input_start + k * dilation;
        if (input_idx >= 0 && input_idx < input_length) {
            float val = input[((batch_idx * num_channels + channel_idx) * input_length) + input_idx];
            max_val = fmaxf(max_val, val);
        }
    }
    
    output[((batch_idx * num_channels + channel_idx) * output_length) + output_idx] = max_val;
}

torch::Tensor maxpool1d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    int batch_size = input.size(0);
    int num_channels = input.size(1);
    int input_length = input.size(2);
    
    // Calculate output length
    int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    
    auto output = torch::zeros({batch_size, num_channels, output_length}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    if (output_length <= 0) return output;
    
    const int threads_per_block = min(1024, output_length);
    const int blocks_per_grid_z = (output_length + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(batch_size, num_channels, blocks_per_grid_z);
    dim3 block(threads_per_block);
    
    maxpool1d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation
    );
    
    return output;
}
"""

maxpool1d_cpp_source = """
torch::Tensor maxpool1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);
"""

# Compile the inline CUDA code for Max Pooling 1D
maxpool1d = load_inline(
    name="maxpool1d",
    cpp_sources=maxpool1d_cpp_source,
    cuda_sources=maxpool1d_source,
    functions=["maxpool1d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 1D with custom CUDA kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        """
        Initializes the Max Pooling 1D layer.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
            padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.maxpool1d_cuda = maxpool1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 1D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 1D applied, shape (batch_size, num_features, output_sequence_length).
        """
        if self.return_indices:
            # Fall back to PyTorch implementation if indices are needed
            return F.max_pool1d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices)
        else:
            # Use custom CUDA implementation
            return self.maxpool1d_cuda.maxpool1d_cuda(x, self.kernel_size, self.stride, self.padding, self.dilation)

batch_size = 64
features = 192
sequence_length = 65536

kernel_size = 8
stride      = 1
padding     = 4
dilation    = 3            

return_indices = False

def get_inputs():
    x = torch.rand(batch_size, features, sequence_length).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]