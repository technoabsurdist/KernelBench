import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for the Channel Shuffle operation
channel_shuffle_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void channel_shuffle_kernel(
    const float* input,
    float* output,
    const int N,
    const int C,
    const int H,
    const int W,
    const int groups) {

    const int total_elements = N * C * H * W;
    const int channels_per_group = C / groups;
    const int spatial_dim = H * W;

    // Using a grid-stride loop to handle any number of elements with a flexible number of blocks
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_elements;
         i += gridDim.x * blockDim.x) {

        // Decompose the linear output index 'i' into 4D coordinates (n, c_out, h, w)
        const int w_out = i % W;
        const int h_out = (i / W) % H;
        const int c_out = (i / spatial_dim) % C;
        const int n_out = i / (C * spatial_dim);

        // This is the inverse of the shuffle operation.
        // The forward PyTorch operation is equivalent to:
        //   c_new = (c_old % channels_per_group) * groups + (c_old / channels_per_group)
        // We are given c_out (c_new) and need to find c_in (c_old).
        //
        // Let c_out = c_new, c_in = c_old, G = groups, CPG = channels_per_group
        // c_new = (c_old % CPG) * G + (c_old / CPG)
        // Let g_old = c_old / CPG, cpg_old = c_old % CPG
        // c_new = cpg_old * G + g_old
        //
        // From c_new, we can find g_old and cpg_old:
        // g_old = c_new % G
        // cpg_old = c_new / G
        //
        // Now, reconstruct c_old:
        // c_old = g_old * CPG + cpg_old
        const int g_old = c_out % groups;
        const int cpg_old = c_out / groups;
        const int c_in = g_old * channels_per_group + cpg_old;

        // Calculate the source linear index
        const int in_idx = n_out * C * spatial_dim + c_in * spatial_dim + h_out * W + w_out;

        output[i] = input[in_idx];
    }
}

torch::Tensor channel_shuffle_cuda(torch::Tensor x, int64_t groups) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4-dimensional");
    
    // Ensure the input tensor is contiguous in memory for the kernel to work correctly.
    // The original PyTorch implementation also has a .contiguous() call.
    x = x.contiguous();

    const auto batch_size = x.size(0);
    const auto channels = x.size(1);
    const auto height = x.size(2);
    const auto width = x.size(3);

    TORCH_CHECK(channels > 0 && channels % groups == 0, "Number of channels must be divisible by groups");

    auto output = torch::empty_like(x);
    const int64_t total_elements = x.numel();
    if (total_elements == 0) {
        return output;
    }

    // Kernel launch configuration
    const int block_size = 1024;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    channel_shuffle_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        groups);

    // Check for any CUDA errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

channel_shuffle_cpp_source = (
    "torch::Tensor channel_shuffle_cuda(torch::Tensor x, int64_t groups);"
)

# Compile the inline CUDA code for channel shuffle.
# This is done once when the Python module is loaded.
channel_shuffle = load_inline(
    name="channel_shuffle",
    cpp_sources=channel_shuffle_cpp_source,
    cuda_sources=channel_shuffle_source,
    functions=["channel_shuffle_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        """
        Optimized ShuffleNet unit with a custom CUDA kernel for Channel Shuffle.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param groups: Number of groups for group convolution and shuffling.
        """
        super(ModelNew, self).__init__()
        
        # Ensure the output channels are divisible by 4 as in the original architecture
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        # First 1x1 group convolution
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # Depthwise 3x3 convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # Second 1x1 group convolution
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Store groups to pass to the custom CUDA kernel
        self.groups = groups
        
        # Shortcut connection
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass for the optimized ShuffleNet unit.

        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Replace the original shuffle operation with our custom CUDA kernel
        out = channel_shuffle.channel_shuffle_cuda(out, self.groups)
        
        out = F.relu(self.bn3(self.conv3(out)))
        
        out += self.shortcut(x)
        return out