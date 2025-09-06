import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for a more efficient Channel Shuffle operation
channel_shuffle_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for the channel shuffle operation.
// This kernel directly maps each element from its source location to its
// shuffled destination, avoiding the overhead of multiple view/transpose
// operations in PyTorch.
__global__ void channel_shuffle_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int total_elements,
    const int C, const int H, const int W,
    const int groups) {

    const int channels_per_group = C / groups;
    const int spatial_dim = H * W;
    const int channel_spatial_dim = C * spatial_dim;

    // Use a grid-stride loop to ensure all elements are processed,
    // regardless of the number of threads launched.
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += blockDim.x * gridDim.x) {

        // Decompose the 1D output index 'idx' into 4D tensor coordinates (n, c_out, h, w)
        const int hw = idx % spatial_dim;
        const int c_out = (idx / spatial_dim) % C;
        const int n = idx / channel_spatial_dim;

        // The shuffle operation is equivalent to a transpose of group and channel-per-group dimensions.
        // Original PyTorch logic:
        // 1. view(N, G, C/G, H, W)
        // 2. transpose(1, 2) -> (N, C/G, G, H, W)
        // 3. view(N, C, H, W)
        // To find the source channel (c_in) for a given output channel (c_out), we reverse this.
        // An output channel c_out corresponds to (c_prime, g) in the (C/G, G) dimensions.
        // c_prime = c_out / groups
        // g = c_out % groups
        // Before the transpose, this was (g, c_prime) in the (G, C/G) dimensions.
        // So, the original channel c_in is g * (C/G) + c_prime.
        const int g = c_out % groups;
        const int c_prime = c_out / groups;
        const int c_in = g * channels_per_group + c_prime;

        // Calculate the 1D index for the input tensor
        const int input_idx = n * channel_spatial_dim + c_in * spatial_dim + hw;

        output[idx] = input[input_idx];
    }
}

// C++ wrapper function that launches the CUDA kernel.
// This function is what gets called from Python.
torch::Tensor channel_shuffle_cuda(torch::Tensor x, int64_t groups) {
    // Input validation checks
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4-dimensional");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    const auto batch_size = x.size(0);
    const auto channels = x.size(1);
    const auto height = x.size(2);
    const auto width = x.size(3);

    TORCH_CHECK(channels > 0 && channels % groups == 0, "Number of channels must be divisible by groups");

    auto out = torch::empty_like(x);
    const int total_elements = x.numel();

    if (total_elements == 0) {
        return out;
    }

    // Configure kernel launch parameters
    const int block_size = 1024;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    // Launch the kernel
    channel_shuffle_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        total_elements,
        channels,
        height,
        width,
        groups
    );
    
    // Check for any CUDA errors during kernel execution
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

channel_shuffle_cpp_source = (
    "torch::Tensor channel_shuffle_cuda(torch::Tensor x, int64_t groups);"
)

# Compile the inline CUDA code using PyTorch's C++ extension utilities
channel_shuffle = load_inline(
    name="channel_shuffle",
    cpp_sources=channel_shuffle_cpp_source,
    cuda_sources=channel_shuffle_source,
    functions=["channel_shuffle_cuda"],
    verbose=True,
)


class ShuffleNetUnitNew(nn.Module):
    """
    ShuffleNet unit implementation, modified to use the custom CUDA kernel
    for the channel shuffle operation.
    """
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitNew, self).__init__()
        
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Store groups for the custom shuffle operation
        self.groups = groups
        
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Use the custom CUDA kernel for channel shuffle
        out = channel_shuffle.channel_shuffle_cuda(out, self.groups)
        
        out = F.relu(self.bn3(self.conv3(out)))
        
        out += self.shortcut(x)
        return out


class ModelNew(nn.Module):
    """
    ShuffleNet architecture, optimized to use the custom ShuffleNetUnitNew.
    """
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        super(ModelNew, self).__init__()
        
        self.conv1 = nn.Conv2d(3, stages_out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)
        
        self.conv5 = nn.Conv2d(stages_out_channels[3], 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)
        
        self.fc = nn.Linear(1024, num_classes)
    
    def _make_stage(self, in_channels, out_channels, repeats, groups):
        layers = []
        # Use the new ShuffleNet unit with the custom CUDA kernel
        layers.append(ShuffleNetUnitNew(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnitNew(out_channels, out_channels, groups))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x