import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for copying a source tensor into a slice of a destination tensor.
# This optimization targets the main bottleneck in the original architecture: the repeated
# memory allocation and data copying caused by `torch.cat` inside the forward loop.
# By pre-allocating a single large output buffer and using this efficient kernel to place
# the output of each layer directly into its final position, we avoid creating multiple
# large intermediate tensors.
copy_into_slice_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void copy_into_slice_kernel(
    float* dest, 
    const float* src, 
    int num_elements,
    int B, 
    int C_dest, 
    int C_src, 
    int H, 
    int W,
    int channel_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        // Since the source tensor is contiguous, its flattened index is `idx`.
        // We need to calculate the corresponding destination index based on logical dimensions.

        // De-flatten the source index `idx` to get logical coordinates (b, c_src, h, w)
        int b = idx / (C_src * H * W);
        int remainder = idx % (C_src * H * W);
        int c_src = remainder / (H * W);
        remainder = remainder % (H * W);
        int h = remainder / W;
        int w = remainder % W;

        // Calculate the destination channel by adding the offset
        int c_dest = c_src + channel_offset;

        // Re-flatten the logical coordinates to get the destination index
        // Stride for batch in dest = C_dest * H * W
        // Stride for channel in dest = H * W
        // Stride for height in dest = W
        long dest_idx = (long)b * C_dest * H * W + 
                        (long)c_dest * H * W + 
                        (long)h * W + 
                        w;
        
        dest[dest_idx] = src[idx];
    }
}

// C++ wrapper function that will be callable from Python
void copy_into_slice_cuda(
    torch::Tensor destination,
    torch::Tensor source,
    int64_t channel_offset
) {
    // Ensure tensors are on the GPU and have the expected properties
    TORCH_CHECK(destination.is_cuda(), "Destination tensor must be a CUDA tensor");
    TORCH_CHECK(source.is_cuda(), "Source tensor must be a CUDA tensor");
    TORCH_CHECK(destination.is_contiguous(), "Destination tensor must be contiguous");
    TORCH_CHECK(source.is_contiguous(), "Source tensor must be contiguous");
    TORCH_CHECK(destination.scalar_type() == torch::kFloat32, "Destination must be a float32 tensor");
    TORCH_CHECK(source.scalar_type() == torch::kFloat32, "Source must be a float32 tensor");

    // Get dimensions for kernel launch and indexing
    const auto batch_size = source.size(0);
    const auto source_channels = source.size(1);
    const auto height = source.size(2);
    const auto width = source.size(3);
    const auto dest_channels = destination.size(1);

    const int num_elements = source.numel();
    if (num_elements == 0) {
        return;
    }

    // Kernel launch configuration
    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    // Launch the CUDA kernel
    copy_into_slice_kernel<<<num_blocks, block_size>>>(
        destination.data_ptr<float>(),
        source.data_ptr<float>(),
        num_elements,
        batch_size,
        dest_channels,
        source_channels,
        height,
        width,
        channel_offset
    );
    
    // Check for any errors during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
"""

copy_into_slice_cpp_source = (
    "void copy_into_slice_cuda(torch::Tensor destination, torch::Tensor source, int64_t channel_offset);"
)

# Compile the inline CUDA code using torch's C++ extension utilities.
# This is done at the module level to avoid recompilation on every model instantiation.
copy_into_slice = load_inline(
    name="copy_into_slice",
    cpp_sources=copy_into_slice_cpp_source,
    cuda_sources=copy_into_slice_source,
    functions=["copy_into_slice_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        :param num_layers: The number of layers in the dense block
        :param num_input_features: The number of input feature maps
        :param growth_rate: The growth rate for the dense block (new features added per layer)
        """
        super(ModelNew, self).__init__()
        self.num_layers = num_layers
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate

        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

        # Assign the compiled custom CUDA function to the model instance
        self.copy_op = copy_into_slice

    def _make_layer(self, in_features: int, growth_rate: int):
        """
        Creates a single layer with BatchNorm, ReLU, Conv2D, and Dropout.
        """
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )
    
    def forward(self, x):
        """
        Optimized forward pass that avoids repeated `torch.cat` calls.
        It pre-allocates the final output tensor and uses a custom CUDA kernel
        to copy the output of each layer into the correct slice of the buffer.

        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Concatenated output tensor with shape (batch_size, num_output_features, height, width)
        """
        batch_size, _, height, width = x.shape
        
        # 1. Pre-allocate the full output buffer once at the beginning.
        final_channels = self.num_input_features + self.num_layers * self.growth_rate
        out_buffer = torch.empty(
            (batch_size, final_channels, height, width),
            dtype=x.dtype,
            device=x.device,
            memory_format=torch.contiguous_format
        )

        # 2. Copy the initial input features into the start of the buffer.
        out_buffer[:, :self.num_input_features, :, :] = x

        # 3. Iteratively compute features and place them in the buffer.
        for i, layer in enumerate(self.layers):
            # a. Define the input for the current layer as a memory-efficient view into the buffer.
            current_channels = self.num_input_features + i * self.growth_rate
            layer_input = out_buffer.narrow(1, 0, current_channels)

            # b. Compute the new feature map using the standard PyTorch layer.
            new_feature = layer(layer_input)

            # c. Use the custom CUDA kernel to copy the new feature into its designated slice.
            # This is an in-place operation on out_buffer.
            self.copy_op.copy_into_slice_cuda(out_buffer, new_feature, current_channels)
        
        return out_buffer