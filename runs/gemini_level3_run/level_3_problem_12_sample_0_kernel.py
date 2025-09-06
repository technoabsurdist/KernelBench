import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA source code for fused and custom operators
vgg_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h> // For FLT_MIN

// --- Kernel 1: Generic In-place ReLU ---
// This kernel can be applied to any tensor to perform ReLU in-place.
__global__ void relu_inplace_kernel(float* data, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        data[index] = fmaxf(0.0f, data[index]);
    }
}

// --- Kernel 2: Max Pooling ---
// A custom kernel for 2D max pooling.
__global__ void max_pool2d_kernel(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w) {

    // Each thread computes one output element
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = N * C * H_out * W_out;

    if (index < total_output_elements) {
        // Decompose the 1D output index into 4D coordinates (n, c, h_out, w_out)
        int w_out = index % W_out;
        int h_out = (index / W_out) % H_out;
        int c = (index / (W_out * H_out)) % C;
        int n = index / (C * W_out * H_out);

        // Calculate the top-left corner of the pooling window in the input tensor
        int h_start = h_out * stride_h - pad_h;
        int w_start = w_out * stride_w - pad_w;

        float max_val = -FLT_MAX;

        // Iterate over the pooling window
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                int h = h_start + i;
                int w = w_start + j;

                // Check if the current position is within the input bounds (not in padding)
                if (h >= 0 && h < H && w >= 0 && w < W) {
                    int input_idx = n * C * H * W + c * H * W + h * W + w;
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }
        output[index] = max_val;
    }
}

// --- C++ Wrapper 1: Conv2d + ReLU Fusion ---
// This function performs a standard convolution using PyTorch's backend (cuDNN)
// and then immediately applies our custom in-place ReLU kernel. This reduces
// memory bandwidth by avoiding a separate read/write for the ReLU operation.
torch::Tensor conv2d_relu(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef dilation,
    int64_t groups) {

    // Use torch's highly optimized conv2d implementation
    auto output = torch::conv2d(input, weight, bias, stride, padding, dilation, groups);

    // Apply ReLU in-place using our generic kernel
    int total_elements = output.numel();
    const int block_size = 1024;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    relu_inplace_kernel<<<num_blocks, block_size>>>(output.data_ptr<float>(), total_elements);
    
    // Check for errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}

// --- C++ Wrapper 2: Custom Max Pooling ---
// This function launches our custom max_pool2d_kernel.
torch::Tensor max_pool2d_custom(
    const torch::Tensor& input,
    torch::IntArrayRef kernel_size,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef dilation) { // Dilation is ignored for simplicity but kept for API consistency

    // Input dimensions
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // Kernel, stride, and padding dimensions
    int kernel_h = kernel_size[0];
    int kernel_w = kernel_size[1];
    int stride_h = stride[0];
    int stride_w = stride[1];
    int pad_h = padding[0];
    int pad_w = padding[1];

    // Calculate output dimensions
    int H_out = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    int W_out = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    // Create the output tensor
    auto output = torch::empty({N, C, H_out, W_out}, input.options());
    int total_output_elements = output.numel();

    const int block_size = 1024;
    const int num_blocks = (total_output_elements + block_size - 1) / block_size;

    max_pool2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, H, W, H_out, W_out,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}

// --- C++ Wrapper 3: Linear + ReLU Fusion ---
// Similar to conv2d_relu, this uses PyTorch's optimized linear operation (cuBLAS)
// and then applies our custom in-place ReLU kernel.
torch::Tensor linear_relu(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {
    
    // Use torch's highly optimized linear implementation
    auto output = torch::linear(input, weight, bias);

    // Apply ReLU in-place using our generic kernel
    int total_elements = output.numel();
    const int block_size = 1024;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    relu_inplace_kernel<<<num_blocks, block_size>>>(output.data_ptr<float>(), total_elements);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

# Define the C++ source for function declarations
vgg_cpp_source = """
torch::Tensor conv2d_relu(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
    torch::IntArrayRef stride, torch::IntArrayRef padding, torch::IntArrayRef dilation,
    int64_t groups);

torch::Tensor max_pool2d_custom(
    const torch::Tensor& input, torch::IntArrayRef kernel_size, torch::IntArrayRef stride,
    torch::IntArrayRef padding, torch::IntArrayRef dilation);

torch::Tensor linear_relu(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias);
"""

# Compile the inline CUDA code using torch's C++ extension loader
fused_vgg_ops = load_inline(
    name="fused_vgg_ops",
    cpp_sources=vgg_cpp_source,
    cuda_sources=vgg_cuda_source,
    functions=["conv2d_relu", "max_pool2d_custom", "linear_relu"],
    verbose=False,
)

# --- Custom PyTorch Modules using the compiled CUDA operators ---

class FusedConv2dReLU(nn.Module):
    """A custom module that fuses Conv2d and ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True):
        super().__init__()
        # We use a standard Conv2d layer to store weights and biases.
        # This makes it compatible with standard PyTorch state_dict loading.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        
    def forward(self, x):
        # Call our custom C++/CUDA function
        return fused_vgg_ops.conv2d_relu(
            x, self.conv.weight, self.conv.bias,
            self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups
        )

class CustomMaxPool2d(nn.Module):
    """A custom module for MaxPool2d using our own kernel."""
    def __init__(self, kernel_size, stride, padding=0):
        super().__init__()
        # Store parameters for the forward pass
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (1, 1) # Dilation is unused in kernel but kept for API consistency

    def forward(self, x):
        # Call our custom C++/CUDA function
        return fused_vgg_ops.max_pool2d_custom(
            x, self.kernel_size, self.stride, self.padding, self.dilation
        )

class FusedLinearReLU(nn.Module):
    """A custom module that fuses Linear and ReLU."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # Use a standard Linear layer to store weights and biases
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        # Call our custom C++/CUDA function
        return fused_vgg_ops.linear_relu(x, self.linear.weight, self.linear.bias)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Initialize the VGG19 model with custom/fused CUDA operators.
        """
        super(ModelNew, self).__init__()
        
        # VGG19 architecture with Conv+ReLU layers fused and MaxPool replaced
        self.features = nn.Sequential(
            # Block 1
            FusedConv2dReLU(3, 64, kernel_size=3, padding=1),
            FusedConv2dReLU(64, 64, kernel_size=3, padding=1),
            CustomMaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            FusedConv2dReLU(64, 128, kernel_size=3, padding=1),
            FusedConv2dReLU(128, 128, kernel_size=3, padding=1),
            CustomMaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            FusedConv2dReLU(128, 256, kernel_size=3, padding=1),
            FusedConv2dReLU(256, 256, kernel_size=3, padding=1),
            FusedConv2dReLU(256, 256, kernel_size=3, padding=1),
            FusedConv2dReLU(256, 256, kernel_size=3, padding=1),
            CustomMaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            FusedConv2dReLU(256, 512, kernel_size=3, padding=1),
            FusedConv2dReLU(512, 512, kernel_size=3, padding=1),
            FusedConv2dReLU(512, 512, kernel_size=3, padding=1),
            FusedConv2dReLU(512, 512, kernel_size=3, padding=1),
            CustomMaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            FusedConv2dReLU(512, 512, kernel_size=3, padding=1),
            FusedConv2dReLU(512, 512, kernel_size=3, padding=1),
            FusedConv2dReLU(512, 512, kernel_size=3, padding=1),
            FusedConv2dReLU(512, 512, kernel_size=3, padding=1),
            CustomMaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classifier with Linear+ReLU layers fused
        self.classifier = nn.Sequential(
            FusedLinearReLU(512 * 7 * 7, 4096),
            nn.Dropout(p=0.0),
            FusedLinearReLU(4096, 4096),
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes) # Final layer has no ReLU, so we use the standard nn.Linear
        )
    
    def forward(self, x):
        """
        Forward pass of the optimized VGG19 model.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x