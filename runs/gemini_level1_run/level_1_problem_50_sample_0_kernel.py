import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for 2D convolution
conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Naive direct convolution implementation for the forward pass
__global__ void conv2d_forward_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int S, int P) {

    // Each thread computes one output element in the feature map.
    // The grid is 3D: (W_out, H_out, N * C_out)
    // The block is 2D: (threads_x, threads_y)

    // Calculate the output (x, y) coordinates for this thread
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Get the batch and output channel index from the z-dimension of the grid
    int c_out = blockIdx.z % C_out;
    int n = blockIdx.z / C_out;

    // Check if the thread is within the output bounds to avoid out-of-bounds access
    if (h_out < H_out && w_out < W_out && n < N) {
        float acc = 0.0f;

        // Iterate over input channels
        for (int c_in = 0; c_in < C_in; ++c_in) {
            // Iterate over kernel height and width
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    // Calculate corresponding input coordinates based on stride and padding
                    int h_in = h_out * S - P + kh;
                    int w_in = w_out * S - P + kw;

                    // Apply padding: only accumulate if the input coordinates are valid (within the input tensor)
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        // Calculate flat indices for NCHW tensor format
                        int input_idx = n * C_in * H_in * W_in +
                                        c_in * H_in * W_in +
                                        h_in * W_in +
                                        w_in;
                        int weight_idx = c_out * C_in * K * K +
                                         c_in * K * K +
                                         kh * K +
                                         kw;
                        
                        acc += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        // Add bias for the current output channel
        acc += bias[c_out];

        // Calculate flat index for the output tensor and write the result
        int output_idx = n * C_out * H_out * W_out +
                         c_out * H_out * W_out +
                         h_out * W_out +
                         w_out;
        output[output_idx] = acc;
    }
}

// C++ wrapper function that launches the CUDA kernel
torch::Tensor conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding) {

    // Ensure tensors are on CUDA and contiguous for direct memory access
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    
    input = input.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    // Get tensor dimensions from the input and weight tensors
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    const int C_out = weight.size(0);
    const int K = weight.size(2); // Assuming kernel is square (KxK)

    // Calculate output dimensions based on the standard convolution formula
    const int H_out = (H_in + 2 * padding - K) / stride + 1;
    const int W_out = (W_in + 2 * padding - K) / stride + 1;

    // Create an output tensor of the correct size, on the same device as the input
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    // Define CUDA grid and block dimensions
    // Use a 2D block for spatial dimensions (H_out, W_out)
    // Use a 3D grid to cover all output elements (W_out, H_out, N*C_out)
    dim3 threads_per_block(16, 16);
    dim3 num_blocks(
        (W_out + threads_per_block.x - 1) / threads_per_block.x,
        (H_out + threads_per_block.y - 1) / threads_per_block.y,
        N * C_out
    );

    // Launch the kernel
    conv2d_forward_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K, stride, padding
    );
    
    // Check for any errors during kernel execution for easier debugging
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

# C++ source for the function signature, required by load_inline
conv2d_cpp_source = "torch::Tensor conv2d_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding);"

# Compile the inline CUDA code. This is done once when the module is imported.
custom_conv2d = load_inline(
    name="custom_conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        # Parameters from the original model's Conv2d layer
        self.in_channels = 3
        self.out_channels = 96
        self.kernel_size = 11
        self.stride = 4
        self.padding = 2
        
        # Create weight and bias as nn.Parameter to be managed by PyTorch.
        # This ensures they are included in model.parameters() for training.
        self.weight = nn.Parameter(torch.Tensor(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        ))
        self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        
        # Initialize parameters using a standard method, similar to nn.Conv2d
        self.reset_parameters()

        # Store the compiled custom convolution function
        self.custom_conv2d_op = custom_conv2d.conv2d_forward_cuda

    def reset_parameters(self) -> None:
        # Kaiming uniform initialization, which is PyTorch's default for Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Call the custom CUDA kernel for the convolution operation
        return self.custom_conv2d_op(x, self.weight, self.bias, self.stride, self.padding)