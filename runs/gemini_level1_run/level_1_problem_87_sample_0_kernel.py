import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation using a custom CUDA kernel.
    This implementation is optimized for 1x1 convolutions by treating the operation
    as a batched matrix-vector multiplication, where each spatial location (h, w)
    is an independent problem. A custom CUDA kernel is used to exploit this structure,
    using shared memory to cache the input channel vector for each spatial location,
    thereby reducing global memory bandwidth requirements.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Define weight and bias as nn.Parameter, same as in nn.Conv2d
        self.weight = Parameter(torch.empty((out_channels, in_channels, 1, 1)))
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        # JIT compile the custom CUDA kernel when the model is instantiated
        self._load_kernel()

    def reset_parameters(self) -> None:
        """
        Initialize weights and bias to match the default behavior of nn.Conv2d.
        """
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def _load_kernel(self):
        """
        Defines and JIT compiles the C++/CUDA source code for the custom operator.
        """
        # The combined C++/CUDA source code for the custom pointwise convolution
        cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

// CUDA Kernel for Pointwise (1x1) Convolution
// Each thread block processes one spatial location (n, h, w).
// Shared memory is used to cache the input vector of size C_in for that location.
// Each thread in the block computes a subset of the output channels.
__global__ void pointwise_conv2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int C_in, int C_out, int H, int W
) {
    // Dynamically allocated shared memory for the input channel vector
    extern __shared__ float x_shared[];

    // Identify the spatial location (n, h, w) this block is responsible for
    int spatial_idx = blockIdx.x;

    // Decompose the 1D spatial index into 3D coordinates (n, h, w)
    const int HW = H * W;
    const int n = spatial_idx / HW;
    const int hw = spatial_idx % HW;
    const int h = hw / W;
    const int w = hw % W;

    // Calculate base pointers for strided access into the input and output tensors
    const float* x_ptr = x + n * C_in * HW + h * W + w;
    float* y_ptr = y + n * C_out * HW + h * W + w;

    // Collaboratively load the input vector x[n, :, h, w] into shared memory.
    // This is a strided load from global memory.
    for (int i = threadIdx.x; i < C_in; i += blockDim.x) {
        x_shared[i] = x_ptr[i * HW];
    }
    __syncthreads(); // Synchronize to ensure all data is loaded before computation

    // Each thread computes a subset of the output channels in a grid-stride loop
    for (int c_out = threadIdx.x; c_out < C_out; c_out += blockDim.x) {
        float sum = 0.0f;
        const float* weight_row_ptr = weight + c_out * C_in;

        // Compute the dot product using the cached input vector from shared memory
        for (int c_in = 0; c_in < C_in; ++c_in) {
            sum += x_shared[c_in] * weight_row_ptr[c_in];
        }

        // Add bias if it is provided
        if (bias != nullptr) {
            sum += bias[c_out];
        }

        // Write the final result. This is a strided write to global memory.
        y_ptr[c_out * HW] = sum;
    }
}

// C++ function to launch the CUDA kernel
void pointwise_conv2d_kernel_launcher(
    const float* x, const float* weight, const float* bias, float* y,
    int N, int C_in, int C_out, int H, int W)
{
    const int total_spatial_locations = N * H * W;
    
    // Heuristic for block size. 256 is generally a good starting point.
    const int block_size = 256;
    const int grid_size = total_spatial_locations;
    const size_t shared_mem_size = C_in * sizeof(float);

    // Launch the kernel
    pointwise_conv2d_kernel<<<grid_size, block_size, shared_mem_size>>>(
        x, weight, bias, y, C_in, C_out, H, W
    );

    // Check for kernel launch errors for easier debugging
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

// The main C++ interface function that PyTorch will call
torch::Tensor pointwise_conv2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    // Input validation checks
    TORCH_CHECK(x.is_cuda(), "Input tensor x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(torch::MemoryFormat::Contiguous), "Input tensor x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(torch::MemoryFormat::Contiguous), "Weight tensor must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor x must be of type float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "Weight tensor must be of type float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor x must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 2, "Weight tensor must be 2D (C_out, C_in)");

    // Get tensor dimensions
    const int N = x.size(0);
    const int C_in = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int C_out = weight.size(0);

    TORCH_CHECK(C_in == weight.size(1), "Mismatch: x.size(1) and weight.size(1) must be equal");

    // Handle optional bias tensor
    bool has_bias = bias.defined() && bias.numel() > 0;
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "Bias tensor must be a CUDA tensor");
        TORCH_CHECK(bias.is_contiguous(torch::MemoryFormat::Contiguous), "Bias tensor must be contiguous");
        TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "Bias tensor must be of type float32");
        TORCH_CHECK(bias.dim() == 1, "Bias tensor must be 1D");
        TORCH_CHECK(bias.size(0) == C_out, "Bias size must match output channels");
    }

    // Create the output tensor with the correct shape and device
    auto y = torch::empty({N, C_out, H, W}, x.options());

    // Call the launcher function to execute the kernel
    pointwise_conv2d_kernel_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        y.data_ptr<float>(),
        N, C_in, C_out, H, W
    );

    return y;
}
        """

        # Define the C++ function signature for the PyTorch binding
        cpp_source = "torch::Tensor pointwise_conv2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);"

        # Use load_inline to JIT compile the C++/CUDA code
        self.pointwise_conv_op = load_inline(
            name="pointwise_conv_op_v2", # Use a unique name to avoid cache conflicts
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=["pointwise_conv2d_cuda"],
            verbose=False, # Set to True for debugging compilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # The weight is stored as (C_out, C_in, 1, 1) to be compatible with nn.Module state_dict.
        # We squeeze it to (C_out, C_in) before passing it to the CUDA kernel.
        squeezed_weight = self.weight.squeeze()
        return self.pointwise_conv_op.pointwise_conv2d_cuda(x, squeezed_weight, self.bias)