import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D Average Pooling
avg_pool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm> // For std::min

__global__ void avg_pool3d_kernel(
    const float* input,
    float* output,
    const long total_output_elements,
    const int N, const int C, const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int K, const int S, const int P) {

    // Using a 1D grid-stride loop for flexibility and to handle any number of output elements.
    for (long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_output_elements;
         idx += blockDim.x * gridDim.x) {

        // Map linear index to 5D output coordinates
        int w_out = idx % W_out;
        int h_out = (idx / W_out) % H_out;
        int d_out = (idx / (W_out * H_out)) % D_out;
        int c = (idx / (W_out * H_out * D_out)) % C;
        int n = idx / (W_out * H_out * D_out * C);

        float sum = 0.0f;
        // The divisor is constant for default AvgPool3d (count_include_pad=True)
        const float divisor = (float)(K * K * K);

        // Loop over the kernel window
        for (int kd = 0; kd < K; ++kd) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    // Calculate corresponding input coordinates
                    int d_in = d_out * S - P + kd;
                    int h_in = h_out * S - P + kh;
                    int w_in = w_out * S - P + kw;

                    // Check if the coordinate is within the valid input bounds (not in padding)
                    if (d_in >= 0 && d_in < D_in &&
                        h_in >= 0 && h_in < H_in &&
                        w_in >= 0 && w_in < W_in) {
                        
                        // Calculate linear input index
                        long input_idx = (long)n * C * D_in * H_in * W_in +
                                         (long)c * D_in * H_in * W_in +
                                         (long)d_in * H_in * W_in +
                                         (long)h_in * W_in +
                                         w_in;
                        sum += input[input_idx];
                    }
                    // Padded values are implicitly 0 and do not contribute to the sum.
                }
            }
        }
        
        output[idx] = sum / divisor;
    }
}

torch::Tensor avg_pool3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(input.dim() == 5, "Input must be a 5D tensor, but got ", input.dim(), "D");

    const int N = input.size(0);
    const int C = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    const int K = kernel_size;
    const int S = stride;
    const int P = padding;

    // Calculate output dimensions based on PyTorch's formula
    const int D_out = (D_in + 2 * P - K) / S + 1;
    const int H_out = (H_in + 2 * P - K) / S + 1;
    const int W_out = (W_in + 2 * P - K) / S + 1;

    TORCH_CHECK(D_out > 0 && H_out > 0 && W_out > 0, "Output dimensions must be positive");

    auto output = torch::zeros({N, C, D_out, H_out, W_out}, input.options());

    const long total_output_elements = (long)N * C * D_out * H_out * W_out;
    if (total_output_elements == 0) {
        return output;
    }

    const int block_size = 256;
    // Use a reasonable number of blocks; the grid-stride loop will handle the rest.
    const int num_blocks = std::min((int)((total_output_elements + block_size - 1) / block_size), 4096);

    avg_pool3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total_output_elements,
        N, C, D_in, H_in, W_in,
        D_out, H_out, W_out,
        K, S, P
    );
    
    // Check for any CUDA errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

avg_pool3d_cpp_source = (
    "torch::Tensor avg_pool3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);"
)

# Compile the inline CUDA code for 3D Average Pooling
avg_pool3d_custom = load_inline(
    name="avg_pool3d_custom",
    cpp_sources=avg_pool3d_cpp_source,
    cuda_sources=avg_pool3d_source,
    functions=["avg_pool3d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Simple model that performs 3D Average Pooling using a custom CUDA kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the custom Average Pooling layer.

        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which uses the kernel size.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        # PyTorch's AvgPool3d defaults stride to kernel_size if not provided.
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.avg_pool_custom = avg_pool3d_custom

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies custom Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch::Tensor: Output tensor with Average Pooling applied.
        """
        return self.avg_pool_custom.avg_pool3d_cuda(x, self.kernel_size, self.stride, self.padding)

batch_size = 16
channels = 32
depth = 128
height = 128
width = 256
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    x = torch.rand(batch_size, channels, depth, height, width).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]