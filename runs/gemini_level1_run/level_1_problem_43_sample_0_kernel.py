import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D Max Pooling
maxpool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h> // For FLT_MAX

__global__ void maxpool3d_forward_kernel(
    const float* input_data,
    float* output_data,
    const int N, const int C, const int iD, const int iH, const int iW,
    const int oD, const int oH, const int oW,
    const int kD, const int kH, const int kW,
    const int sD, const int sH, const int sW,
    const int pD, const int pH, const int pW,
    const int dD, const int dH, const int dW) {

    // Using a grid-stride loop to ensure all output elements are processed
    // regardless of the number of blocks launched.
    for (long i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < (long)N * C * oD * oH * oW; 
         i += gridDim.x * blockDim.x) {
        
        // Decompose the linear index to get n, c, od, oh, ow
        const int ow = i % oW;
        const int oh = (i / oW) % oH;
        const int od = (i / (oW * oH)) % oD;
        const int c = (i / (oW * oH * oD)) % C;
        const int n = i / (oW * oH * oD * C);

        // Initialize max value to the smallest possible float
        float max_val = -FLT_MAX;

        // Loop over the kernel window
        for (int kd = 0; kd < kD; ++kd) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    // Calculate input coordinates
                    const int id = od * sD - pD + kd * dD;
                    const int ih = oh * sH - pH + kh * dH;
                    const int iw = ow * sW - pW + kw * dW;

                    // Check if the coordinates are within the input bounds (not in padding)
                    if (id >= 0 && id < iD && ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
                        // Calculate the linear index for the input tensor
                        const long input_idx = (long)n * C * iD * iH * iW +
                                               (long)c * iD * iH * iW +
                                               (long)id * iH * iW +
                                               (long)ih * iW +
                                               iw;
                        // Update max value
                        max_val = fmaxf(max_val, input_data[input_idx]);
                    }
                }
            }
        }
        // Write the max value to the output tensor. If the window was entirely in padding,
        // the value will be -FLT_MAX, which matches PyTorch's behavior.
        output_data[i] = max_val;
    }
}

torch::Tensor maxpool3d_forward_cuda(
    torch::Tensor input,
    int kD, int kH, int kW,
    int sD, int sH, int sW,
    int pD, int pH, int pW,
    int dD, int dH, int dW) {

    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(input.dim() == 5, "Input must be a 5D tensor (N, C, D, H, W)");

    // Get input dimensions
    const int N = input.size(0);
    const int C = input.size(1);
    const int iD = input.size(2);
    const int iH = input.size(3);
    const int iW = input.size(4);

    // Calculate output dimensions (floor mode)
    const int oD = (iD + 2 * pD - dD * (kD - 1) - 1) / sD + 1;
    const int oH = (iH + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    const int oW = (iW + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
    
    TORCH_CHECK(oD > 0 && oH > 0 && oW > 0, "Output size is invalid. Check kernel, stride, and padding parameters.");

    // Create output tensor
    auto output = torch::empty({N, C, oD, oH, oW}, input.options());

    // Kernel launch configuration
    const long total_outputs = (long)N * C * oD * oH * oW;
    if (total_outputs == 0) {
        return output;
    }
    
    const int block_size = 256;
    // Use a heuristic for the number of blocks, can be tuned
    const int num_blocks = std::min((int)((total_outputs + block_size - 1) / block_size), 4096);

    // Launch the kernel
    maxpool3d_forward_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, iD, iH, iW,
        oD, oH, oW,
        kD, kH, kW,
        sD, sH, sW,
        pD, pH, pW,
        dD, dH, dW
    );
    
    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA error in maxpool3d_forward_kernel: ", cudaGetErrorString(err));
    }

    return output;
}
"""

maxpool3d_cpp_source = """
torch::Tensor maxpool3d_forward_cuda(
    torch::Tensor input,
    int kD, int kH, int kW,
    int sD, int sH, int sW,
    int pD, int pH, int pW,
    int dD, int dH, int dW);
"""

# JIT compile the custom CUDA kernel
maxpool3d_op = load_inline(
    name="maxpool3d_op",
    cpp_sources=maxpool3d_cpp_source,
    cuda_sources=maxpool3d_source,
    functions=["maxpool3d_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 3D using a custom CUDA kernel.
    """
    def __init__(self, kernel_size, stride = None, padding = 0, dilation = 1, return_indices: bool = False, ceil_mode: bool = False):
        """
        Initializes the custom Max Pooling 3D layer.

        Args:
            kernel_size (int or tuple): Size of the kernel for the max pooling operation.
            stride (int or tuple, optional): Stride of the pooling operation. Defaults to kernel_size.
            padding (int or tuple, optional): Padding applied to the input tensor. Defaults to 0.
            dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Not supported.
            ceil_mode (bool, optional): Not supported.
        """
        super(ModelNew, self).__init__()
        
        if return_indices or ceil_mode:
            raise NotImplementedError("Custom MaxPool3D kernel does not support `return_indices` or `ceil_mode`.")

        self.kernel_size = self._to_tuple(kernel_size)
        self.stride = self._to_tuple(stride if stride is not None else kernel_size)
        self.padding = self._to_tuple(padding)
        self.dilation = self._to_tuple(dilation)

    def _to_tuple(self, val):
        if isinstance(val, int):
            return (val, val, val)
        if isinstance(val, (tuple, list)) and len(val) == 3:
            return tuple(val)
        raise ValueError("Parameter must be an int or a 3-element tuple/list.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies custom Max Pooling 3D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 3D applied.
        """
        return maxpool3d_op.maxpool3d_forward_cuda(
            x,
            self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.dilation[0], self.dilation[1], self.dilation[2]
        )