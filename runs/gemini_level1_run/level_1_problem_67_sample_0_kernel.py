import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for im2col operation for 1D convolution
# This kernel transforms the input tensor into a column matrix,
# which allows the convolution to be performed as a single matrix multiplication (GEMM).
# This is a standard and highly efficient way to implement convolutions on GPUs.
conv1d_im2col_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel to perform the im2col transformation for Conv1d
// It unnests the sliding windows of the input tensor into columns of a new matrix.
// This uses a grid-stride loop to ensure all elements are processed regardless of grid size.
__global__ void im2col1d_kernel(
    const float* data_im, // Input tensor data of shape (N, C_in, L_in)
    const int N, const int C_in, const int L_in,
    const int K, const int P, const int S, const int D,
    const int L_out,
    float* data_col // Output tensor data of shape (N, C_in * K, L_out)
) {
    // Total number of elements in the output 'col' matrix
    const int num_elements = N * C_in * K * L_out;
    // Grid-stride loop
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < num_elements;
         index += blockDim.x * gridDim.x) {

        // Decompose the 1D thread index into the 4D coordinate of the conceptual output matrix (N, C_in, K, L_out)
        const int l_out = index % L_out;
        const int k = (index / L_out) % K;
        const int c_in = (index / (L_out * K)) % C_in;
        const int n = index / (L_out * K * C_in);

        // Calculate the corresponding input's length coordinate based on convolution parameters
        const int l_in = l_out * S - P + k * D;

        // The flattened index in the output 'col' matrix is just the loop index
        const int col_index = index;
        // The flattened index in the input 'im' matrix
        const int im_index = n * (C_in * L_in) + c_in * L_in + l_in;

        // Check if the calculated input coordinate is within bounds (not in the padded area)
        if (l_in >= 0 && l_in < L_in) {
            data_col[col_index] = data_im[im_index];
        } else {
            // If out of bounds, this corresponds to a padded value, which is 0.
            data_col[col_index] = 0.0f;
        }
    }
}

// C++ wrapper function that will be called from Python
torch::Tensor im2col1d_cuda(
    torch::Tensor input,
    int kernel_size, int stride, int padding, int dilation) {

    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "Input must be a 3D tensor (N, C, L)");
    input = input.contiguous(); // Ensure tensor is contiguous in memory

    // Get input dimensions
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int L_in = input.size(2);

    // Calculate output length using the standard formula for convolution
    const int L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Output length must be positive. Check convolution parameters.");

    // Create the output tensor for the 'col' matrix
    auto col = torch::zeros({N, C_in * kernel_size, L_out}, input.options());

    // Set up grid and block dimensions for the CUDA kernel launch
    const int num_elements = N * C_in * kernel_size * L_out;
    const int block_size = 256;
    // Calculate number of blocks, ensuring it's at least 1.
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    // Launch the im2col kernel
    im2col1d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        N, C_in, L_in,
        kernel_size, padding, stride, dilation,
        L_out,
        col.data_ptr<float>()
    );

    // Check for any CUDA errors during kernel execution
    C10_CUDA_CHECK(cudaGetLastError());

    return col;
}
"""

# C++ source for the function signature
conv1d_im2col_cpp_source = """
torch::Tensor im2col1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);
"""

# Compile the inline CUDA/C++ code using PyTorch's JIT compiler
im2col_op = load_inline(
    name="im2col_op",
    cpp_sources=conv1d_im2col_cpp_source,
    cuda_sources=conv1d_im2col_source,
    functions=["im2col1d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Performs a standard 1D convolution by lowering the operation to a matrix
    multiplication (GEMM) using a custom CUDA `im2col` kernel. This implementation
    is functionally equivalent to nn.Conv1d for groups=1.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections. Must be 1 for this implementation.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()

        if groups != 1:
            raise NotImplementedError("Custom Conv1D kernel only supports groups=1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Define learnable weight and bias parameters, same as nn.Conv1d
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        # Store the compiled custom operator
        self.im2col_op = im2col_op

    def reset_parameters(self) -> None:
        # Initialize weights and bias using the same method as nn.Conv1d for consistency
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using the im2col + GEMM strategy.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        N, C_in, L_in = x.shape

        # Step 1: Use the custom CUDA kernel to transform the input tensor into a column matrix.
        # The shape of 'col' will be (N, C_in * kernel_size, L_out).
        col = self.im2col_op.im2col1d_cuda(x, self.kernel_size, self.stride, self.padding, self.dilation)
        
        # Step 2: Reshape the weight tensor for matrix multiplication.
        # Original weight shape: (C_out, C_in, K)
        # Reshaped weight shape: (C_out, C_in * K)
        weight_gemm = self.weight.view(self.out_channels, -1)

        # Step 3: Reshape the 'col' matrix for the main GEMM operation.
        # The goal is to have a shape of (C_in * K, N * L_out).
        # Original col shape: (N, C_in * K, L_out)
        # Permuted shape:     (C_in * K, L_out, N)
        # Reshaped shape:     (C_in * K, L_out * N)
        L_out = col.shape[2]
        col_gemm = col.permute(1, 2, 0).reshape(C_in * self.kernel_size, N * L_out)

        # Step 4: Perform the matrix multiplication (the core of the convolution).
        # (C_out, C_in * K) @ (C_in * K, N * L_out) -> (C_out, N * L_out)
        output = torch.matmul(weight_gemm, col_gemm)

        # Step 5: Reshape the output of the GEMM back to the desired output tensor shape.
        # Original output shape: (C_out, N * L_out)
        # Reshaped shape:      (C_out, N, L_out)
        # Permuted shape:      (N, C_out, L_out)
        output = output.view(self.out_channels, N, L_out).permute(1, 0, 2)

        # Step 6: Add the bias if it exists.
        if self.bias is not None:
            # Reshape bias to (1, C_out, 1) for broadcasting
            output += self.bias.view(1, -1, 1)

        return output