import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + Bias + ReLU
fused_gemm_bias_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused kernel for GEMM (x @ w.T) + bias + ReLU
// This kernel computes `out = relu(x @ w.T + b)`
// x: input tensor with shape (M, K) or (batch_size, in_features)
// w: weight tensor with shape (N, K) or (out_features, in_features)
// b: bias tensor with shape (N) or (out_features,)
// out: output tensor with shape (M, N) or (batch_size, out_features)
__global__ void fused_gemm_bias_relu_kernel(
    const float* x,
    const float* w,
    const float* b,
    float* out,
    int M,
    int N,
    int K) {

    // Using a grid-stride loop to ensure all output elements are processed.
    // Each thread is responsible for computing one element of the output matrix.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < M * N; i += stride) {
        // Map the 1D index 'i' to 2D indices (row, col) of the output matrix
        int row = i / N;
        int col = i % N;

        // --- Start of GEMM part ---
        // Compute the dot product for one output element
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            // x is indexed by [row, k]
            // w is indexed by [col, k] (since we are computing x @ w.T)
            acc += x[row * K + k] * w[col * K + k];
        }

        // --- Start of Bias Addition part ---
        acc += b[col];

        // --- Start of ReLU part ---
        out[i] = fmaxf(0.0f, acc);
    }
}

// C++ wrapper function that launches the CUDA kernel
torch::Tensor fused_gemm_bias_relu_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b) {

    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor x must be on CUDA");
    TORCH_CHECK(w.is_cuda(), "Weight tensor w must be on CUDA");
    TORCH_CHECK(b.is_cuda(), "Bias tensor b must be on CUDA");
    TORCH_CHECK(x.dim() == 2, "Input tensor x must be 2D");
    TORCH_CHECK(w.dim() == 2, "Weight tensor w must be 2D");
    TORCH_CHECK(b.dim() == 1, "Bias tensor b must be 1D");

    const int M = x.size(0); // batch_size
    const int K = x.size(1); // in_features
    const int N = w.size(0); // out_features

    TORCH_CHECK(w.size(1) == K, "Weight tensor w has incompatible shape");
    TORCH_CHECK(b.size(0) == N, "Bias tensor b has incompatible shape");

    // Create the output tensor on the same device and with the same dtype as input
    auto out = torch::empty({M, N}, x.options());

    // Kernel launch configuration
    const int block_size = 256;
    // Calculate the number of blocks needed
    const int num_blocks = (M * N + block_size - 1) / block_size;

    // Launch the kernel
    fused_gemm_bias_relu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N, K);

    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# C++ source for the function signature, required by load_inline
fused_gemm_bias_relu_cpp_source = """
torch::Tensor fused_gemm_bias_relu_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b);
"""

# JIT compile the CUDA and C++ code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_gemm_bias_relu_cpp_source,
    cuda_sources=fused_gemm_bias_relu_source,
    functions=["fused_gemm_bias_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that uses a custom fused CUDA kernel for GEMM + Bias + ReLU.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        # We still define the original layers to hold the parameters (weights and biases)
        # so they are properly registered, tracked by optimizers, and can be saved/loaded.
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # Store the compiled custom operator
        self.fused_op = fused_op

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        # Call the custom fused CUDA kernel.
        # It is assumed that the model and the input tensor 'x' have already been
        # moved to the same CUDA device before calling the forward pass.
        return self.fused_op.fused_gemm_bias_relu_cuda(x, self.gemm.weight, self.bias)