import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for the fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// GELU approximation using tanh, common in high-performance kernels
__device__ __forceinline__ float gelu_forward(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// Kernel 1: Fused (Subtract -> GlobalAvgPool -> GELU)
// This kernel reads the GEMM output, subtracts a parameter, computes the mean over the feature dimension,
// and applies GELU, all in one pass. This avoids materializing large intermediate tensors.
// Each CUDA block is responsible for computing one element of the output tensor (i.e., one row of the batch).
__global__ void fused_subtract_mean_gelu_kernel(
    const float* gemm_out,
    const float* subtract_param,
    float* out,
    int B,
    int O
) {
    // Use dynamically allocated shared memory for the reduction
    extern __shared__ float sdata[];

    // Each block processes one row of the input `gemm_out`
    int b = blockIdx.x;

    // Each thread computes a partial sum for its part of the row
    float partial_sum = 0.0f;
    for (int j = threadIdx.x; j < O; j += blockDim.x) {
        // Fused Subtract operation
        float val = gemm_out[b * O + j] - subtract_param[j];
        partial_sum += val;
    }

    sdata[threadIdx.x] = partial_sum;
    __syncthreads();

    // Perform reduction in shared memory. This sums up the partial sums from all threads in the block.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // The first thread in the block writes the final result
    if (threadIdx.x == 0) {
        float total_sum = sdata[0];
        // Fused GlobalAvgPool operation
        float mean = total_sum / O;
        // Fused GELU operation
        // Note: The original model's LogSumExp on a single-element tensor is an identity op, so it's omitted.
        out[b] = gelu_forward(mean);
    }
}

// Kernel 2: Fused Residual Add with broadcasting
// This kernel adds a (B, 1) tensor to a (B, I) tensor.
__global__ void fused_residual_add_kernel(
    const float* gelu_out,
    const float* original_x,
    float* out,
    int B,
    int I
) {
    // Standard 2D grid-stride loop
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < B && col < I) {
        // Broadcasting: gelu_out[row] is added to every element in the corresponding row of original_x
        float residual_val = gelu_out[row];
        float original_val = original_x[row * I + col];
        out[row * I + col] = residual_val + original_val;
    }
}

// C++ wrapper functions to be called from Python
torch::Tensor fused_subtract_mean_gelu_cuda(torch::Tensor gemm_out, torch::Tensor subtract_param) {
    TORCH_CHECK(gemm_out.is_cuda(), "gemm_out must be a CUDA tensor");
    TORCH_CHECK(subtract_param.is_cuda(), "subtract_param must be a CUDA tensor");
    TORCH_CHECK(gemm_out.is_contiguous(), "gemm_out must be contiguous");
    TORCH_CHECK(subtract_param.is_contiguous(), "subtract_param must be contiguous");
    TORCH_CHECK(gemm_out.dim() == 2, "gemm_out must be 2D");
    TORCH_CHECK(subtract_param.dim() == 1, "subtract_param must be 1D");
    TORCH_CHECK(gemm_out.size(1) == subtract_param.size(0), "Dimension mismatch");

    const int B = gemm_out.size(0);
    const int O = gemm_out.size(1);

    auto out = torch::zeros({B, 1}, gemm_out.options());

    const int block_size = 512; // A good default for reduction kernels
    const int num_blocks = B;
    
    // Shared memory size for the reduction
    size_t smem_size = block_size * sizeof(float);

    fused_subtract_mean_gelu_kernel<<<num_blocks, block_size, smem_size>>>(
        gemm_out.data_ptr<float>(),
        subtract_param.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        O
    );
    
    return out;
}

torch::Tensor fused_residual_add_cuda(torch::Tensor gelu_out, torch::Tensor original_x) {
    TORCH_CHECK(gelu_out.is_cuda(), "gelu_out must be a CUDA tensor");
    TORCH_CHECK(original_x.is_cuda(), "original_x must be a CUDA tensor");
    TORCH_CHECK(gelu_out.is_contiguous(), "gelu_out must be contiguous");
    TORCH_CHECK(original_x.is_contiguous(), "original_x must be contiguous");
    TORCH_CHECK(gelu_out.dim() == 2 && gelu_out.size(1) == 1, "gelu_out must be of shape (B, 1)");
    TORCH_CHECK(original_x.dim() == 2, "original_x must be 2D");
    TORCH_CHECK(gelu_out.size(0) == original_x.size(0), "Batch dimension mismatch");

    const int B = original_x.size(0);
    const int I = original_x.size(1);

    auto out = torch::zeros_like(original_x);

    const dim3 block_dim(16, 16);
    const dim3 grid_dim(
        (I + block_dim.x - 1) / block_dim.x,
        (B + block_dim.y - 1) / block_dim.y
    );

    fused_residual_add_kernel<<<grid_dim, block_dim>>>(
        gelu_out.data_ptr<float>(),
        original_x.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        I
    );

    return out;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_subtract_mean_gelu_cuda(torch::Tensor gemm_out, torch::Tensor subtract_param);
torch::Tensor fused_residual_add_cuda(torch::Tensor gelu_out, torch::Tensor original_x);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_subtract_mean_gelu_cuda", "fused_residual_add_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that replaces a series of operations with two fused CUDA kernels.
    1. Fused (Subtract -> GlobalAvgPool -> GELU)
    2. Fused (ResidualAdd with broadcasting)
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        # The GEMM operation is kept as a standard PyTorch layer, as it's highly optimized.
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        # Store the compiled custom operators
        self.fused_ops = fused_ops

    def forward(self, x):
        # The original model clones, so we do too to match behavior.
        original_x = x.clone().detach()

        # 1. Gemm (using PyTorch's highly optimized cuBLAS implementation)
        gemm_out = self.gemm(x)

        # 2. Fused Kernel 1: (Subtract -> GlobalAvgPool -> GELU)
        # This single kernel replaces three separate operations and avoids materializing
        # the large intermediate tensor after the subtraction.
        gelu_out = self.fused_ops.fused_subtract_mean_gelu_cuda(gemm_out, self.subtract)

        # 3. Fused Kernel 2: ResidualAdd with broadcasting
        # This kernel performs the final addition.
        x = self.fused_ops.fused_residual_add_cuda(gelu_out, original_x)

        return x