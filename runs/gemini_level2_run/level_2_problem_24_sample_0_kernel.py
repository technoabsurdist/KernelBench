import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

min_softmax_fused_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Helper function to find the next power of 2.
// e.g., next_pow2(24) -> 32
int next_pow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n > 0 ? n : 1;
}

__global__ void min_reduce_softmax_kernel(
    const float* input,
    float* output,
    int B, int C, int D, int H, int W,
    long stride_b, long stride_c, long stride_d, long stride_h, long stride_w,
    long out_stride_b, long out_stride_c, long out_stride_h, long out_stride_w) {

    // Each block is responsible for one spatial location (b, h, w) across all channels.
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int w = blockIdx.x;

    // Each thread in the block is responsible for one channel.
    const int c = threadIdx.x;

    // Shared memory for inter-thread communication within the block.
    // Used for the two reduction steps (max and sum) in the softmax calculation.
    extern __shared__ float sdata[];

    float min_val;

    // --- 1. Min Reduction ---
    // Each thread finds the minimum value along the depth dimension (D) for its assigned channel.
    // This check ensures we only process valid channels.
    if (c < C) {
        const float* p_in = input + b * stride_b + c * stride_c + h * stride_h + w * stride_w;
        min_val = p_in[0]; // Initialize with the first element
        for (int d = 1; d < D; ++d) {
            min_val = fminf(min_val, p_in[d * stride_d]);
        }
        sdata[c] = min_val;
    } else {
        // Threads outside the valid channel range initialize with the identity for the max reduction.
        sdata[c] = -FLT_MAX;
    }
    __syncthreads();

    // --- 2. Softmax ---
    // The vector to be softmaxed is now in shared memory (sdata).
    
    // Step 2a: Find the maximum value in sdata for numerical stability.
    // This is a parallel reduction using shared memory. blockDim.x is a power of 2.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (c < s) {
            sdata[c] = fmaxf(sdata[c], sdata[c + s]);
        }
        __syncthreads();
    }
    // The maximum value is now in sdata[0], accessible by all threads in the block.
    float max_val = sdata[0];

    // Step 2b: Subtract max from the original min_val, exponentiate, and store back to sdata.
    float exp_val = 0.0f;
    if (c < C) {
        exp_val = expf(min_val - max_val);
        sdata[c] = exp_val;
    } else {
        // Threads outside the valid channel range use the identity for the sum reduction.
        sdata[c] = 0.0f;
    }
    __syncthreads();

    // Step 2c: Sum the exponentiated values.
    // This is another parallel reduction.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (c < s) {
            sdata[c] += sdata[c + s];
        }
        __syncthreads();
    }
    // The sum is now in sdata[0].
    float sum_exp = sdata[0];

    // Step 2d: Divide to get the final softmax probability and write to the output tensor.
    if (c < C) {
        float final_val = exp_val / (sum_exp + 1e-8f); // Add epsilon for stability
        float* p_out = output + b * out_stride_b + c * out_stride_c + h * out_stride_h + w * out_stride_w;
        *p_out = final_val;
    }
}

// C++ wrapper function that launches the CUDA kernel.
torch::Tensor min_reduce_softmax_cuda(torch::Tensor input, int dim) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(input.dim() == 5, "Input must be a 5D tensor of shape (B, C, D, H, W)");
    TORCH_CHECK(dim == 2, "Custom kernel only supports reduction over dim=2 (depth)");

    const auto sizes = input.sizes();
    const int B = sizes[0];
    const int C = sizes[1];
    const int D = sizes[2];
    const int H = sizes[3];
    const int W = sizes[4];

    // The reduction is over dim=2, so the output shape will be (B, C, H, W)
    auto output = torch::empty({B, C, H, W}, input.options());

    // Get tensor strides for correct memory access in the kernel
    const auto strides = input.strides();
    const long stride_b = strides[0];
    const long stride_c = strides[1];
    const long stride_d = strides[2];
    const long stride_h = strides[3];
    const long stride_w = strides[4];

    const auto out_strides = output.strides();
    const long out_stride_b = out_strides[0];
    const long out_stride_c = out_strides[1];
    const long out_stride_h = out_strides[2];
    const long out_stride_w = out_strides[3];

    // Configure kernel launch parameters
    dim3 grid_dim(W, H, B);
    
    // Pad block size to the next power of 2 for efficient parallel reduction.
    // The number of threads per block will equal the padded channel count.
    int block_size = next_pow2(C);
    if (block_size > 1024) block_size = 1024; // Respect device limits
    dim3 block_dim(block_size);

    // Shared memory size depends on the padded block size.
    size_t shared_mem_size = block_size * sizeof(float);

    // Launch the kernel
    min_reduce_softmax_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W,
        stride_b, stride_c, stride_d, stride_h, stride_w,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w
    );
    
    // Check for kernel launch errors for easier debugging.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

min_softmax_fused_cpp_source = """
torch::Tensor min_reduce_softmax_cuda(torch::Tensor input, int dim);
"""

# JIT compile the custom CUDA kernel
min_softmax_fused = load_inline(
    name="min_softmax_fused",
    cpp_sources=min_softmax_fused_cpp_source,
    cuda_sources=min_softmax_fused_cuda_source,
    functions=["min_reduce_softmax_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses the min reduction and softmax operations into a single custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W)
        """
        x = self.conv(x)
        # Replace torch.min and torch.softmax with the custom fused kernel.
        # The kernel internally handles the min reduction along dim=2 and softmax along dim=1.
        x = min_softmax_fused.min_reduce_softmax_cuda(x, self.dim)
        return x