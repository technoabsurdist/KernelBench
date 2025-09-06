import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for the fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// --- Kernel 1: Global Average Pooling ---
// This kernel computes the global average pool over the last two dimensions (H, W)
// of a 4D tensor (B, C, H, W).
// Grid is launched with (B, C) blocks, and each block computes the average for one plane.
__global__ void global_avg_pool_2d_kernel(
    const float* input,
    float* output,
    int B, int C, int H, int W)
{
    // Use shared memory for efficient parallel reduction within a block.
    extern __shared__ float sdata[];

    int b = blockIdx.x;
    int c = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const float* plane_in = input + b * C * H * W + c * H * W;
    int plane_size = H * W;

    // Each thread computes a partial sum from the input plane.
    float my_sum = 0.0f;
    for (int i = tid; i < plane_size; i += block_size) {
        my_sum += plane_in[i];
    }

    sdata[tid] = my_sum;
    __syncthreads();

    // Perform reduction in shared memory.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final result (mean) to the output tensor.
    if (tid == 0) {
        output[b * C + c] = sdata[0] / (float)plane_size;
    }
}

// --- Kernel 2: Bias -> LogSumExp -> Mul ---
// This kernel fuses the addition of bias, log-sum-exp reduction over the channel dimension,
// and multiplication by a scalar.
// It assumes the number of channels C is a power of two, and the block size is equal to C.
// Grid is launched with (B) blocks, each handling one item in the batch.
__global__ void bias_lse_mul_1d_kernel(
    const float* input,      // Shape (B, C)
    const float* bias,       // Shape (C)
    float* output,           // Shape (B)
    int B, int C,
    float mul_factor)
{
    extern __shared__ float sdata[];
    int b = blockIdx.x;
    int tid = threadIdx.x; // Corresponds to channel index

    // Step 1: Load input and add bias
    float val = input[b * C + tid] + bias[tid];
    sdata[tid] = val;
    __syncthreads();

    // Step 2: Numerically stable LogSumExp reduction
    // Pass 1: Find the maximum value across channels for the current batch item.
    for (unsigned int s = C / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // Pass 2: Compute sum(exp(val - max_val))
    sdata[tid] = expf(val - max_val);
    __syncthreads();
    for (unsigned int s = C / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float sum_exp = sdata[0];

    // Step 3: Final calculation and multiplication by the scalar factor.
    // Thread 0 writes the final result for the batch item.
    if (tid == 0) {
        float lse = max_val + logf(sum_exp);
        output[b] = lse * mul_factor;
    }
}

// --- C++ Wrapper Functions ---

torch::Tensor global_avg_pool_2d_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");

    int B = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::zeros({B, C}, input.options());

    const int block_size = 256; // A common choice for reduction kernels
    dim3 grid_dim(B, C);
    dim3 block_dim(block_size);
    size_t shared_mem_size = block_size * sizeof(float);

    global_avg_pool_2d_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W
    );
    return output;
}

torch::Tensor bias_lse_mul_1d_cuda(torch::Tensor input, torch::Tensor bias, float mul_factor) {
    TORCH_CHECK(input.is_cuda() && bias.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(input.dim() == 2, "Input must be a 2D tensor");
    TORCH_CHECK(bias.dim() == 1, "Bias must be a 1D tensor");
    TORCH_CHECK(input.size(1) == bias.size(0), "Input dim 1 must match bias size");

    int B = input.size(0);
    int C = input.size(1);

    TORCH_CHECK((C > 0) && ((C & (C - 1)) == 0), "C must be a power of 2 for this kernel version");

    auto output = torch::zeros({B}, input.options());

    dim3 grid_dim(B);
    dim3 block_dim(C); // Set block size to C, as required by the kernel
    size_t shared_mem_size = C * sizeof(float);

    bias_lse_mul_1d_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C,
        mul_factor
    );
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor global_avg_pool_2d_cuda(torch::Tensor input);
torch::Tensor bias_lse_mul_1d_cuda(torch::Tensor input, torch::Tensor bias, float mul_factor);
"""

# JIT compile the CUDA kernels
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["global_avg_pool_2d_cuda", "bias_lse_mul_1d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that replaces the sequence of pooling, bias addition, log-sum-exp,
    sum, and multiplication with two custom fused CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        # The ConvTranspose2d layer is kept as is, as it's highly optimized in cuDNN.
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Store the compiled custom operators
        self.fused_ops = fused_ops

    def forward(self, x):
        # 1. Perform the transposed convolution (unchanged)
        x = self.conv_transpose(x)
        
        # 2. Call the first custom kernel for global average pooling.
        # This replaces `torch.mean(x, dim=(2, 3), keepdim=True)`
        # Input shape: (B, C, H, W), Output shape: (B, C)
        pooled = self.fused_ops.global_avg_pool_2d_cuda(x)
        
        # Prepare the bias tensor for the kernel (needs to be 1D)
        bias_1d = self.bias.squeeze()
        
        # 3. Call the second custom kernel to fuse the remaining operations:
        # - Add bias
        # - Log-sum-exp over channels
        # - Sum over trivial spatial dimensions (handled by reduction)
        # - Multiply by 10.0
        # Input shapes: (B, C), (C), scalar. Output shape: (B)
        x = self.fused_ops.bias_lse_mul_1d_cuda(pooled, bias_1d, 10.0)
        
        # The original model's output shape is (batch_size, 1).
        # We unsqueeze the last dimension to match it.
        return x.unsqueeze(1)