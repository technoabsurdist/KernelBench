import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation
# This kernel fuses the following sequence:
# 1. Mean pooling (across depth, dim=2)
# 2. Bias addition (broadcasted)
# 3. Softmax (across channels, dim=1)
# 4. Tanh activation
# 5. Scaling
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat> // For FLT_MAX
#include <cmath>  // For expf, tanhf

// Helper for block-level sum reduction using shared memory.
// The final result is in s_reduce[0].
__device__ void block_reduce_sum(float* s_reduce, float val) {
    s_reduce[threadIdx.x] = val;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            s_reduce[threadIdx.x] += s_reduce[threadIdx.x + offset];
        }
        __syncthreads();
    }
}

// Helper for block-level max reduction using shared memory.
// The final result is in s_reduce[0].
__device__ void block_reduce_max(float* s_reduce, float val) {
    s_reduce[threadIdx.x] = val;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            s_reduce[threadIdx.x] = fmaxf(s_reduce[threadIdx.x], s_reduce[threadIdx.x + offset]);
        }
        __syncthreads();
    }
}

__global__ void fused_op_kernel(
    const float* x_conv,      // Input from conv_transpose
    const float* bias,        // Bias tensor
    float* out,               // Output tensor
    const float scaling_factor,
    const int B, const int C, const int D, const int H, const int W
) {
    // Shared memory layout:
    // s_mem[0...C-1]: for intermediate mean+bias values
    // s_mem[C...C+blockDim.x-1]: for block-wide reductions
    extern __shared__ float s_mem[];
    float* s_data = s_mem;
    float* s_reduce = s_mem + C;

    // Grid setup: Each block processes one (b, h, w) "pixel"
    const int hw = blockIdx.x;
    const int b = blockIdx.y;

    if (hw >= H * W || b >= B) return;

    const int h = hw / W;
    const int w = hw % W;

    // --- 1. Mean Pooling & Bias Addition ---
    // Each thread computes the mean for one or more channels
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float sum = 0.0f;
        // Iterate over depth D
        for (int d = 0; d < D; ++d) {
            long long x_idx = (long long)b * C * D * H * W +
                              (long long)c * D * H * W +
                              (long long)d * H * W +
                              (long long)h * W + w;
            sum += x_conv[x_idx];
        }
        // Store mean + bias in shared memory. Bias is (1,C,1,1,1), so bias[c] is valid.
        s_data[c] = (sum / (float)D) + bias[c];
    }
    __syncthreads();

    // --- 2. Softmax ---
    // a. Find max value
    float thread_max = -FLT_MAX;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        thread_max = fmaxf(thread_max, s_data[c]);
    }
    block_reduce_max(s_reduce, thread_max);
    const float max_val = s_reduce[0];
    __syncthreads(); // Ensure all threads see the max_val from s_reduce[0]

    // b. Calculate sum of exps
    float thread_sum_exp = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        thread_sum_exp += expf(s_data[c] - max_val);
    }
    block_reduce_sum(s_reduce, thread_sum_exp);
    const float sum_exp = s_reduce[0];
    __syncthreads(); // Ensure all threads see the sum_exp from s_reduce[0]

    // --- 3. Final Calculation & Write ---
    // Each thread calculates softmax, tanh, scale for its channels and writes to output
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float softmax_val = expf(s_data[c] - max_val) / sum_exp;
        float tanh_val = tanhf(softmax_val);
        float final_val = tanh_val * scaling_factor;

        // Output shape is (B, C, 1, H, W)
        long long out_idx = (long long)b * C * H * W +
                            (long long)c * H * W +
                            (long long)h * W + w;
        out[out_idx] = final_val;
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor x_conv,
    torch::Tensor bias,
    double scaling_factor
) {
    TORCH_CHECK(x_conv.is_cuda(), "x_conv must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x_conv.dim() == 5, "x_conv must be a 5D tensor");
    TORCH_CHECK(bias.dim() == 5, "bias must be a 5D tensor");
    TORCH_CHECK(x_conv.is_contiguous(), "x_conv must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    const auto B = x_conv.size(0);
    const auto C = x_conv.size(1);
    const auto D = x_conv.size(2);
    const auto H = x_conv.size(3);
    const auto W = x_conv.size(4);

    // The output tensor has the same shape except D=1
    auto out = torch::empty({B, C, 1, H, W}, x_conv.options());

    const int block_size = 128; // A reasonable block size
    dim3 threads(block_size);
    dim3 blocks(H * W, B);

    // Shared memory size: C for data + block_size for reduction
    const size_t shared_mem_size = (C + block_size) * sizeof(float);

    fused_op_kernel<<<blocks, threads, shared_mem_size>>>(
        x_conv.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        (float)scaling_factor,
        B, C, D, H, W
    );
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_op_cuda(torch::Tensor x_conv, torch::Tensor bias, double scaling_factor);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Model that performs a series of operations with a fused custom kernel:
    1. Transposed 3D convolution (PyTorch)
    2. Fused op (mean pool -> bias add -> softmax -> tanh -> scale) (Custom CUDA)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        # The custom kernel expects contiguous tensors for safe direct memory access
        x = self.fused_op.fused_op_cuda(x.contiguous(), self.bias.contiguous(), self.scaling_factor)
        return x