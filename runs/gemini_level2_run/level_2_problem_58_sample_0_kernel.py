import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// Fused kernel: LogSumExp -> HardSwish-like -> Subtract Bias -> Clamp
// The reduction is performed over the channel dimension (dim=1) for each spatial location.
__global__ void fused_lse_hswish_sub_clamp_kernel(
    const float* x,      // Input tensor (N, C, D, H, W)
    const float* bias,   // Bias tensor (scalar)
    float* out,          // Output tensor (N, 1, D, H, W)
    const int C,         // Channel size
    const int spatial_stride) { // D * H * W

    // Each block computes one spatial location. The grid is 1D over all spatial locations.
    const int spatial_idx = blockIdx.x;

    // Shared memory for block-wide reduction
    extern __shared__ float sdata[];
    float* s_max = sdata;
    float* s_sum = (float*)&s_max[blockDim.x];

    const float* x_ptr = x + spatial_idx;

    // --- Step 1: Find max value for numerical stability ---
    float thread_max = -FLT_MAX;
    // Each thread iterates over a stride of channels
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        thread_max = fmaxf(thread_max, x_ptr[c * spatial_stride]);
    }

    // Block-wide reduction for max
    s_max[threadIdx.x] = thread_max;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    const float block_max = s_max[0];
    __syncthreads();


    // --- Step 2: Compute sum of exps ---
    float thread_sum_exp = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        thread_sum_exp += expf(x_ptr[c * spatial_stride] - block_max);
    }

    // Block-wide reduction for sum
    s_sum[threadIdx.x] = thread_sum_exp;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + offset];
        }
        __syncthreads();
    }
    const float block_sum_exp = s_sum[0];


    // --- Step 3: Final computation by thread 0 ---
    if (threadIdx.x == 0) {
        // LogSumExp
        float lse_val = block_max + logf(block_sum_exp);

        // HardSwish-like: y * sigmoid(y + 3) / 6
        float sigmoid_arg = lse_val + 3.0f;
        float sigmoid_val = 1.0f / (1.0f + expf(-sigmoid_arg));
        float hswish_val = lse_val * sigmoid_val / 6.0f;

        // Subtract bias
        float sub_val = hswish_val - *bias;

        // Clamp
        float final_val = fmaxf(-1.0f, fminf(1.0f, sub_val));

        // Write to output
        out[spatial_idx] = final_val;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Input bias must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 5, "Input x must be a 5D tensor");
    TORCH_CHECK(bias.numel() == 1, "Bias must be a scalar tensor");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");

    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto D = x.size(2);
    const auto H = x.size(3);
    const auto W = x.size(4);

    // Output tensor has same spatial dimensions but C=1
    auto out = torch::empty({N, 1, D, H, W}, x.options());

    const int num_spatial_elements = N * D * H * W;
    const int spatial_stride = D * H * W;

    // Use a standard block size, which should be a power of 2 for the reduction.
    const int block_size = 256;
    const int num_blocks = num_spatial_elements;

    // Shared memory size: one array for max, one for sum
    const size_t shared_mem_size = 2 * block_size * sizeof(float);

    fused_lse_hswish_sub_clamp_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        C,
        spatial_stride
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

fused_op_cpp_source = "torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor bias);"

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
    Model that performs a 3D transposed convolution, followed by a fused custom
    kernel for LogSumExp, HardSwish, subtraction, and clamp operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # The bias is a single parameter, as in the original model.
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        # The custom kernel expects a contiguous tensor for correct memory access.
        # While ConvTranspose3d usually returns a contiguous tensor, this is a safe practice.
        x_contiguous = x.contiguous()
        return fused_op.fused_op_cuda(x_contiguous, self.bias)