import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused kernel for: softmax(dim=1) -> add bias -> mul scaling_factor -> sigmoid
__global__ void fused_softmax_bias_scale_sigmoid_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float scaling_factor,
    const int N, const int C, const int H, const int W) {

    // Each block processes one spatial location (n, h, w) across all C channels.
    // Threads in the block correspond to the channel index.
    // blockDim.x is expected to be C.
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;

    const int spatial_idx = blockIdx.x;
    if (spatial_idx >= N * H * W) return;

    // Calculate n, h, w from the block index
    const int w = spatial_idx % W;
    const int h = (spatial_idx / W) % H;
    const int n = spatial_idx / (H * W);

    // Base pointer for the current input slice (n, :, h, w)
    const float* input_ptr = input + n * C * H * W + h * W + w;
    // Base pointer for the current output slice
    float* output_ptr = output + n * C * H * W + h * W + w;

    // --- Stage 1: Find max for stable softmax ---
    // Load data into shared memory with channel stride
    sdata[tid] = input_ptr[tid * H * W];
    __syncthreads();

    // Parallel reduction to find max
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tid < i) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + i]);
        }
        __syncthreads();
    }
    const float max_val = sdata[0];
    __syncthreads();

    // --- Stage 2: Calculate sum of exps ---
    // Load data again, subtract max, and exponentiate
    float val = input_ptr[tid * H * W];
    const float exp_val = expf(val - max_val);
    sdata[tid] = exp_val;
    __syncthreads();

    // Parallel reduction to find sum
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tid < i) {
            sdata[tid] += sdata[tid + i];
        }
        __syncthreads();
    }
    // After reduction, sdata[0] contains the sum for the block.
    // This __syncthreads() ensures all threads have finished the reduction
    // before any thread proceeds to read the result from sdata[0].
    __syncthreads();
    const float sum_val = sdata[0];

    // --- Stage 3: Final fused computation and write ---
    // Denominator for softmax is sum_val. Add a small epsilon for stability.
    const float inv_sum_val = 1.0f / (sum_val + 1e-8f);
    
    // Softmax
    const float softmax_val = exp_val * inv_sum_val;
    
    // Bias is a 1D array of size C, so we can just index by channel (tid)
    const float bias_val = bias[tid];
    
    // Scale and apply sigmoid
    const float scaled_val = (softmax_val + bias_val) * scaling_factor;
    const float final_val = 1.0f / (1.0f + expf(-scaled_val));

    // Write to output
    output_ptr[tid * H * W] = final_val;
}

torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    const float scaling_factor) {

    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    // Ensure input is NCHW and contiguous
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(input.is_contiguous(torch::MemoryFormat::Contiguous), "Input must be NCHW contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    TORCH_CHECK(bias.dim() == 1, "Bias must be a 1D tensor for this kernel");

    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);

    TORCH_CHECK(C == bias.size(0), "Bias size must match input channel size");
    TORCH_CHECK(C <= 1024, "Channel size must be <= 1024 for this kernel implementation");

    auto output = torch::empty_like(input);

    const dim3 threads(C);
    const dim3 blocks(N * H * W);
    // Shared memory size is C floats
    const int shared_mem_size = C * sizeof(float);

    fused_softmax_bias_scale_sigmoid_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        N, C, H, W
    );
    
    // Check for any CUDA errors that might have occurred during the kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

# Define the C++ wrapper source string for the CUDA function
fused_op_cpp_source = """
torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor bias, const float scaling_factor);
"""

# Compile the inline CUDA code using torch's C++ extension loader
# This will be compiled on the fly the first time it's used.
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, then a fused operation of:
    softmax, bias addition, scaling, and sigmoid using a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        # Use the standard, highly optimized ConvTranspose2d from PyTorch
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        
        # The bias is still a learnable parameter of the model
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # The scaling factor is a constant
        self.scaling_factor = scaling_factor
        
        # Store the compiled fused operator from the JIT compilation
        self.fused_op = fused_op.fused_op_cuda

    def forward(self, x):
        # 1. Apply the standard ConvTranspose2d layer
        x = self.conv_transpose(x)
        
        # The custom kernel expects a 1D bias tensor of size C.
        # The nn.Parameter `self.bias` has shape (C, 1, 1), so we squeeze it.
        bias_1d = self.bias.squeeze()

        # 2. Apply the custom fused CUDA kernel for the sequence of
        #    softmax -> add bias -> scale -> sigmoid
        x = self.fused_op(x, bias_1d, self.scaling_factor)
        
        return x