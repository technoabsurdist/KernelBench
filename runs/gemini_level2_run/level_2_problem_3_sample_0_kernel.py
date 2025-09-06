import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels and their C++ wrappers
fused_ops_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu_forward(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

// Fused Add + LayerNorm Kernel
// Normalizes over the last dimension of the input tensor.
template <typename T>
__global__ void fused_add_layernorm_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T sum_weight,
    float epsilon,
    int rows,
    int cols) {

    // Each block processes one row (one feature vector to be normalized)
    int row_idx = blockIdx.x;
    if (row_idx >= rows) return;

    // Use shared memory for reduction
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    const T* row_input = input + row_idx * cols;
    T* row_output = output + row_idx * cols;

    // --- Step 1: Calculate mean ---
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        local_sum += (float)row_input[i] + (float)sum_weight;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sdata[0] = sdata[0] / cols; // Store mean in sdata[0]
    }
    __syncthreads();
    float mean = sdata[0];

    // --- Step 2: Calculate variance ---
    float local_var_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = (float)row_input[i] + (float)sum_weight;
        float dev = val - mean;
        local_var_sum += dev * dev;
    }
    sdata[tid] = local_var_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sdata[0] = sdata[0] / cols; // Store variance in sdata[0]
    }
    __syncthreads();
    float var = sdata[0];
    float rstd = rsqrtf(var + epsilon);

    // --- Step 3: Normalize and apply affine transform ---
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = (float)row_input[i] + (float)sum_weight;
        float normalized_val = (val - mean) * rstd;
        row_output[i] = (T)(normalized_val * (float)gamma[i] + (float)beta[i]);
    }
}

// Fused AvgPool3D + GELU Kernel
template <typename T>
__global__ void fused_avgpool3d_gelu_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int kD, int kH, int kW) {

    // Each thread computes one output element
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total_outputs = (long long)N * C * D_out * H_out * W_out;

    if (idx >= total_outputs) return;

    // Calculate output coordinates
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int d_out = (idx / (W_out * H_out)) % D_out;
    int c = (idx / (W_out * H_out * D_out)) % C;
    int n = idx / (W_out * H_out * D_out * C);

    // Calculate input window start coordinates (stride is same as kernel size)
    int d_start = d_out * kD;
    int h_start = h_out * kH;
    int w_start = w_out * kW;

    float sum = 0.0f;
    for (int kd = 0; kd < kD; ++kd) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                int d_in = d_start + kd;
                int h_in = h_start + kh;
                int w_in = w_start + kw;

                long long input_idx = (long long)n * C * D_in * H_in * W_in +
                                      (long long)c * D_in * H_in * W_in +
                                      (long long)d_in * H_in * W_in +
                                      (long long)h_in * W_in +
                                      w_in;
                sum += (float)input[input_idx];
            }
        }
    }

    float avg = sum / (kD * kH * kW);
    float gelu_val = gelu_forward(avg);

    output[idx] = (T)gelu_val;
}

// C++ Wrapper for Fused Add + LayerNorm
torch::Tensor fused_add_layernorm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float sum_weight,
    float epsilon) {

    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "Gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "Beta must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "Gamma must be contiguous");
    TORCH_CHECK(beta.is_contiguous(), "Beta must be contiguous");

    auto input_sizes = input.sizes();
    int rank = input_sizes.size();
    TORCH_CHECK(rank > 0, "Input tensor cannot be empty");

    long cols = input_sizes[rank - 1];
    long rows = input.numel() / cols;

    TORCH_CHECK(gamma.numel() == cols, "Gamma size must match the size of the last dimension of input");
    TORCH_CHECK(beta.numel() == cols, "Beta size must match the size of the last dimension of input");

    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = rows;
    const int shared_mem_size = block_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_add_layernorm_kernel_launcher", ([&] {
        fused_add_layernorm_kernel<scalar_t><<<num_blocks, block_size, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            (scalar_t)sum_weight,
            epsilon,
            rows,
            cols
        );
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// C++ Wrapper for Fused AvgPool3D + GELU
torch::Tensor fused_avgpool3d_gelu_cuda(
    torch::Tensor input,
    int kD, int kH, int kW) {

    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 5, "Input must be a 5D tensor (N, C, D, H, W)");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    const auto sizes = input.sizes();
    const int N = sizes[0];
    const int C = sizes[1];
    const int D_in = sizes[2];
    const int H_in = sizes[3];
    const int W_in = sizes[4];

    // Assuming stride is equal to kernel size for nn.AvgPool3d default behavior
    const int D_out = D_in / kD;
    const int H_out = H_in / kH;
    const int W_out = W_in / kW;

    auto output = torch::empty({N, C, D_out, H_out, W_out}, input.options());
    long long total_outputs = output.numel();

    if (total_outputs == 0) {
        return output;
    }

    const int block_size = 256;
    const int num_blocks = (total_outputs + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_avgpool3d_gelu_kernel_launcher", ([&] {
        fused_avgpool3d_gelu_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, C, D_in, H_in, W_in,
            D_out, H_out, W_out,
            kD, kH, kW
        );
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

# C++ source for function declarations (the "header")
fused_ops_cpp_source = """
torch::Tensor fused_add_layernorm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float sum_weight,
    float epsilon);

torch::Tensor fused_avgpool3d_gelu_cuda(
    torch::Tensor input,
    int kD, int kH, int kW);
"""

# JIT compile the custom operators
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_cuda_source,
    functions=["fused_add_layernorm_cuda", "fused_avgpool3d_gelu_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses (sum + LayerNorm) and (AvgPool + GELU) into custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        
        # We still need the LayerNorm module to hold the learnable parameters (gamma, beta) and epsilon
        self.norm = nn.LayerNorm(norm_shape)
        
        # Store pooling kernel size for the custom kernel
        self.pool_kernel_size = pool_kernel_size
        
        # The AvgPool and GELU layers are no longer needed as they are fused.
        
        # Store the compiled custom operators
        self.fused_ops = fused_ops

    def forward(self, x):
        # 1. Apply the standard, highly optimized ConvTranspose3d
        x = self.conv_transpose(x)
        
        # 2. Apply the fused (element-wise add + LayerNorm) kernel
        # The original LayerNorm normalizes over the last dimension, which is consistent with our kernel.
        x = self.fused_ops.fused_add_layernorm_cuda(
            x, self.norm.weight, self.norm.bias, self.sum_weight.item(), self.norm.eps
        )
        
        # 3. Apply the fused (AvgPool3d + GELU) kernel
        x = self.fused_ops.fused_avgpool3d_gelu_cuda(
            x, self.pool_kernel_size[0], self.pool_kernel_size[1], self.pool_kernel_size[2]
        )
        
        return x