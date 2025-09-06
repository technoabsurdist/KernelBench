import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for the custom operators
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat> // For FLT_MIN

// Kernel for 3D Max Pooling
__global__ void max_pool3d_kernel(const float* input, float* output,
                                  const int N, const int C,
                                  const int D_in, const int H_in, const int W_in,
                                  const int D_out, const int H_out, const int W_out,
                                  const int kernel_size, const int stride) {
    // Using a 1D grid-stride loop to cover all output elements
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total_threads = (long long)N * C * D_out * H_out * W_out;

    if (idx >= total_threads) return;

    // Decompose the 1D index 'idx' into 5D output coordinates (n, c, d_out, h_out, w_out)
    const int w_out = idx % W_out;
    const int h_out = (idx / W_out) % H_out;
    const int d_out = (idx / (W_out * H_out)) % D_out;
    const int c = (idx / (W_out * H_out * D_out)) % C;
    const int n = idx / (W_out * H_out * D_out * C);

    // Calculate the starting coordinates of the pooling window in the input tensor
    const int d_in_start = d_out * stride;
    const int h_in_start = h_out * stride;
    const int w_in_start = w_out * stride;

    float max_val = -FLT_MAX;

    // Iterate over the 3D pooling window
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int d_in = d_in_start + kd;
                const int h_in = h_in_start + kh;
                const int w_in = w_in_start + kw;

                // Calculate the 1D index for the input tensor
                long long input_idx = (long long)n * C * D_in * H_in * W_in +
                                      (long long)c * D_in * H_in * W_in +
                                      (long long)d_in * H_in * W_in +
                                      (long long)h_in * W_in +
                                      w_in;
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }
    }

    // Write the maximum value to the output tensor
    output[idx] = max_val;
}

// Fused Kernel for 3D Max Pooling followed by Summation across the channel dimension
__global__ void max_pool3d_sum_dim1_kernel(const float* input, float* output,
                                           const int N, const int C,
                                           const int D_in, const int H_in, const int W_in,
                                           const int D_out, const int H_out, const int W_out,
                                           const int kernel_size, const int stride) {
    // Each block computes one output element (summed over all channels)
    long long block_idx = blockIdx.x;
    long long total_blocks = (long long)N * D_out * H_out * W_out;

    if (block_idx >= total_blocks) return;

    // Thread index within the block
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Decompose the block index into 4D output coordinates (n, d_out, h_out, w_out)
    const int w_out = block_idx % W_out;
    const int h_out = (block_idx / W_out) % H_out;
    const int d_out = (block_idx / (W_out * H_out)) % D_out;
    const int n = block_idx / (W_out * H_out * D_out);

    // Calculate the starting coordinates of the pooling window
    const int d_in_start = d_out * stride;
    const int h_in_start = h_out * stride;
    const int w_in_start = w_out * stride;

    float partial_sum = 0.0f;

    // Each thread processes a subset of channels
    for (int c = tid; c < C; c += block_size) {
        float max_val = -FLT_MAX;

        // Find the max value in the 3D window for the current channel 'c'
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int d_in = d_in_start + kd;
                    const int h_in = h_in_start + kh;
                    const int w_in = w_in_start + kw;

                    long long input_idx = (long long)n * C * D_in * H_in * W_in +
                                          (long long)c * D_in * H_in * W_in +
                                          (long long)d_in * H_in * W_in +
                                          (long long)h_in * W_in +
                                          w_in;
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }
        partial_sum += max_val;
    }

    // In-block reduction using shared memory to sum up partial sums from all threads
    extern __shared__ float sdata[];
    sdata[tid] = partial_sum;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 of the block writes the final result to the output tensor
    if (tid == 0) {
        output[block_idx] = sdata[0];
    }
}

// C++ wrapper for the 3D Max Pooling kernel
torch::Tensor max_pool3d_cuda(torch::Tensor input, int kernel_size, int stride) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(input.dim() == 5, "Input must be a 5D tensor (N, C, D, H, W)");
    input = input.contiguous();

    const int N = input.size(0);
    const int C = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    const int D_out = (D_in - kernel_size) / stride + 1;
    const int H_out = (H_in - kernel_size) / stride + 1;
    const int W_out = (W_in - kernel_size) / stride + 1;

    auto output = torch::zeros({N, C, D_out, H_out, W_out}, input.options());

    const long long total_threads = (long long)N * C * D_out * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;

    max_pool3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, D_in, H_in, W_in, D_out, H_out, W_out,
        kernel_size, stride);

    return output;
}

// C++ wrapper for the fused 3D Max Pooling and Summation kernel
torch::Tensor max_pool3d_sum_dim1_cuda(torch::Tensor input, int kernel_size, int stride) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(input.dim() == 5, "Input must be a 5D tensor (N, C, D, H, W)");
    input = input.contiguous();

    const int N = input.size(0);
    const int C = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    const int D_out = (D_in - kernel_size) / stride + 1;
    const int H_out = (H_in - kernel_size) / stride + 1;
    const int W_out = (W_in - kernel_size) / stride + 1;

    auto output = torch::zeros({N, 1, D_out, H_out, W_out}, input.options());

    const long long num_blocks = (long long)N * D_out * H_out * W_out;
    const int block_size = 256; // Threads per block for reduction over C
    const size_t shared_mem_size = block_size * sizeof(float);

    max_pool3d_sum_dim1_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, D_in, H_in, W_in, D_out, H_out, W_out,
        kernel_size, stride);

    return output;
}
"""

# C++ source for function declarations
cpp_source = """
torch::Tensor max_pool3d_cuda(torch::Tensor input, int kernel_size, int stride);
torch::Tensor max_pool3d_sum_dim1_cuda(torch::Tensor input, int kernel_size, int stride);
"""

# Compile the inline CUDA code
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["max_pool3d_cuda", "max_pool3d_sum_dim1_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that replaces the two max pooling layers and the sum operation
    with custom CUDA kernels. The second max pooling and the sum are fused into a
    single kernel to improve memory bandwidth efficiency.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        # The ConvTranspose3d layer is complex and its PyTorch implementation is
        # highly optimized (using cuDNN), so we keep it as is.
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

        # Parameters for the custom pooling operations.
        # nn.MaxPool3d defaults to stride=kernel_size.
        self.max_pool1_ks = 2
        self.max_pool1_stride = 2
        self.max_pool2_ks = 3
        self.max_pool2_stride = 3

    def forward(self, x):
        # 1. Apply the standard ConvTranspose3d layer
        x = self.conv_transpose(x)
        
        # 2. Apply the custom CUDA kernel for the first MaxPool3d
        x = custom_ops.max_pool3d_cuda(x, self.max_pool1_ks, self.max_pool1_stride)
        
        # 3. Apply the custom fused CUDA kernel for the second MaxPool3d and the sum operation
        x = custom_ops.max_pool3d_sum_dim1_cuda(x, self.max_pool2_ks, self.max_pool2_stride)
        
        return x