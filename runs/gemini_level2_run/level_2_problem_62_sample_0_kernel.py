import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GroupNorm + LeakyReLU + Add
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// CUDA kernel for fused GroupNorm -> LeakyReLU -> x*2
__global__ void fused_groupnorm_leaky_relu_add_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N,
    int C,
    int G,
    int D,
    float eps,
    float negative_slope) {

    // Shared memory for reduction
    extern __shared__ float s_mem[];

    // Identify which group this block is processing
    const int group_idx = blockIdx.x;
    const int n = group_idx / G; // Batch index
    const int g = group_idx % G; // Group index

    // --- 1. Calculate Mean ---
    float sum = 0.0f;
    // Each thread sums up a portion of the D elements in the group
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const int c_offset = g * D + i;
        const int idx = n * C + c_offset;
        sum += in[idx];
    }
    s_mem[threadIdx.x] = sum;
    __syncthreads();

    // Parallel reduction in shared memory to get the total sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_mem[threadIdx.x] += s_mem[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 computes the mean and writes it to shared memory for broadcasting
    if (threadIdx.x == 0) {
        s_mem[0] = s_mem[0] / D;
    }
    __syncthreads();
    const float mean = s_mem[0];

    // --- 2. Calculate Variance ---
    float sum_sq_diff = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const int c_offset = g * D + i;
        const int idx = n * C + c_offset;
        const float diff = in[idx] - mean;
        sum_sq_diff += diff * diff;
    }
    s_mem[threadIdx.x] = sum_sq_diff;
    __syncthreads();

    // Parallel reduction for sum of squared differences
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_mem[threadIdx.x] += s_mem[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 computes inverse standard deviation and broadcasts
    if (threadIdx.x == 0) {
        const float variance = s_mem[0] / D;
        s_mem[0] = rsqrtf(variance + eps);
    }
    __syncthreads();
    const float inv_std_dev = s_mem[0];

    // --- 3. Apply Normalization, Activation, and Final Operation ---
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const int c_offset = g * D + i;
        const int idx = n * C + c_offset;

        // Group Normalization
        const float normalized = (in[idx] - mean) * inv_std_dev;
        const float scaled = normalized * weight[c_offset] + bias[c_offset];

        // Leaky ReLU
        const float activated = (scaled > 0) ? scaled : scaled * negative_slope;

        // Element-wise sum (x + x is equivalent to x * 2)
        out[idx] = activated * 2.0f;
    }
}

// C++ wrapper function to be called from Python
torch::Tensor fused_groupnorm_leaky_relu_add_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    double eps,
    double negative_slope) {

    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor 'x' must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Input tensor 'weight' must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Input tensor 'bias' must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor 'x' must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Input tensor 'weight' must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Input tensor 'bias' must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor 'x' must be of type float32");

    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto G = num_groups;
    TORCH_CHECK(C % G == 0, "Number of channels must be divisible by number of groups");
    const auto D = C / G;

    auto out = torch::empty_like(x);

    // Kernel launch configuration
    const int block_size = 256; // A common choice, good for occupancy
    const dim3 grid_dim(N * G);
    const dim3 block_dim(block_size);
    const size_t shared_mem_size = block_size * sizeof(float);

    // Launch the fused kernel
    fused_groupnorm_leaky_relu_add_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, G, D,
        static_cast<float>(eps),
        static_cast<float>(negative_slope)
    );
    
    // Check for any CUDA errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_groupnorm_leaky_relu_add_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    double eps,
    double negative_slope);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_groupnorm_leaky_relu_add_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    An optimized model that fuses GroupNorm, LeakyReLU, and element-wise sum into a single CUDA kernel.
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        # The linear layer remains a standard, highly optimized PyTorch operator
        self.fc = nn.Linear(input_size, hidden_size)

        # We need to define the learnable parameters for GroupNorm ourselves,
        # as we are replacing the nn.GroupNorm layer.
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

        # Store constants needed by the CUDA kernel
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope

    def forward(self, x):
        """
        Performs the forward pass using the custom fused CUDA kernel.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        # 1. Apply the standard, optimized linear layer
        x = self.fc(x)

        # 2. Apply the custom fused kernel for GroupNorm -> LeakyReLU -> x + x
        x = fused_op.fused_groupnorm_leaky_relu_add_cuda(
            x,
            self.weight,
            self.bias,
            self.num_groups,
            self.eps,
            self.negative_slope
        )
        return x