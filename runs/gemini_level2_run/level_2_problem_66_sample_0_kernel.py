import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for a fused (Linear + Dropout + Softmax) operation
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <limits>

// Using a fixed block size for simplicity. A common choice for high occupancy.
constexpr int THREADS_PER_BLOCK = 1024;

// Functors for reduction operations
struct MaxOp {
    __device__ float operator()(float a, float b) const { return max(a, b); }
};

struct SumOp {
    __device__ float operator()(float a, float b) const { return a + b; }
};

// Helper for a parallel block-wide reduction
template <typename T, typename Op>
__device__ T block_reduce(T val, Op op, T* s_reduce) {
    int tid = threadIdx.x;
    s_reduce[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_reduce[tid] = op(s_reduce[tid], s_reduce[tid + s]);
        }
        __syncthreads();
    }
    return s_reduce[0];
}

__global__ void fused_linear_dropout_softmax_kernel(
    const float* __restrict__ x,
    const float* __restrict__ W_t, // Weight must be pre-transposed: [in_features, out_features]
    const float* __restrict__ b,
    float* __restrict__ out,
    const int batch_size,
    const int in_features,
    const int out_features,
    const float p,
    const bool is_training,
    const uint64_t seed,
    const uint64_t offset)
{
    // Each block is responsible for computing one full row of the output matrix.
    const int row_idx = blockIdx.x;
    if (row_idx >= batch_size) return;

    const int tid = threadIdx.x;

    // Use dynamic shared memory. s_logits stores the intermediate results for the row.
    // s_reduce is a temporary buffer for the reduction operations.
    extern __shared__ float s_mem[];
    float* s_logits = s_mem;
    float* s_reduce = &s_mem[out_features];

    const float* x_row = x + row_idx * in_features;
    float* out_row = out + row_idx * out_features;

    // Step 1: Fused Matmul + Bias + Dropout
    // Each thread computes a subset of the output features for the current row.
    for (int col = tid; col < out_features; col += THREADS_PER_BLOCK) {
        float dot = 0.0f;
        // Compute dot product: x_row[k] * W_t[k, col]
        for (int k = 0; k < in_features; ++k) {
            dot += x_row[k] * W_t[k * out_features + col];
        }
        // Add bias
        float logit = dot + b[col];

        // Apply dropout if in training mode.
        // For pre-softmax dropout, we set dropped elements to -inf.
        if (is_training && p > 0.0f) {
            curandState_t state;
            // Initialize cuRAND state. Each element gets a unique seed.
            curand_init(seed, row_idx * out_features + col, offset, &state);
            float rand_val = curand_uniform(&state);
            if (rand_val < p) {
                logit = -std::numeric_limits<float>::infinity();
            }
            // Note: Scaling (inverted dropout) is omitted as it's not standard
            // for dropout applied directly before softmax.
        }
        s_logits[col] = logit;
    }
    __syncthreads(); // Ensure all threads have computed their logits for the row.

    // Step 2: Softmax calculated in-place in shared memory
    // Part A: Find the maximum value in the row for numerical stability.
    float thread_max = -std::numeric_limits<float>::infinity();
    for (int col = tid; col < out_features; col += THREADS_PER_BLOCK) {
        thread_max = max(thread_max, s_logits[col]);
    }
    float row_max = block_reduce(thread_max, MaxOp(), s_reduce);
    __syncthreads();

    // Part B: Calculate the sum of exponentials.
    float thread_sum = 0.0f;
    for (int col = tid; col < out_features; col += THREADS_PER_BLOCK) {
        thread_sum += expf(s_logits[col] - row_max);
    }
    float row_sum = block_reduce(thread_sum, SumOp(), s_reduce);
    __syncthreads();

    // Part C: Final division and write to global output memory.
    // Add a small epsilon to prevent division by zero if all inputs were dropped.
    row_sum += 1e-6;
    for (int col = tid; col < out_features; col += THREADS_PER_BLOCK) {
        out_row[col] = expf(s_logits[col] - row_max) / row_sum;
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor x,
    torch::Tensor W_t, // Pre-transposed weight
    torch::Tensor b,
    double p,
    bool is_training)
{
    const auto batch_size = x.size(0);
    const auto in_features = x.size(1);
    const auto out_features = W_t.size(1);

    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(W_t.is_cuda(), "Input W_t must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "Input b must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input x must be 2D");
    TORCH_CHECK(W_t.dim() == 2, "Input W_t must be 2D");
    TORCH_CHECK(b.dim() == 1, "Input b must be 1D");
    TORCH_CHECK(x.size(1) == W_t.size(0), "Dimension mismatch: x.size(1) != W_t.size(0)");
    TORCH_CHECK(b.size(0) == out_features, "Dimension mismatch: b.size(0) != out_features");

    auto out = torch::empty({batch_size, out_features}, x.options());

    // Use a simple static counter for the cuRAND offset to ensure
    // different random numbers across different forward passes.
    static uint64_t offset_counter = 0;

    const int threads = THREADS_PER_BLOCK;
    const int blocks = batch_size;

    // Calculate the required dynamic shared memory size per block.
    size_t shared_mem_size = (out_features + threads) * sizeof(float);

    // Request sufficient dynamic shared memory for the kernel launch.
    cudaFuncSetAttribute(fused_linear_dropout_softmax_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shared_mem_size);

    // Launch the fused kernel.
    fused_linear_dropout_softmax_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        W_t.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        static_cast<float>(p),
        is_training,
        12345, // Fixed seed for reproducibility
        offset_counter++
    );

    return out;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor x,
    torch::Tensor W_t,
    torch::Tensor b,
    double p,
    bool is_training);
"""

# JIT compile the CUDA and C++ code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_op_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    An optimized model that fuses matrix multiplication, dropout, and softmax
    into a single custom CUDA kernel.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p

        # To maximize performance, we store the weight parameter pre-transposed.
        # This avoids a costly transpose operation in every forward pass.
        self.weight_t = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        self.reset_parameters()

        # Store the compiled custom operator
        self.fused_op = fused_op

    def reset_parameters(self) -> None:
        """
        Initialize parameters in a way that is consistent with nn.Linear.
        """
        # We initialize based on the "original" non-transposed shape [out, in]
        # by transposing our stored weight_t for the initialization function.
        nn.init.kaiming_uniform_(self.weight_t.t(), a=math.sqrt(5))

        # Calculate fan_in from the non-transposed shape for bias initialization
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_t.t())
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Call the custom CUDA kernel, passing the pre-transposed weight.
        # The kernel handles matmul, bias add, dropout, and softmax internally.
        return self.fused_op.fused_op_cuda(
            x,
            self.weight_t,
            self.bias,
            self.dropout_p,
            self.training  # Pass the model's training mode to the kernel
        )