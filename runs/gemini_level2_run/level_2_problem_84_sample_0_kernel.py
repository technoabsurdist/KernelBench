import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused BatchNorm, scaling, and Softmax
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_bn_scale_softmax_kernel(
    const float* __restrict__ inp,          // (batch_size, features)
    float* __restrict__ out,                // (batch_size, features)
    const float* __restrict__ bn_weight,    // (features,)
    const float* __restrict__ bn_bias,      // (features,)
    const float* __restrict__ bn_mean,      // (features,)
    const float* __restrict__ bn_var,       // (features,)
    const float* __restrict__ scale,        // (1,) or broadcastable
    const float bn_eps,
    const int batch_size,
    const int features) {

    // One block per row in the batch
    const int row_idx = blockIdx.x;
    if (row_idx >= batch_size) return;

    // Shared memory to hold one row of transformed data
    // and for reductions.
    extern __shared__ float s_data[];

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int items_per_thread = (features + block_size - 1) / block_size;

    // --- 1. Load, apply BN and Scale, and store to shared memory ---
    for (int i = 0; i < items_per_thread; ++i) {
        int col_idx = tid + i * block_size;
        if (col_idx < features) {
            // Load input value
            float val = inp[row_idx * features + col_idx];

            // Pre-compute BN scale and bias for inference
            float inv_std = rsqrtf(bn_var[col_idx] + bn_eps);
            float bn_s = bn_weight[col_idx] * inv_std;
            float bn_b = bn_bias[col_idx] - bn_mean[col_idx] * bn_s;

            // Apply BN, then scale
            float transformed_val = (val * bn_s + bn_b) * (*scale);
            
            s_data[col_idx] = transformed_val;
        }
    }
    __syncthreads();

    // --- 2. Find max value in the row (reduction in shared memory) ---
    // Each thread finds max of its elements
    float thread_max = -INFINITY;
    for (int i = 0; i < items_per_thread; ++i) {
        int col_idx = tid + i * block_size;
        if (col_idx < features) {
            thread_max = fmaxf(thread_max, s_data[col_idx]);
        }
    }
    
    // Use scratch space at the end of s_data for reduction
    float* reduce_space = s_data + features;
    reduce_space[tid] = thread_max;
    __syncthreads();

    // Reduce to find the row max
    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            reduce_space[tid] = fmaxf(reduce_space[tid], reduce_space[tid + s]);
        }
        __syncthreads();
    }
    const float row_max = reduce_space[0];

    // --- 3. Compute exp(x - max) and sum (reduction) ---
    // Each thread computes sum of its elements
    float thread_sum = 0.0f;
    for (int i = 0; i < items_per_thread; ++i) {
        int col_idx = tid + i * block_size;
        if (col_idx < features) {
            float val = expf(s_data[col_idx] - row_max);
            s_data[col_idx] = val; // Store intermediate result for final step
            thread_sum += val;
        }
    }
    reduce_space[tid] = thread_sum;
    __syncthreads();

    // Reduce to find the row sum
    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            reduce_space[tid] += reduce_space[tid + s];
        }
        __syncthreads();
    }
    const float row_sum = reduce_space[0];
    // Add a small epsilon to avoid division by zero
    const float inv_row_sum = 1.0f / (row_sum + 1e-12f);

    // --- 4. Normalize and write to output ---
    for (int i = 0; i < items_per_thread; ++i) {
        int col_idx = tid + i * block_size;
        if (col_idx < features) {
            out[row_idx * features + col_idx] = s_data[col_idx] * inv_row_sum;
        }
    }
}

torch::Tensor fused_bn_scale_softmax_forward(
    torch::Tensor inp,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    torch::Tensor scale,
    double bn_eps) {

    const auto batch_size = inp.size(0);
    const auto features = inp.size(1);

    TORCH_CHECK(inp.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(inp.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(bn_weight.is_contiguous(), "bn_weight must be contiguous");
    TORCH_CHECK(bn_bias.is_contiguous(), "bn_bias must be contiguous");
    TORCH_CHECK(bn_mean.is_contiguous(), "bn_mean must be contiguous");
    TORCH_CHECK(bn_var.is_contiguous(), "bn_var must be contiguous");
    TORCH_CHECK(scale.is_contiguous(), "scale must be contiguous");

    auto out = torch::empty_like(inp);

    const int block_size = 1024;
    const dim3 grid_size(batch_size);
    const dim3 block_dim(block_size);
    
    // Shared memory: features for the row data + block_size for reduction scratchpad
    const int shared_mem_size = (features + block_size) * sizeof(float);

    fused_bn_scale_softmax_kernel<<<grid_size, block_dim, shared_mem_size>>>(
        inp.data_ptr<float>(),
        out.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        scale.data_ptr<float>(),
        (float)bn_eps,
        batch_size,
        features
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return out;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_bn_scale_softmax_forward(
    torch::Tensor inp,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    torch::Tensor scale,
    double bn_eps);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_bn_scale_softmax_forward"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses BatchNorm, scaling, and Softmax into a single CUDA kernel.
    The nn.Linear layer is kept as is, leveraging the highly optimized cuBLAS library.
    This model is designed for inference mode.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        
        # Store the parameters and buffers for the fused kernel.
        # This ensures they are managed by the PyTorch module (e.g., moved to the correct device).
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('bn_running_mean', torch.zeros(out_features))
        self.register_buffer('bn_running_var', torch.ones(out_features))
        self.bn_eps = bn_eps
        
        self.scale = nn.Parameter(torch.ones(scale_shape))
        
        # This custom module implements the inference-time behavior of BatchNorm.
        self.eval()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # 1. Matrix Multiplication (using PyTorch's optimized nn.Linear)
        x = self.gemm(x)
        
        # 2. Fused BatchNorm + Scaling + Softmax
        x = fused_op.fused_bn_scale_softmax_forward(
            x,
            self.bn_weight,
            self.bn_bias,
            self.bn_running_mean,
            self.bn_running_var,
            self.scale,
            self.bn_eps
        )
        return x