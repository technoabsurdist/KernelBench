import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for Fused Scale + BatchNorm
fused_scale_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel to calculate sum and sum of squares for each feature across the batch
// This is the first pass for calculating batch statistics (mean and variance)
__global__ void calc_stats_kernel(
    const float* __restrict__ x,      // Input tensor [B, F]
    const float* __restrict__ scale,  // Scale tensor [F]
    float* __restrict__ sums,         // Output sums [F]
    float* __restrict__ sum_sqs,      // Output sum of squares [F]
    int B, int F)
{
    // Each block processes one feature column
    int j = blockIdx.x;
    if (j >= F) return;

    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sum_sq = &sdata[blockDim.x];

    int tid = threadIdx.x;
    s_sum[tid] = 0.0f;
    s_sum_sq[tid] = 0.0f;

    float feature_scale = scale[j];

    // Each thread sums up a portion of the column
    for (int i = tid; i < B; i += blockDim.x) {
        float val = x[i * F + j] * feature_scale;
        s_sum[tid] += val;
        s_sum_sq[tid] += val * val;
    }
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final result for this feature
    if (tid == 0) {
        sums[j] = s_sum[0];
        sum_sqs[j] = s_sum_sq[0];
    }
}

// Kernel to update the running mean and variance buffers
__global__ void update_running_stats_kernel(
    const float* __restrict__ sums,
    const float* __restrict__ sum_sqs,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    int B, int F,
    float momentum)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= F) return;

    float mean = sums[j] / B;
    float var = (sum_sqs[j] / B) - (mean * mean);

    running_mean[j] = (1.0f - momentum) * running_mean[j] + momentum * mean;
    running_var[j] = (1.0f - momentum) * running_var[j] + momentum * var;
}

// Kernel to apply normalization using batch statistics (for training)
__global__ void apply_batch_normalization_kernel(
    const float* __restrict__ x,
    const float* __restrict__ scale,
    const float* __restrict__ sums,
    const float* __restrict__ sum_sqs,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int B, int F,
    float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * F) return;

    int j = idx % F; // feature index

    float mean = sums[j] / B;
    float var = (sum_sqs[j] / B) - (mean * mean);

    float val = x[idx] * scale[j];
    float inv_std = rsqrtf(var + eps);
    float normalized_val = (val - mean) * inv_std;

    y[idx] = gamma[j] * normalized_val + beta[j];
}

// Kernel to apply normalization using running statistics (for inference)
__global__ void apply_inference_normalization_kernel(
    const float* __restrict__ x,
    const float* __restrict__ scale,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int B, int F,
    float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * F) return;

    int j = idx % F; // feature index

    float mean = running_mean[j];
    float var = running_var[j];

    float val = x[idx] * scale[j];
    float inv_std = rsqrtf(var + eps);
    float normalized_val = (val - mean) * inv_std;

    y[idx] = gamma[j] * normalized_val + beta[j];
}


// C++ wrapper function that dispatches the appropriate kernels
torch::Tensor fused_scale_batchnorm_forward(
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    bool training,
    double momentum,
    double eps)
{
    const int B = x.size(0);
    const int F = x.size(1);
    auto y = torch::empty_like(x);

    if (training) {
        auto sums = torch::empty({F}, x.options());
        auto sum_sqs = torch::empty({F}, x.options());

        // Kernel 1: Calculate stats
        const int block_size_stats = 512;
        const int num_blocks_stats = F;
        const size_t shared_mem_size = 2 * block_size_stats * sizeof(float);
        calc_stats_kernel<<<num_blocks_stats, block_size_stats, shared_mem_size>>>(
            x.data_ptr<float>(), scale.data_ptr<float>(),
            sums.data_ptr<float>(), sum_sqs.data_ptr<float>(),
            B, F);

        // Kernel 2: Update running stats
        const int block_size_update = 256;
        const int num_blocks_update = (F + block_size_update - 1) / block_size_update;
        update_running_stats_kernel<<<num_blocks_update, block_size_update>>>(
            sums.data_ptr<float>(), sum_sqs.data_ptr<float>(),
            bn_running_mean.data_ptr<float>(), bn_running_var.data_ptr<float>(),
            B, F, static_cast<float>(momentum));

        // Kernel 3: Apply normalization using batch stats
        const int block_size_norm = 256;
        const int num_blocks_norm = (B * F + block_size_norm - 1) / block_size_norm;
        apply_batch_normalization_kernel<<<num_blocks_norm, block_size_norm>>>(
            x.data_ptr<float>(), scale.data_ptr<float>(),
            sums.data_ptr<float>(), sum_sqs.data_ptr<float>(),
            bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
            y.data_ptr<float>(),
            B, F, static_cast<float>(eps));
    } else { // Inference mode
        // Kernel 4: Apply normalization using running stats
        const int block_size_inf = 256;
        const int num_blocks_inf = (B * F + block_size_inf - 1) / block_size_inf;
        apply_inference_normalization_kernel<<<num_blocks_inf, block_size_inf>>>(
            x.data_ptr<float>(), scale.data_ptr<float>(),
            bn_running_mean.data_ptr<float>(), bn_running_var.data_ptr<float>(),
            bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
            y.data_ptr<float>(),
            B, F, static_cast<float>(eps));
    }
    return y;
}
"""

fused_scale_bn_cpp_source = """
torch::Tensor fused_scale_batchnorm_forward(
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    bool training,
    double momentum,
    double eps);
"""

# Compile the inline CUDA code
fused_scale_bn = load_inline(
    name="fused_scale_bn",
    cpp_sources=fused_scale_bn_cpp_source,
    cuda_sources=fused_scale_bn_source,
    functions=["fused_scale_batchnorm_forward"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses the scaling and batch normalization operations
    into a custom CUDA kernel.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # Standard GEMM layer, difficult to beat cuBLAS
        self.gemm = nn.Linear(in_features, out_features)
        
        # Custom scaling parameter
        self.scale = nn.Parameter(torch.randn(scale_shape))
        
        # BatchNorm1d layer is kept to hold parameters (weight, bias) and
        # state (running_mean, running_var) which are managed by our custom kernel.
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        
        # The compiled custom operator
        self.fused_op = fused_scale_bn

    def forward(self, x):
        # 1. Perform GEMM
        x = self.gemm(x)
        
        # 2. Call the fused kernel for scale and batchnorm
        # The custom op needs to know if we are in training or eval mode.
        # self.training is a boolean attribute of nn.Module that handles this.
        x = self.fused_op.fused_scale_batchnorm_forward(
            x,
            self.scale,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.training,
            self.bn.momentum,
            self.bn.eps
        )
        return x