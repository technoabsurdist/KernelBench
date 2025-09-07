import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_conv_gn_tanh_hardswish_residual_lse_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void group_norm_kernel(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    const int N,
    const int C,
    const int HxW,
    const int groups,
    const int group_size,
    const float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * HxW) return;
    
    int n = idx / (C * HxW);
    int c = (idx / HxW) % C;
    int hw = idx % HxW;
    
    int group_id = c / group_size;
    int group_start = group_id * group_size * HxW;
    int group_end = (group_id + 1) * group_size * HxW;
    
    // Compute mean
    float sum = 0.0f;
    for (int i = group_start + n * C * HxW; i < group_start + group_size * HxW + n * C * HxW; i += HxW) {
        for (int j = 0; j < HxW && i + j < group_end + n * C * HxW; ++j) {
            sum += input[i + j];
        }
    }
    float mean = sum / (group_size * HxW);
    
    // Compute variance
    float sum_sq = 0.0f;
    for (int i = group_start + n * C * HxW; i < group_start + group_size * HxW + n * C * HxW; i += HxW) {
        for (int j = 0; j < HxW && i + j < group_end + n * C * HxW; ++j) {
            float diff = input[i + j] - mean;
            sum_sq += diff * diff;
        }
    }
    float var = sum_sq / (group_size * HxW);
    float inv_std = rsqrtf(var + eps);
    
    // Normalize, scale and shift
    float norm_val = (input[idx] - mean) * inv_std;
    output[idx] = norm_val * weight[c] + bias[c];
}

__global__ void tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

__global__ void hardswish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float relu6_val = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        output[idx] = x * relu6_val / 6.0f;
    }
}

__global__ void residual_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void logsumexp_kernel(const float* input, float* output, int N, int C, int HxW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_hw = HxW;
    if (idx >= N * total_hw) return;
    
    int n = idx / total_hw;
    int hw = idx % total_hw;
    
    // Find max for numerical stability
    float max_val = input[n * C * HxW + hw];
    for (int c = 1; c < C; ++c) {
        max_val = fmaxf(max_val, input[n * C * HxW + c * HxW + hw]);
    }
    
    // Compute sum of exp
    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        sum += expf(input[n * C * HxW + c * HxW + hw] - max_val);
    }
    
    output[idx] = logf(sum) + max_val;
}

torch::Tensor fused_conv_gn_tanh_hardswish_residual_lse(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int groups,
    float eps
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(gn_weight);
    CHECK_INPUT(gn_bias);
    
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto H_in = input.size(2);
    auto W_in = input.size(3);
    
    auto K = weight.size(0);
    auto C_out = K;
    auto kH = weight.size(2);
    auto kW = weight.size(3);
    
    auto H_out = H_in - kH + 1;
    auto W_out = W_in - kW + 1;
    
    // Convolution using cuDNN (simplified - in practice, you'd implement your own conv)
    auto conv_out = torch::conv2d(input, weight, bias, 1, 0, 1, 1);
    
    auto conv_size = conv_out.numel();
    auto conv_data = conv_out.data_ptr<float>();
    
    // Group Norm
    auto gn_out = torch::zeros_like(conv_out);
    auto gn_data = gn_out.data_ptr<float>();
    
    const int block_size = 256;
    const int num_blocks_gn = (conv_size + block_size - 1) / block_size;
    
    int group_size = C_out / groups;
    group_norm_kernel<<<num_blocks_gn, block_size>>>(
        conv_data,
        gn_data,
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        N,
        C_out,
        H_out * W_out,
        groups,
        group_size,
        eps
    );
    
    // Tanh
    auto tanh_out = torch::zeros_like(gn_out);
    auto tanh_data = tanh_out.data_ptr<float>();
    const int num_blocks_tanh = (conv_size + block_size - 1) / block_size;
    tanh_kernel<<<num_blocks_tanh, block_size>>>(gn_data, tanh_data, conv_size);
    
    // HardSwish
    auto hardswish_out = torch::zeros_like(tanh_out);
    auto hardswish_data = hardswish_out.data_ptr<float>();
    const int num_blocks_hardswish = (conv_size + block_size - 1) / block_size;
    hardswish_kernel<<<num_blocks_hardswish, block_size>>>(tanh_data, hardswish_data, conv_size);
    
    // Residual Addition
    auto residual_out = torch::zeros_like(conv_out);
    auto residual_data = residual_out.data_ptr<float>();
    const int num_blocks_residual = (conv_size + block_size - 1) / block_size;
    residual_add_kernel<<<num_blocks_residual, block_size>>>(conv_data, hardswish_data, residual_data, conv_size);
    
    // LogSumExp
    auto lse_out = torch::zeros({N, 1, H_out, W_out}, torch::kFloat32);
    auto lse_data = lse_out.data_ptr<float>();
    const int lse_size = N * H_out * W_out;
    const int num_blocks_lse = (lse_size + block_size - 1) / block_size;
    logsumexp_kernel<<<num_blocks_lse, block_size>>>(residual_data, lse_data, N, C_out, H_out * W_out);
    
    return lse_out;
}
"""

fused_conv_gn_tanh_hardswish_residual_lse_cpp_source = """
torch::Tensor fused_conv_gn_tanh_hardswish_residual_lse(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int groups,
    float eps
);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_conv_gn_tanh_hardswish_residual_lse",
    cpp_sources=fused_conv_gn_tanh_hardswish_residual_lse_cpp_source,
    cuda_sources=fused_conv_gn_tanh_hardswish_residual_lse_source,
    functions=["fused_conv_gn_tanh_hardswish_residual_lse"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.eps = eps
        
        # Convolution parameters
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.randn(out_channels))
        
        # GroupNorm parameters
        self.gn_weight = nn.Parameter(torch.ones(out_channels))
        self.gn_bias = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.conv_weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        return fused_op.fused_conv_gn_tanh_hardswish_residual_lse(
            x,
            self.conv_weight,
            self.conv_bias,
            self.gn_weight,
            self.gn_bias,
            self.groups,
            self.eps
        )