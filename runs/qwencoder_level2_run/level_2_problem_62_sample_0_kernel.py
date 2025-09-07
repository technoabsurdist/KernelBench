import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + group norm + leaky relu + elementwise sum
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

__global__ void fused_gn_lrelu_kernel(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    int batch_size,
    int num_channels,
    int group_size,
    int elements_per_channel,
    float eps,
    float negative_slope
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_channels * elements_per_channel;
    
    if (idx < total_elements) {
        int batch_idx = idx / (num_channels * elements_per_channel);
        int channel_idx = (idx % (num_channels * elements_per_channel)) / elements_per_channel;
        int element_idx = idx % elements_per_channel;
        
        // Group normalization calculation
        int group_idx = channel_idx / group_size;
        int within_group_idx = channel_idx % group_size;
        
        // Calculate mean and variance for the group
        float sum = 0.0f;
        float sum_sq = 0.0f;
        int group_start = group_idx * group_size;
        int group_end = (group_idx + 1) * group_size;
        
        for (int c = group_start; c < group_end; c++) {
            for (int e = 0; e < elements_per_channel; e++) {
                int input_idx = batch_idx * (num_channels * elements_per_channel) + 
                               c * elements_per_channel + e;
                float val = input[input_idx];
                sum += val;
                sum_sq += val * val;
            }
        }
        
        int group_elements = group_size * elements_per_channel;
        float mean = sum / group_elements;
        float var = (sum_sq / group_elements) - (mean * mean);
        float inv_std = rsqrtf(var + eps);
        
        // Normalize, scale, and shift
        float normalized = (input[idx] - mean) * inv_std;
        float scaled = normalized * weight[channel_idx] + bias[channel_idx];
        
        // Leaky ReLU
        float activated = scaled > 0 ? scaled : scaled * negative_slope;
        
        // Element-wise sum (x + x = 2*x)
        output[idx] = activated + activated;
    }
}

torch::Tensor fused_matmul_gn_lrelu_sum(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int num_groups,
    float eps,
    float negative_slope
) {
    // Get dimensions
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    int batch_size = input_sizes[0];
    int input_features = input_sizes[1];
    int output_features = weight_sizes[0];
    
    // Perform matrix multiplication
    auto matmul_result = torch::matmul(input, weight.t());
    
    // Apply fused group norm + leaky relu + sum
    auto output = torch::zeros_like(matmul_result);
    
    int num_channels = output_features;
    int elements_per_channel = 1;
    int group_size = num_channels / num_groups;
    
    const int block_size = 256;
    int total_elements = batch_size * num_channels * elements_per_channel;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_gn_lrelu_kernel<<<num_blocks, block_size>>>(
        matmul_result.data_ptr<float>(),
        output.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        batch_size,
        num_channels,
        group_size,
        elements_per_channel,
        eps,
        negative_slope
    );
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_matmul_gn_lrelu_sum(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int num_groups,
    float eps,
    float negative_slope
);
"""

# Compile the inline CUDA code
fused_module = load_inline(
    name="fused_matmul_gn_lrelu_sum",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_matmul_gn_lrelu_sum"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernels.
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope
        
        # Linear layer parameters
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))
        
        # GroupNorm parameters
        self.gn_weight = nn.Parameter(torch.ones(hidden_size))
        self.gn_bias = nn.Parameter(torch.zeros(hidden_size))
        
        # Load the fused CUDA module
        self.fused_op = fused_module

    def forward(self, x):
        """
        Performs the forward pass with fused operations.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        return self.fused_op.fused_matmul_gn_lrelu_sum(
            x,
            self.weight,
            self.bias,
            self.gn_weight,
            self.gn_bias,
            self.num_groups,
            self.eps,
            self.negative_slope
        )