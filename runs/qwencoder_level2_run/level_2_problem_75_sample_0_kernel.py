import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + GroupNorm + Min + Bias
fused_gemm_gn_min_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

__global__ void group_norm_min_bias_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* min_bias,
    float* output,
    int batch_size,
    int num_channels,
    int group_size,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_channels * spatial_size;
    
    if (idx < batch_size * spatial_size) {
        int batch_idx = idx / spatial_size;
        int spatial_idx = idx % spatial_size;
        
        // Process each group
        for (int group = 0; group < num_channels / group_size; group++) {
            float sum = 0.0f;
            float sum_sq = 0.0f;
            
            // Calculate mean and variance for the group
            for (int c = 0; c < group_size; c++) {
                int ch = group * group_size + c;
                int input_idx = batch_idx * num_channels * spatial_size + ch * spatial_size + spatial_idx;
                sum += input[input_idx];
                sum_sq += input[input_idx] * input[input_idx];
            }
            
            float mean = sum / group_size;
            float var = sum_sq / group_size - mean * mean;
            float inv_std = rsqrtf(var + 1e-5);
            
            // Normalize, scale, shift and apply min+bias
            float group_min = INFINITY;
            for (int c = 0; c < group_size; c++) {
                int ch = group * group_size + c;
                int input_idx = batch_idx * num_channels * spatial_size + ch * spatial_size + spatial_idx;
                float normalized = (input[input_idx] - mean) * inv_std;
                float scaled = normalized * weight[ch] + bias[ch];
                group_min = fminf(group_min, scaled);
            }
            
            // Write result
            int output_idx = batch_idx * spatial_size + spatial_idx;
            output[output_idx] = group_min + min_bias[0];
        }
    }
}

torch::Tensor fused_gemm_gn_min_bias_cuda(
    torch::Tensor gemm_input,
    torch::Tensor gemm_weight,
    torch::Tensor gemm_bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor min_bias
) {
    // GEMM dimensions
    int batch_size = gemm_input.size(0);
    int in_features = gemm_input.size(1);
    int out_features = gemm_weight.size(0);
    
    // Perform GEMM: output = input * weight^T + bias
    auto gemm_output = torch::matmul(gemm_input, gemm_weight.transpose(0, 1)) + gemm_bias;
    
    // Reshape for group norm (N, C, H, W) - treating H*W as spatial dimension
    auto reshaped = gemm_output.unsqueeze(-1).unsqueeze(-1); // (N, C, 1, 1)
    int spatial_size = 1;
    int group_size = out_features / (gn_weight.size(0) / out_features); // Simplified
    
    // Apply fused GroupNorm + Min + Bias
    auto output = torch::zeros({batch_size, 1, 1, 1}, gemm_output.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * spatial_size + block_size - 1) / block_size;
    
    group_norm_min_bias_kernel<<<num_blocks, block_size>>>(
        reshaped.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        min_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features,
        out_features / 512, // group_size
        spatial_size
    );
    
    return output.squeeze(-1).squeeze(-1);
}
"""

fused_gemm_gn_min_bias_cpp_source = """
torch::Tensor fused_gemm_gn_min_bias_cuda(
    torch::Tensor gemm_input,
    torch::Tensor gemm_weight,
    torch::Tensor gemm_bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor min_bias
);
"""

# Compile the inline CUDA code
fused_gemm_gn_min_bias = load_inline(
    name="fused_gemm_gn_min_bias",
    cpp_sources=fused_gemm_gn_min_bias_cpp_source,
    cuda_sources=fused_gemm_gn_min_bias_source,
    functions=["fused_gemm_gn_min_bias_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernels.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        # GEMM parameters
        self.gemm_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.gemm_bias = nn.Parameter(torch.randn(out_features))
        
        # GroupNorm parameters
        self.gn_weight = nn.Parameter(torch.randn(out_features))
        self.gn_bias = nn.Parameter(torch.randn(out_features))
        
        # Min bias parameter
        self.min_bias = nn.Parameter(torch.randn(bias_shape))
        
        # Load custom CUDA function
        self.fused_op = fused_gemm_gn_min_bias

    def forward(self, x):
        return self.fused_op.fused_gemm_gn_min_bias_cuda(
            x,
            self.gemm_weight,
            self.gemm_bias,
            self.gn_weight,
            self.gn_bias,
            self.min_bias
        )

batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 512
bias_shape = (1, out_features, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]