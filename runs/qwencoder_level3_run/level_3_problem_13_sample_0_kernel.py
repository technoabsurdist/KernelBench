import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused BatchNorm + ReLU + Conv + AvgPool
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

__global__ void batch_norm_relu_conv_pool_kernel(
    const float* input,
    const float* conv_weight,
    const float* bn_weight,
    const float* bn_bias,
    const float* bn_running_mean,
    const float* bn_running_var,
    float* output,
    int batch_size,
    int input_channels,
    int output_channels,
    int height,
    int width,
    float eps
) {
    // Each thread block handles one output pixel
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y;
    int out_hw = blockIdx.z;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || out_ch >= output_channels || out_hw >= (height/2)*(width/2)) 
        return;
    
    int out_h = out_hw / (width/2);
    int out_w = out_hw % (width/2);
    
    // Shared memory for partial sums
    extern __shared__ float shared_mem[];
    float* shared_sum = shared_mem;
    
    // Compute convolution for this output pixel
    float sum = 0.0f;
    
    for (int in_ch = tid; in_ch < input_channels; in_ch += blockDim.x) {
        // For 1x1 convolution, we only need one input pixel per output pixel
        int in_h = out_h * 2;
        int in_w = out_w * 2;
        
        if (in_h < height && in_w < width) {
            int input_idx = batch_idx * (input_channels * height * width) + 
                           in_ch * (height * width) + 
                           in_h * width + in_w;
                           
            int weight_idx = out_ch * input_channels + in_ch;
            
            // Apply batch norm to input
            float x = input[input_idx];
            float mean = bn_running_mean[in_ch];
            float var = bn_running_var[in_ch];
            float gamma = bn_weight[in_ch];
            float beta = bn_bias[in_ch];
            
            float normalized = (x - mean) / sqrtf(var + eps);
            float bn_out = gamma * normalized + beta;
            
            // Apply ReLU
            float relu_out = fmaxf(0.0f, bn_out);
            
            // Convolution (1x1)
            sum += relu_out * conv_weight[weight_idx];
        }
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        int output_idx = batch_idx * (output_channels * (height/2) * (width/2)) +
                        out_ch * ((height/2) * (width/2)) +
                        out_h * (width/2) + out_w;
        output[output_idx] = shared_sum[0];
    }
}

torch::Tensor fused_bn_relu_conv_pool_cuda(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var
) {
    auto batch_size = input.size(0);
    auto input_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto output_channels = conv_weight.size(0);
    
    auto output = torch::zeros({batch_size, output_channels, height/2, width/2}, 
                              torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    const int block_size = 256;
    dim3 grid_size(batch_size, output_channels, (height/2)*(width/2));
    
    float eps = 1e-5;
    
    size_t shared_mem_size = block_size * sizeof(float);
    
    batch_norm_relu_conv_pool_kernel<<<grid_size, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_running_mean.data_ptr<float>(),
        bn_running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_channels,
        output_channels,
        height,
        width,
        eps
    );
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_bn_relu_conv_pool_cuda(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var
);
"""

# Compile the inline CUDA code
fused_kernel = load_inline(
    name="fused_bn_relu_conv_pool",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_bn_relu_conv_pool_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(ModelNew, self).__init__()
        
        # BatchNorm parameters
        self.bn_weight = nn.Parameter(torch.ones(num_input_features))
        self.bn_bias = nn.Parameter(torch.zeros(num_input_features))
        self.bn_running_mean = nn.Parameter(torch.zeros(num_input_features), requires_grad=False)
        self.bn_running_var = nn.Parameter(torch.ones(num_input_features), requires_grad=False)
        
        # Conv parameters
        self.conv_weight = nn.Parameter(torch.randn(num_output_features, num_input_features, 1, 1) * 0.1)
        
        # Register the custom kernel
        self.fused_kernel = fused_kernel
        
    def forward(self, x):
        return self.fused_kernel.fused_bn_relu_conv_pool_cuda(
            x,
            self.conv_weight.squeeze(-1).squeeze(-1),  # Remove spatial dimensions for 1x1 conv
            self.bn_weight,
            self.bn_bias,
            self.bn_running_mean,
            self.bn_running_var
        )