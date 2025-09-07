import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv3d + scale + tanh + scale + sigmoid
fused_conv3d_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

__global__ void fused_conv3d_activation_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* scaling_factor,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_size,
    int output_depth,
    int output_height,
    int output_width
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (idx >= total_elements) return;
    
    // Decode output position
    int temp = idx;
    int out_w = temp % output_width;
    temp /= output_width;
    int out_h = temp % output_height;
    temp /= output_height;
    int out_d = temp % output_depth;
    temp /= output_depth;
    int out_c = temp % out_channels;
    int batch = temp / out_channels;
    
    // Convolution computation
    float sum = 0.0f;
    
    for (int in_c = 0; in_c < in_channels; in_c++) {
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int in_d = out_d + kd - kernel_size/2;
                    int in_h = out_h + kh - kernel_size/2;
                    int in_w = out_w + kw - kernel_size/2;
                    
                    if (in_d >= 0 && in_d < input_depth &&
                        in_h >= 0 && in_h < input_height &&
                        in_w >= 0 && in_w < input_width) {
                        
                        int input_idx = batch * (in_channels * input_depth * input_height * input_width) +
                                       in_c * (input_depth * input_height * input_width) +
                                       in_d * (input_height * input_width) +
                                       in_h * input_width +
                                       in_w;
                                       
                        int weight_idx = out_c * (in_channels * kernel_size * kernel_size * kernel_size) +
                                        in_c * (kernel_size * kernel_size * kernel_size) +
                                        kd * (kernel_size * kernel_size) +
                                        kh * kernel_size +
                                        kw;
                                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Apply scaling factor, tanh, bias scaling, and sigmoid
    int bias_idx = out_c;
    float scaled = sum * scaling_factor[bias_idx];
    float tanh_val = tanhf(scaled);
    float bias_scaled = tanh_val * bias[bias_idx];
    float sigmoid_val = 1.0f / (1.0f + expf(-bias_scaled));
    
    output[idx] = sigmoid_val;
}

torch::Tensor fused_conv3d_activation_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scaling_factor,
    torch::Tensor bias
) {
    // Get tensor dimensions
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_depth = input.size(2);
    auto input_height = input.size(3);
    auto input_width = input.size(4);
    
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2); // Assuming cubic kernel
    
    // Calculate output dimensions (assuming same padding with odd kernel size)
    auto output_depth = input_depth;
    auto output_height = input_height;
    auto output_width = input_width;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Calculate total number of output elements
    auto total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    // Launch kernel
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_conv3d_activation_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        scaling_factor.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        kernel_size,
        output_depth,
        output_height,
        output_width
    );
    
    return output;
}
"""

fused_conv3d_activation_cpp_source = """
torch::Tensor fused_conv3d_activation_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scaling_factor,
    torch::Tensor bias
);
"""

# Compile the inline CUDA code
fused_conv3d_activation = load_inline(
    name="fused_conv3d_activation",
    cpp_sources=fused_conv3d_activation_cpp_source,
    cuda_sources=fused_conv3d_activation_source,
    functions=["fused_conv3d_activation_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for conv3d + activation operations
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Initialize weight parameter manually to match Conv3d weight format
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Store the fused operation function
        self.fused_op = fused_conv3d_activation

    def forward(self, x):
        return self.fused_op.fused_conv3d_activation_cuda(x, self.weight, self.scaling_factor, self.bias)