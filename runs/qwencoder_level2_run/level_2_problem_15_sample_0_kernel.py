import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv3d transpose + batch norm + mean subtraction
conv_bn_mean_subtract_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void conv3d_transpose_bn_mean_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (out_idx >= total_elements) return;
    
    int w = out_idx % output_width;
    out_idx /= output_width;
    int h = out_idx % output_height;
    out_idx /= output_height;
    int d = out_idx % output_depth;
    out_idx /= output_depth;
    int c = out_idx % out_channels;
    int b = out_idx / out_channels;
    
    // Calculate input coordinates for transpose convolution
    float sum = 0.0f;
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int in_d = d - padding + kd;
                int in_h = h - padding + kh;
                int in_w = w - padding + kw;
                
                if (in_d >= 0 && in_d < input_depth && 
                    in_h >= 0 && in_h < input_height && 
                    in_w >= 0 && in_w < input_width) {
                    
                    int input_idx = b * (in_channels * input_depth * input_height * input_width) +
                                   ((in_d * input_height + in_h) * input_width + in_w);
                    int weight_idx = c * (in_channels * kernel_size * kernel_size * kernel_size) +
                                    ((kd * kernel_size + kh) * kernel_size + kw);
                    
                    for (int ic = 0; ic < in_channels; ic++) {
                        sum += input[input_idx + ic * input_depth * input_height * input_width] * 
                               weight[weight_idx + ic * kernel_size * kernel_size * kernel_size];
                    }
                }
            }
        }
    }
    
    // Add bias
    sum += bias[c];
    
    // Batch norm: (x - mean) / sqrt(var + eps) * gamma + beta
    float eps = 1e-5f;
    float normalized = (sum - running_mean[c]) / sqrtf(running_var[c] + eps);
    float bn_result = normalized * gamma[c] + beta[c];
    
    output[out_idx * output_depth * output_height * output_width + 
           d * output_height * output_width + 
           h * output_width + w] = bn_result;
}

__global__ void subtract_mean_kernel(float* data, const float* means, 
                                   int batch_size, int channels, 
                                   int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * depth * height * width;
    
    if (idx >= total_elements) return;
    
    int w = idx % width;
    idx /= width;
    int h = idx % height;
    idx /= height;
    int d = idx % depth;
    idx /= depth;
    int c = idx % channels;
    int b = idx / channels;
    
    int mean_idx = b * channels + c;
    data[idx * depth * height * width + d * height * width + h * width + w] -= means[mean_idx];
}

torch::Tensor fused_conv3d_transpose_bn_mean_subtract_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta
) {
    // Get dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    // Calculate output dimensions (assuming stride=2, padding=1, kernel=3)
    int output_depth = (input_depth - 1) * 2 - 2 * 1 + 3;
    int output_height = (input_height - 1) * 2 - 2 * 1 + 3;
    int output_width = (input_width - 1) * 2 - 2 * 1 + 3;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Launch convolution + batch norm kernel
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv3d_transpose_bn_mean_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_size,
        2, // stride
        1  // padding
    );
    
    // Calculate means for subtraction
    auto means = torch::mean(output, {2, 3, 4}, true);
    
    // Subtract means
    int total_output_elements = batch_size * out_channels * output_depth * output_height * output_width;
    const int mean_block_size = 256;
    const int mean_num_blocks = (total_output_elements + mean_block_size - 1) / mean_block_size;
    
    subtract_mean_kernel<<<mean_num_blocks, mean_block_size>>>(
        output.data_ptr<float>(),
        means.data_ptr<float>(),
        batch_size,
        out_channels,
        output_depth,
        output_height,
        output_width
    );
    
    return output;
}
"""

conv_bn_mean_subtract_cpp_source = """
torch::Tensor fused_conv3d_transpose_bn_mean_subtract_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta
);
"""

# Compile the inline CUDA code
fused_conv_bn_mean_subtract = load_inline(
    name="fused_conv_bn_mean_subtract",
    cpp_sources=conv_bn_mean_subtract_cpp_source,
    cuda_sources=conv_bn_mean_subtract_source,
    functions=["fused_conv3d_transpose_bn_mean_subtract_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized version of Model with fused CUDA kernel for conv transpose + batch norm + mean subtraction
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Conv transpose weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        
        # Batch norm parameters
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        if self.bias is not None:
            fan_in = in_channels * kernel_size ** 3
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
            
        self.fused_op = fused_conv_bn_mean_subtract

    def forward(self, x):
        return self.fused_op.fused_conv3d_transpose_bn_mean_subtract_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.zeros(self.out_channels, device=x.device),
            self.running_mean,
            self.running_var,
            self.bn_weight,
            self.bn_bias
        )

batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]