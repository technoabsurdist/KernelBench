import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv3d transpose + clamp + div
conv_clamp_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void conv_transpose3d_clamp_div_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const float min_value,
    const float divisor
) {
    // Calculate output position
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (out_idx >= total_output_elements) return;
    
    // Decompose output index
    int temp = out_idx;
    const int w_out = temp % output_width;
    temp /= output_width;
    const int h_out = temp % output_height;
    temp /= output_height;
    const int d_out = temp % output_depth;
    temp /= output_depth;
    const int c_out = temp % out_channels;
    const int n = temp / out_channels;
    
    // Calculate input region that contributes to this output element
    const int d_start = d_out * stride - padding;
    const int h_start = h_out * stride - padding;
    const int w_start = w_out * stride - padding;
    
    float sum = 0.0f;
    
    // Iterate through kernel
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                const int d_in = d_start + kd;
                const int h_in = h_start + kh;
                const int w_in = w_start + kw;
                
                // Check if input position is valid
                if (d_in >= 0 && d_in < input_depth &&
                    h_in >= 0 && h_in < input_height &&
                    w_in >= 0 && w_in < input_width) {
                    
                    // Iterate through input channels
                    for (int c_in = 0; c_in < in_channels; c_in++) {
                        // Calculate weight index (output channel, input channel, kernel dimensions)
                        const int weight_idx = c_out * (in_channels * kernel_size * kernel_size * kernel_size) +
                                              c_in * (kernel_size * kernel_size * kernel_size) +
                                              kd * (kernel_size * kernel_size) +
                                              kh * kernel_size +
                                              kw;
                        
                        // Calculate input index
                        const int input_idx = n * (in_channels * input_depth * input_height * input_width) +
                                             c_in * (input_depth * input_height * input_width) +
                                             d_in * (input_height * input_width) +
                                             h_in * input_width +
                                             w_in;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add bias
    sum += bias[c_out];
    
    // Apply clamp and division
    sum = fmaxf(sum, min_value);
    sum = sum / divisor;
    
    // Write output
    output[out_idx] = sum;
}

torch::Tensor conv_transpose3d_clamp_div_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    float min_value,
    float divisor
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    // Get dimensions
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_depth = input.size(2);
    const int input_height = input.size(3);
    const int input_width = input.size(4);
    
    // Calculate output dimensions
    const int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size;
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, weight.size(0), output_depth, output_height, output_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    // Calculate total number of output elements
    const int total_output_elements = batch_size * weight.size(0) * output_depth * output_height * output_width;
    
    // Configure kernel launch parameters
    const int block_size = 256;
    const int num_blocks = (total_output_elements + block_size - 1) / block_size;
    
    // Launch kernel
    conv_transpose3d_clamp_div_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        weight.size(0),
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        min_value,
        divisor
    );
    
    return output;
}
"""

conv_clamp_div_cpp_source = """
torch::Tensor conv_transpose3d_clamp_div_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    float min_value,
    float divisor
);
"""

# Compile the inline CUDA code
conv_clamp_div = load_inline(
    name="conv_clamp_div",
    cpp_sources=conv_clamp_div_cpp_source,
    cuda_sources=conv_clamp_div_source,
    functions=["conv_transpose3d_clamp_div_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused conv3d transpose + clamp + div operation
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor
        self.fused_op = conv_clamp_div

    def forward(self, x):
        return self.fused_op.conv_transpose3d_clamp_div_cuda(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.conv_transpose.kernel_size[0],
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0],
            self.min_value,
            self.divisor
        )