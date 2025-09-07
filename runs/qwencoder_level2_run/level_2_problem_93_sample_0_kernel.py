import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_conv_transpose_add_min_gelu_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    float add_value,
    float multiply_value
) {
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y;
    int out_h = threadIdx.x + blockIdx.z * blockDim.x;
    int out_w = threadIdx.y + blockIdx.z * blockDim.y * blockDim.x;
    
    if (batch_idx >= batch_size || out_ch >= out_channels || out_h >= output_height || out_w >= output_width)
        return;
        
    float sum = 0.0f;
    
    // Conv transpose operation
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int in_h = (out_h - kh + kernel_size - 1) / stride;
                int in_w = (out_w - kw + kernel_size - 1) / stride;
                
                if ((out_h - kh + kernel_size - 1) % stride == 0 && 
                    (out_w - kw + kernel_size - 1) % stride == 0 &&
                    in_h >= 0 && in_h < input_height &&
                    in_w >= 0 && in_w < input_width) {
                    
                    int input_idx = batch_idx * (in_channels * input_height * input_width) +
                                    in_ch * (input_height * input_width) +
                                    in_h * input_width + in_w;
                                    
                    int weight_idx = out_ch * (in_channels * kernel_size * kernel_size) +
                                     in_ch * (kernel_size * kernel_size) +
                                     kh * kernel_size + kw;
                                     
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias if exists
    if (bias != nullptr) {
        sum += bias[out_ch];
    }
    
    // Add value
    sum += add_value;
    
    // Min with 0
    sum = fminf(sum, 0.0f);
    
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float gelu_coeff = 0.7978845608028654f; // sqrt(2/pi)
    float x3 = sum * sum * sum;
    float inner = gelu_coeff * (sum + 0.044715f * x3);
    float tanh_inner = tanhf(inner);
    sum = 0.5f * sum * (1.0f + tanh_inner);
    
    // Multiply by value
    sum *= multiply_value;
    
    int output_idx = batch_idx * (out_channels * output_height * output_width) +
                     out_ch * (output_height * output_width) +
                     out_h * output_width + out_w;
                     
    output[output_idx] = sum;
}

torch::Tensor fused_conv_transpose_add_min_gelu_mul_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    float add_value,
    float multiply_value
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);
    
    auto out_channels = weight.size(0);
    auto output_height = (input_height - 1) * stride + kernel_size;
    auto output_width = (input_width - 1) * stride + kernel_size;
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    dim3 block_size(16, 16);
    dim3 grid_size(batch_size, out_channels, (output_height * output_width + 255) / 256);
    
    fused_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        add_value,
        multiply_value
    );
    
    return output;
}
"""

fused_conv_transpose_add_min_gelu_mul_cpp_source = """
torch::Tensor fused_conv_transpose_add_min_gelu_mul_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    float add_value,
    float multiply_value
);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_conv_transpose_add_min_gelu_mul",
    cpp_sources=fused_conv_transpose_add_min_gelu_mul_cpp_source,
    cuda_sources=fused_conv_transpose_add_min_gelu_mul_source,
    functions=["fused_conv_transpose_add_min_gelu_mul_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for conv transpose + add + min + GELU + multiply
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.add_value = add_value
        self.multiply_value = multiply_value
        
        # Create the transposed convolution layer to get weights and bias
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        
    def forward(self, x):
        return fused_op.fused_conv_transpose_add_min_gelu_mul_cuda(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.kernel_size,
            self.stride,
            self.add_value,
            self.multiply_value
        )