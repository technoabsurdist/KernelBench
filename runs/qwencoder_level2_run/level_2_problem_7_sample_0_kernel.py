import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv3d + relu + leaky_relu + gelu + sigmoid + bias
fused_conv3d_activations_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__device__ float gelu_approx(float x) {
    float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    return x * cdf;
}

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_conv3d_activation_bias_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_size) {
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    if (out_idx >= total_output_elements) return;
    
    int w = out_idx % output_w;
    out_idx /= output_w;
    int h = out_idx % output_h;
    out_idx /= output_h;
    int d = out_idx % output_d;
    out_idx /= output_d;
    int oc = out_idx % out_channels;
    int b = out_idx / out_channels;
    
    float sum = 0.0f;
    
    // Convolution operation
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int in_d = d + kd - kernel_size/2;
                    int in_h = h + kh - kernel_size/2;
                    int in_w = w + kw - kernel_size/2;
                    
                    if (in_d >= 0 && in_d < input_d && 
                        in_h >= 0 && in_h < input_h && 
                        in_w >= 0 && in_w < input_w) {
                        
                        int input_idx = b * (in_channels * input_d * input_h * input_w) + 
                                       ic * (input_d * input_h * input_w) + 
                                       in_d * (input_h * input_w) + 
                                       in_h * input_w + in_w;
                                       
                        int weight_idx = oc * (in_channels * kernel_size * kernel_size * kernel_size) + 
                                        ic * (kernel_size * kernel_size * kernel_size) + 
                                        kd * (kernel_size * kernel_size) + 
                                        kh * kernel_size + kw;
                                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Apply activations in sequence: relu -> leaky_relu -> gelu -> sigmoid
    float val = sum;
    val = fmaxf(0.0f, val);  // ReLU
    val = val > 0 ? val : 0.01f * val;  // Leaky ReLU with negative_slope=0.01
    val = gelu_approx(val);  // GELU approximation
    val = sigmoid(val);  // Sigmoid
    
    // Add bias
    val += bias[oc];
    
    output[out_idx * (output_d * output_h * output_w) + 
           d * (output_h * output_w) + 
           h * output_w + w] = val;
}

torch::Tensor fused_conv3d_activations_bias_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_d = input.size(2);
    int input_h = input.size(3);
    int input_w = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);  // Assuming cubic kernel
    
    // Calculate output dimensions (assuming same padding with odd kernel size)
    int output_d = input_d;
    int output_h = input_h;
    int output_w = input_w;
    
    auto output = torch::zeros({batch_size, out_channels, output_d, output_h, output_w}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_output_elements = batch_size * out_channels * output_d * output_h * output_w;
    const int threads_per_block = 256;
    const int num_blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;
    
    fused_conv3d_activation_bias_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_size
    );
    
    return output;
}
"""

fused_conv3d_activations_bias_cpp_source = """
torch::Tensor fused_conv3d_activations_bias_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_conv3d_activations_bias = load_inline(
    name="fused_conv3d_activations_bias",
    cpp_sources=fused_conv3d_activations_bias_cpp_source,
    cuda_sources=fused_conv3d_activations_bias_source,
    functions=["fused_conv3d_activations_bias_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model with fused conv3d + activations + bias operation using custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.fused_op = fused_conv3d_activations_bias

    def forward(self, x):
        return self.fused_op.fused_conv3d_activations_bias_cuda(x, self.weight, self.bias)

# Helper functions remain the same
batch_size = 64
in_channels = 8
out_channels = 32
depth, height, width = 32, 64, 64
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]