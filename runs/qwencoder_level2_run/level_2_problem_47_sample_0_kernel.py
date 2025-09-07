import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for conv3d + mish + tanh fusion
conv3d_mish_tanh_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void conv3d_mish_tanh_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_size,
    int stride,
    int padding
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    if (out_idx >= total_elements) return;
    
    int w = out_idx % output_w;
    out_idx /= output_w;
    int h = out_idx % output_h;
    out_idx /= output_h;
    int d = out_idx % output_d;
    out_idx /= output_d;
    int oc = out_idx % out_channels;
    int batch = out_idx / out_channels;
    
    float sum = 0.0f;
    
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int in_d = d * stride - padding + kd;
                    int in_h = h * stride - padding + kh;
                    int in_w = w * stride - padding + kw;
                    
                    if (in_d >= 0 && in_d < input_d && 
                        in_h >= 0 && in_h < input_h && 
                        in_w >= 0 && in_w < input_w) {
                        
                        int input_idx = batch * (in_channels * input_d * input_h * input_w) +
                                       ic * (input_d * input_h * input_w) +
                                       in_d * (input_h * input_w) +
                                       in_h * input_w +
                                       in_w;
                                       
                        int weight_idx = oc * (in_channels * kernel_size * kernel_size * kernel_size) +
                                        ic * (kernel_size * kernel_size * kernel_size) +
                                        kd * (kernel_size * kernel_size) +
                                        kh * kernel_size +
                                        kw;
                                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    sum += bias[oc];
    
    // Apply Mish activation: x * tanh(softplus(x))
    float softplus_val = logf(1.0f + expf(sum));
    float mish_val = sum * tanhf(softplus_val);
    
    // Apply Tanh activation
    output[out_idx] = tanhf(mish_val);
}

torch::Tensor conv3d_mish_tanh_fused_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding
) {
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_d = input_sizes[2];
    int input_h = input_sizes[3];
    int input_w = input_sizes[4];
    
    int out_channels = weight_sizes[0];
    int kernel_size = weight_sizes[2]; // Assuming cubic kernel
    
    int output_d = (input_d + 2 * padding - kernel_size) / stride + 1;
    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_d, output_h, output_w}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv3d_mish_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_size,
        stride,
        padding
    );
    
    return output;
}
"""

conv3d_mish_tanh_fused_cpp_source = """
torch::Tensor conv3d_mish_tanh_fused_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding
);
"""

# Compile the inline CUDA code for fused conv3d + mish + tanh
conv3d_mish_tanh_fused = load_inline(
    name="conv3d_mish_tanh_fused",
    cpp_sources=conv3d_mish_tanh_fused_cpp_source,
    cuda_sources=conv3d_mish_tanh_fused_source,
    functions=["conv3d_mish_tanh_fused_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused Conv3d + Mish + Tanh using custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.conv3d_mish_tanh_fused = conv3d_mish_tanh_fused

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        return self.conv3d_mish_tanh_fused.conv3d_mish_tanh_fused_cuda(
            x, self.weight, self.bias, self.stride, self.padding
        )

batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 32, 64, 64
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]