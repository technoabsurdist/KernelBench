import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv + activation + batch norm
conv_act_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

__global__ void softplus_tanh_multiply_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float softplus_val = logf(1.0f + expf(input[idx]));
        float tanh_val = tanhf(softplus_val);
        output[idx] = tanh_val * input[idx];
    }
}

__global__ void batch_norm_inference_kernel(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int batch_size,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        int channel = (idx / (height * width)) % channels;
        float normalized = (input[idx] - running_mean[channel]) / sqrtf(running_var[channel] + eps);
        output[idx] = weight[channel] * normalized + bias[channel];
    }
}

torch::Tensor fused_conv_act_bn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
) {
    // Convolution using cuBLAS
    auto conv_output = torch::conv2d(input, weight, bias, 1, 1, 1, 1);
    
    // Activation: tanh(softplus(x)) * x
    auto act_output = torch::zeros_like(conv_output);
    int size = conv_output.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    softplus_tanh_multiply_kernel<<<num_blocks, block_size>>>(
        conv_output.data_ptr<float>(),
        act_output.data_ptr<float>(),
        size
    );
    
    // Batch normalization
    auto bn_output = torch::zeros_like(act_output);
    int batch_size = act_output.size(0);
    int channels = act_output.size(1);
    int height = act_output.size(2);
    int width = act_output.size(3);
    int total_elements = batch_size * channels * height * width;
    
    const int bn_block_size = 256;
    const int bn_num_blocks = (total_elements + bn_block_size - 1) / bn_block_size;
    
    batch_norm_inference_kernel<<<bn_num_blocks, bn_block_size>>>(
        act_output.data_ptr<float>(),
        bn_output.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        batch_size,
        channels,
        height,
        width
    );
    
    return bn_output;
}
"""

conv_act_bn_cpp_source = """
torch::Tensor fused_conv_act_bn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
);
"""

# Compile the inline CUDA code
fused_conv_act_bn = load_inline(
    name="fused_conv_act_bn",
    cpp_sources=conv_act_bn_cpp_source,
    cuda_sources=conv_act_bn_source,
    functions=["fused_conv_act_bn_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused convolution, activation, and batch normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.fused_op = fused_conv_act_bn

    def forward(self, x):
        if self.training:
            # During training, use standard PyTorch operations
            x = self.conv(x)
            x = torch.multiply(torch.tanh(torch.nn.functional.softplus(x)), x)
            x = self.bn(x)
            return x
        else:
            # During inference, use fused CUDA kernel
            return self.fused_op.fused_conv_act_bn_cuda(
                x,
                self.conv.weight,
                self.conv.bias,
                self.bn.weight,
                self.bn.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps
            )