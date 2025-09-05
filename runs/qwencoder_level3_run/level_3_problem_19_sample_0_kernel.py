import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv-bn-relu
conv_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void conv_bn_relu_kernel(
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
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx < total_elements) {
        int w = idx % out_width;
        int h = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % out_channels;
        int b = idx / (out_width * out_height * out_channels);
        
        float sum = 0.0f;
        int kernel_radius = kernel_size / 2;
        
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_y = h * stride + ky - padding;
                int in_x = w * stride + kx - padding;
                
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    for (int ic = 0; ic < in_channels; ic++) {
                        int input_idx = b * (in_channels * in_height * in_width) + 
                                       ic * (in_height * in_width) + 
                                       in_y * in_width + in_x;
                                       
                        int weight_idx = c * (in_channels * kernel_size * kernel_size) + 
                                        ic * (kernel_size * kernel_size) + 
                                        ky * kernel_size + kx;
                                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias
        sum += bias[c];
        
        // Batch norm
        float mean = running_mean[c];
        float var = running_var[c];
        float gamma_val = gamma[c];
        float beta_val = beta[c];
        
        float normalized = (sum - mean) / sqrtf(var + eps);
        float bn_result = gamma_val * normalized + beta_val;
        
        // ReLU
        float result = fmaxf(0.0f, bn_result);
        
        output[idx] = result;
    }
}

torch::Tensor conv_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta,
    int kernel_size,
    int stride,
    int padding,
    float eps
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(0);
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_elements = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv_bn_relu_kernel<<<num_blocks, block_size>>>(
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
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        eps
    );
    
    return output;
}
"""

conv_bn_relu_cpp_source = """
torch::Tensor conv_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta,
    int kernel_size,
    int stride,
    int padding,
    float eps
);
"""

# Compile the inline CUDA code for fused conv-bn-relu
conv_bn_relu = load_inline(
    name="conv_bn_relu",
    cpp_sources=conv_bn_relu_cpp_source,
    cuda_sources=conv_bn_relu_source,
    functions=["conv_bn_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for depthwise separable conv
depthwise_separable_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv_kernel(
    const float* input,
    const float* depthwise_weight,
    const float* pointwise_weight,
    const float* dw_bias,
    const float* pw_bias,
    const float* dw_running_mean,
    const float* dw_running_var,
    const float* dw_gamma,
    const float* dw_beta,
    const float* pw_running_mean,
    const float* pw_running_var,
    const float* pw_gamma,
    const float* pw_beta,
    float* output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx < total_elements) {
        int w = idx % out_width;
        int h = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % out_channels;
        int b = idx / (out_width * out_height * out_channels);
        
        // Depthwise conv
        float dw_sum = 0.0f;
        int kernel_radius = kernel_size / 2;
        int dw_channel = c % in_channels; // Assuming out_channels is multiple of in_channels
        
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_y = h * stride + ky - padding;
                int in_x = w * stride + kx - padding;
                
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    int input_idx = b * (in_channels * in_height * in_width) + 
                                   dw_channel * (in_height * in_width) + 
                                   in_y * in_width + in_x;
                                   
                    int weight_idx = dw_channel * (kernel_size * kernel_size) + 
                                    ky * kernel_size + kx;
                                    
                    dw_sum += input[input_idx] * depthwise_weight[weight_idx];
                }
            }
        }
        
        // Depthwise bias and batch norm
        dw_sum += dw_bias[dw_channel];
        float dw_mean = dw_running_mean[dw_channel];
        float dw_var = dw_running_var[dw_channel];
        float dw_gamma_val = dw_gamma[dw_channel];
        float dw_beta_val = dw_beta[dw_channel];
        float dw_normalized = (dw_sum - dw_mean) / sqrtf(dw_var + eps);
        float dw_bn_result = dw_gamma_val * dw_normalized + dw_beta_val;
        float dw_relu_result = fmaxf(0.0f, dw_bn_result);
        
        // Pointwise conv
        float pw_sum = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
            int dw_idx = b * (in_channels * out_height * out_width) + 
                        ic * (out_height * out_width) + 
                        h * out_width + w;
            int pw_weight_idx = c * in_channels + ic;
            pw_sum += dw_relu_result * pointwise_weight[pw_weight_idx];
        }
        
        // Pointwise bias and batch norm
        pw_sum += pw_bias[c];
        float pw_mean = pw_running_mean[c];
        float pw_var = pw_running_var[c];
        float pw_gamma_val = pw_gamma[c];
        float pw_beta_val = pw_beta[c];
        float pw_normalized = (pw_sum - pw_mean) / sqrtf(pw_var + eps);
        float pw_bn_result = pw_gamma_val * pw_normalized + pw_beta_val;
        float pw_relu_result = fmaxf(0.0f, pw_bn_result);
        
        output[idx] = pw_relu_result;
    }
}

torch::Tensor depthwise_separable_conv_cuda(
    torch::Tensor input,
    torch::Tensor depthwise_weight,
    torch::Tensor pointwise_weight,
    torch::Tensor dw_bias,
    torch::Tensor pw_bias,
    torch::Tensor dw_running_mean,
    torch::Tensor dw_running_var,
    torch::Tensor dw_gamma,
    torch::Tensor dw_beta,
    torch::Tensor pw_running_mean,
    torch::Tensor pw_running_var,
    torch::Tensor pw_gamma,
    torch::Tensor pw_beta,
    int kernel_size,
    int stride,
    int padding,
    float eps
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = pointwise_weight.size(0);
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_elements = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    depthwise_conv_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        depthwise_weight.data_ptr<float>(),
        pointwise_weight.data_ptr<float>(),
        dw_bias.data_ptr<float>(),
        pw_bias.data_ptr<float>(),
        dw_running_mean.data_ptr<float>(),
        dw_running_var.data_ptr<float>(),
        dw_gamma.data_ptr<float>(),
        dw_beta.data_ptr<float>(),
        pw_running_mean.data_ptr<float>(),
        pw_running_var.data_ptr<float>(),
        pw_gamma.data_ptr<float>(),
        pw_beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        eps
    );
    
    return output;
}
"""

depthwise_separable_conv_cpp_source = """
torch::Tensor depthwise_separable_conv_cuda(
    torch::Tensor input,
    torch::Tensor depthwise_weight,
    torch::Tensor pointwise_weight,
    torch::Tensor dw_bias,
    torch::Tensor pw_bias,
    torch::Tensor dw_running_mean,
    torch::Tensor dw_running_var,
    torch::Tensor dw_gamma,
    torch::Tensor dw_beta,
    torch::Tensor pw_running_mean,
    torch::Tensor pw_running_var,
    torch::Tensor pw_gamma,
    torch::Tensor pw_beta,
    int kernel_size,
    int stride,
    int padding,
    float eps
);
"""

# Compile the inline CUDA code for depthwise separable conv
depthwise_separable_conv = load_inline(
    name="depthwise_separable_conv",
    cpp_sources=depthwise_separable_conv_cpp_source,
    cuda_sources=depthwise_separable_conv_source,
    functions=["depthwise_separable_conv_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class FusedConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(FusedConvBnReLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.eps = 1e-5
        
        # Conv parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Batch norm parameters
        self.running_mean = nn.Parameter(torch.randn(out_channels), requires_grad=False)
        self.running_var = nn.Parameter(torch.randn(out_channels), requires_grad=False)
        self.gamma = nn.Parameter(torch.randn(out_channels))
        self.beta = nn.Parameter(torch.randn(out_channels))
        
    def forward(self, x):
        return conv_bn_relu.conv_bn_relu_cuda(
            x, self.weight, self.bias, self.running_mean, self.running_var,
            self.gamma, self.beta, self.kernel_size, self.stride, self.padding, self.eps
        )

class FusedDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(FusedDepthwiseSeparableConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = 3
        self.padding = 1
        self.eps = 1e-5
        
        # Depthwise conv parameters
        self.depthwise_weight = nn.Parameter(torch.randn(in_channels, 1, 3, 3))
        self.dw_bias = nn.Parameter(torch.randn(in_channels))
        
        # Pointwise conv parameters
        self.pointwise_weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        self.pw_bias = nn.Parameter(torch.randn(out_channels))
        
        # Depthwise batch norm parameters
        self.dw_running_mean = nn.Parameter(torch.randn(in_channels), requires_grad=False)
        self.dw_running_var = nn.Parameter(torch.randn(in_channels), requires_grad=False)
        self.dw_gamma = nn.Parameter(torch.randn(in_channels))
        self.dw_beta = nn.Parameter(torch.randn(in_channels))
        
        # Pointwise batch norm parameters
        self.pw_running_mean = nn.Parameter(torch.randn(out_channels), requires_grad=False)
        self.pw_running_var = nn.Parameter(torch.randn(out_channels), requires_grad=False)
        self.pw_gamma = nn.Parameter(torch.randn(out_channels))
        self.pw_beta = nn.Parameter(torch.randn(out_channels))
        
    def forward(self, x):
        return depthwise_separable_conv.depthwise_separable_conv_cuda(
            x,
            self.depthwise_weight,
            self.pointwise_weight,
            self.dw_bias,
            self.pw_bias,
            self.dw_running_mean,
            self.dw_running_var,
            self.dw_gamma,
            self.dw_beta,
            self.pw_running_mean,
            self.pw_running_var,
            self.pw_gamma,
            self.pw_beta,
            self.kernel_size,
            self.stride,
            self.padding,
            self.eps
        )

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        """
        MobileNetV1 architecture implementation with custom CUDA kernels.
        """
        super(ModelNew, self).__init__()
        
        def conv_bn(inp, oup, stride):
            return FusedConvBnReLU(inp, oup, 3, stride, 1)
        
        def conv_dw(inp, oup, stride):
            return FusedDepthwiseSeparableConv(inp, oup, stride)
        
        self.model = nn.Sequential(
            conv_bn(input_channels, int(32 * alpha), 2),
            conv_dw(int(32 * alpha), int(64 * alpha), 1),
            conv_dw(int(64 * alpha), int(128 * alpha), 2),
            conv_dw(int(128 * alpha), int(128 * alpha), 1),
            conv_dw(int(128 * alpha), int(256 * alpha), 2),
            conv_dw(int(256 * alpha), int(256 * alpha), 1),
            conv_dw(int(256 * alpha), int(512 * alpha), 2),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(1024 * alpha), 2),
            conv_dw(int(1024 * alpha), int(1024 * alpha), 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(int(1024 * alpha), num_classes)
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_channels, height, width)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x