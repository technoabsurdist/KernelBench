import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Fused 1x1 Conv + BatchNorm + ReLU6 kernel
conv1x1_bn_relu6_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1x1_bn_relu6_kernel(
    const float* input, const float* weight, const float* bn_weight, const float* bn_bias,
    const float* running_mean, const float* running_var, float* output,
    int batch_size, int in_channels, int out_channels, int spatial_size, float eps) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_channels * spatial_size;
    
    if (tid < total_threads) {
        int s = tid % spatial_size;
        int oc = (tid / spatial_size) % out_channels;
        int b = tid / (spatial_size * out_channels);
        
        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
            sum += input[b * in_channels * spatial_size + ic * spatial_size + s] * 
                   weight[oc * in_channels + ic];
        }
        
        // BatchNorm
        float mean = running_mean[oc];
        float var = running_var[oc];
        float scale = bn_weight[oc];
        float bias = bn_bias[oc];
        
        sum = scale * (sum - mean) / sqrtf(var + eps) + bias;
        
        // ReLU6
        sum = fminf(fmaxf(sum, 0.0f), 6.0f);
        
        output[tid] = sum;
    }
}

torch::Tensor conv1x1_bn_relu6_cuda(torch::Tensor input, torch::Tensor weight, 
                                     torch::Tensor bn_weight, torch::Tensor bn_bias,
                                     torch::Tensor running_mean, torch::Tensor running_var,
                                     float eps) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    auto spatial_size = height * width;
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * out_channels * spatial_size + threads - 1) / threads;
    
    conv1x1_bn_relu6_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, spatial_size, eps
    );
    
    return output;
}
"""

# Fused Depthwise Conv + BatchNorm + ReLU6 kernel
depthwise_bn_relu6_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_bn_relu6_kernel(
    const float* input, const float* weight, const float* bn_weight, const float* bn_bias,
    const float* running_mean, const float* running_var, float* output,
    int batch_size, int channels, int height, int width, int out_height, int out_width,
    int kernel_size, int stride, int padding, float eps) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * channels * out_height * out_width;
    
    if (tid < total_threads) {
        int ow = tid % out_width;
        int oh = (tid / out_width) % out_height;
        int c = (tid / (out_width * out_height)) % channels;
        int b = tid / (channels * out_height * out_width);
        
        float sum = 0.0f;
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;
                
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    sum += input[b * channels * height * width + c * height * width + ih * width + iw] *
                           weight[c * kernel_size * kernel_size + kh * kernel_size + kw];
                }
            }
        }
        
        // BatchNorm
        float mean = running_mean[c];
        float var = running_var[c];
        float scale = bn_weight[c];
        float bias = bn_bias[c];
        
        sum = scale * (sum - mean) / sqrtf(var + eps) + bias;
        
        // ReLU6
        sum = fminf(fmaxf(sum, 0.0f), 6.0f);
        
        output[tid] = sum;
    }
}

torch::Tensor depthwise_bn_relu6_cuda(torch::Tensor input, torch::Tensor weight,
                                       torch::Tensor bn_weight, torch::Tensor bn_bias,
                                       torch::Tensor running_mean, torch::Tensor running_var,
                                       int kernel_size, int stride, int padding, float eps) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_height = (height + 2 * padding - kernel_size) / stride + 1;
    auto out_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, channels, out_height, out_width}, input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * channels * out_height * out_width + threads - 1) / threads;
    
    depthwise_bn_relu6_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width, out_height, out_width,
        kernel_size, stride, padding, eps
    );
    
    return output;
}
"""

# Fused 1x1 Conv + BatchNorm with optional residual
conv1x1_bn_residual_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1x1_bn_residual_kernel(
    const float* input, const float* weight, const float* bn_weight, const float* bn_bias,
    const float* running_mean, const float* running_var, const float* residual,
    float* output, int batch_size, int in_channels, int out_channels, 
    int spatial_size, bool use_residual, float eps) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_channels * spatial_size;
    
    if (tid < total_threads) {
        int s = tid % spatial_size;
        int oc = (tid / spatial_size) % out_channels;
        int b = tid / (spatial_size * out_channels);
        
        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
            sum += input[b * in_channels * spatial_size + ic * spatial_size + s] * 
                   weight[oc * in_channels + ic];
        }
        
        // BatchNorm
        float mean = running_mean[oc];
        float var = running_var[oc];
        float scale = bn_weight[oc];
        float bias = bn_bias[oc];
        
        sum = scale * (sum - mean) / sqrtf(var + eps) + bias;
        
        // Add residual if needed
        if (use_residual) {
            sum += residual[tid];
        }
        
        output[tid] = sum;
    }
}

torch::Tensor conv1x1_bn_residual_cuda(torch::Tensor input, torch::Tensor weight,
                                        torch::Tensor bn_weight, torch::Tensor bn_bias,
                                        torch::Tensor running_mean, torch::Tensor running_var,
                                        torch::Tensor residual, bool use_residual, float eps) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    auto spatial_size = height * width;
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * out_channels * spatial_size + threads - 1) / threads;
    
    conv1x1_bn_residual_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        use_residual ? residual.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, spatial_size, use_residual, eps
    );
    
    return output;
}
"""

conv1x1_bn_relu6_cpp = "torch::Tensor conv1x1_bn_relu6_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float);"
depthwise_bn_relu6_cpp = "torch::Tensor depthwise_bn_relu6_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, float);"
conv1x1_bn_residual_cpp = "torch::Tensor conv1x1_bn_residual_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, bool, float);"

fused_ops = load_inline(
    name="fused_mbconv_ops",
    cpp_sources=[conv1x1_bn_relu6_cpp, depthwise_bn_relu6_cpp, conv1x1_bn_residual_cpp],
    cuda_sources=[conv1x1_bn_relu6_source, depthwise_bn_relu6_source, conv1x1_bn_residual_source],
    functions=["conv1x1_bn_relu6_cuda", "depthwise_bn_relu6_cuda", "conv1x1_bn_residual_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(ModelNew, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.expand_ratio = expand_ratio
        hidden_dim = in_channels * expand_ratio
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        
        if expand_ratio != 1:
            self.expand_conv_weight = nn.Parameter(torch.randn(hidden_dim, in_channels, 1, 1))
            self.expand_bn_weight = nn.Parameter(torch.ones(hidden_dim))
            self.expand_bn_bias = nn.Parameter(torch.zeros(hidden_dim))
            self.register_buffer('expand_bn_mean', torch.zeros(hidden_dim))
            self.register_buffer('expand_bn_var', torch.ones(hidden_dim))
            
            nn.init.kaiming_normal_(self.expand_conv_weight, mode='fan_out', nonlinearity='relu')
        
        self.depthwise_conv_weight = nn.Parameter(torch.randn(hidden_dim, 1, kernel_size, kernel_size))
        self.depthwise_bn_weight = nn.Parameter(torch.ones(hidden_dim))
        self.depthwise_bn_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.register_buffer('depthwise_bn_mean', torch.zeros(hidden_dim))
        self.register_buffer('depthwise_bn_var', torch.ones(hidden_dim))
        
        self.project_conv_weight = nn.Parameter(torch.randn(out_channels, hidden_dim, 1, 1))
        self.project_bn_weight = nn.Parameter(torch.ones(out_channels))
        self.project_bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer('project_bn_mean', torch.zeros(out_channels))
        self.register_buffer('project_bn_var', torch.ones(out_channels))
        
        nn.init.kaiming_normal_(self.depthwise_conv_weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.project_conv_weight, mode='fan_out', nonlinearity='linear')
        
        self.fused_ops = fused_ops
        self.eps = 1e-5
    
    def forward(self, x):
        identity = x
        
        if self.expand_ratio != 1:
            x = self.fused_ops.conv1x1_bn_relu6_cuda(
                x.contiguous(), 
                self.expand_conv_weight.view(self.hidden_dim, -1),
                self.expand_bn_weight,
                self.expand_bn_bias,
                self.expand_bn_mean,
                self.expand_bn_var,
                self.eps
            )
        
        x = self.fused_ops.depthwise_bn_relu6_cuda(
            x.contiguous(),
            self.depthwise_conv_weight.view(self.hidden_dim, -1),
            self.depthwise_bn_weight,
            self.depthwise_bn_bias,
            self.depthwise_bn_mean,
            self.depthwise_bn_var,
            self.kernel_size,
            self.stride,
            self.padding,
            self.eps
        )
        
        x = self.fused_ops.conv1x1_bn_residual_cuda(
            x.contiguous(),
            self.project_conv_weight.view(self.project_conv_weight.size(0), -1),
            self.project_bn_weight,
            self.project_bn_bias,
            self.project_bn_mean,
            self.project_bn_var,
            identity.contiguous(),
            self.use_residual,
            self.eps
        )
        
        return x