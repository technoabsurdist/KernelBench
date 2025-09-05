import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused BatchNorm + ReLU
batchnorm_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void batchnorm_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    const float eps,
    const int num_features,
    const int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_features * spatial_size;
    
    if (idx < total_elements) {
        int feature_idx = (idx / spatial_size) % num_features;
        float mean = running_mean[feature_idx];
        float var = running_var[feature_idx];
        float w = weight[feature_idx];
        float b = bias[feature_idx];
        
        float normalized = (input[idx] - mean) / sqrtf(var + eps);
        float scaled = w * normalized + b;
        output[idx] = fmaxf(scaled, 0.0f);  // ReLU
    }
}

torch::Tensor batchnorm_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    auto output = torch::zeros_like(input);
    int batch_size = input.size(0);
    int num_features = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int spatial_size = height * width;
    int total_elements = batch_size * num_features * spatial_size;
    
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    batchnorm_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(eps),
        num_features,
        spatial_size
    );
    
    return output;
}
"""

batchnorm_relu_cpp_source = """
torch::Tensor batchnorm_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps
);
"""

# Custom CUDA kernel for fused Conv2D + BatchNorm + ReLU
conv_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>

__global__ void im2col_kernel(
    const float* data_im,
    float* data_col,
    const int channels,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = channels * height_col * width_col;
    
    if (index < total_threads) {
        int w_out = index % width_col;
        index /= width_col;
        int h_out = index % height_col;
        int channel_in = index / height_col;
        int channel_out = channel_in * kernel_h * kernel_w;
        int h_in = h_out * stride_h - pad_h;
        int w_in = w_out * stride_w - pad_w;
        
        float* data_col_ptr = data_col + (channel_out * height_col + h_out) * width_col + w_out;
        const float* data_im_ptr = data_im + (channel_in * height + h_in) * width + w_in;
        
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                int h = h_in + i;
                int w = w_in + j;
                
                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

__global__ void conv_forward_kernel(
    const float* input,
    const float* weight,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int out_height,
    const int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * out_height * out_width;
    
    if (idx < total_outputs) {
        int w_out = idx % out_width;
        idx /= out_width;
        int h_out = idx % out_height;
        idx /= out_height;
        int out_ch = idx % out_channels;
        int batch = idx / out_channels;
        
        float sum = 0;
        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
            for (int i = 0; i < kernel_h; ++i) {
                for (int j = 0; j < kernel_w; ++j) {
                    int h_in = h_out + i;
                    int w_in = w_out + j;
                    sum += input[((batch * in_channels + in_ch) * height + h_in) * width + w_in] *
                           weight[((out_ch * in_channels + in_ch) * kernel_h + i) * kernel_w + j];
                }
            }
        }
        output[idx * out_height * out_width + h_out * out_width + w_out] = sum;
    }
}

__global__ void batchnorm_relu_after_conv_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    const float eps,
    const int batch_size,
    const int num_features,
    const int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_features * spatial_size;
    
    if (idx < total_elements) {
        int feature_idx = (idx / spatial_size) % num_features;
        float mean = running_mean[feature_idx];
        float var = running_var[feature_idx];
        float w = weight[feature_idx];
        float b = bias[feature_idx];
        
        float normalized = (input[idx] - mean) / sqrtf(var + eps);
        float scaled = w * normalized + b;
        output[idx] = fmaxf(scaled, 0.0f);  // ReLU
    }
}

torch::Tensor conv_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps,
    int kernel_size,
    int padding,
    int stride
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;
    int out_channels = weight.size(0);
    
    // Convolution output
    auto conv_output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Perform convolution
    const int conv_block_size = 256;
    int conv_total_outputs = batch_size * out_channels * out_height * out_width;
    const int conv_num_blocks = (conv_total_outputs + conv_block_size - 1) / conv_block_size;
    
    conv_forward_kernel<<<conv_num_blocks, conv_block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        kernel_size,
        out_height,
        out_width
    );
    
    // BatchNorm + ReLU
    auto final_output = torch::zeros_like(conv_output);
    int spatial_size = out_height * out_width;
    int total_elements = batch_size * out_channels * spatial_size;
    
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    batchnorm_relu_after_conv_kernel<<<num_blocks, block_size>>>(
        conv_output.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        final_output.data_ptr<float>(),
        static_cast<float>(eps),
        batch_size,
        out_channels,
        spatial_size
    );
    
    return final_output;
}
"""

conv_bn_relu_cpp_source = """
torch::Tensor conv_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps,
    int kernel_size,
    int padding,
    int stride
);
"""

# Compile the inline CUDA code
batchnorm_relu = load_inline(
    name="batchnorm_relu",
    cpp_sources=batchnorm_relu_cpp_source,
    cuda_sources=batchnorm_relu_source,
    functions=["batchnorm_relu_cuda"],
    verbose=False,
)

conv_bn_relu = load_inline(
    name="conv_bn_relu",
    cpp_sources=conv_bn_relu_cpp_source,
    cuda_sources=conv_bn_relu_source,
    functions=["conv_bn_relu_cuda"],
    verbose=False,
)

class FusedBatchNormReLU(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(FusedBatchNormReLU, self).__init__()
        self.num_features = num_features
        self.eps = eps
        
        # BatchNorm parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        self.batchnorm_relu = batchnorm_relu

    def forward(self, x):
        if x.is_cuda and x.dtype == torch.float32:
            return self.batchnorm_relu.batchnorm_relu_cuda(
                x, self.weight, self.bias, 
                self.running_mean, self.running_var, self.eps
            )
        else:
            x = F.batch_norm(x, self.running_mean, self.running_var, 
                            self.weight, self.bias, False, 0.1, self.eps)
            return F.relu(x, inplace=True)

class FusedConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, eps=1e-5):
        super(FusedConvBNReLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.eps = eps
        
        # Conv parameters
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.conv_weight, mode='fan_out', nonlinearity='relu')
        
        # BatchNorm parameters
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        
        self.conv_bn_relu = conv_bn_relu

    def forward(self, x):
        if x.is_cuda and x.dtype == torch.float32 and self.kernel_size == 3 and self.padding == 1:
            return self.conv_bn_relu.conv_bn_relu_cuda(
                x, self.conv_weight, torch.zeros(self.out_channels, device=x.device),
                self.bn_weight, self.bn_bias,
                self.running_mean, self.running_var, self.eps,
                self.kernel_size, self.padding, self.stride
            )
        else:
            x = F.conv2d(x, self.conv_weight, None, self.stride, self.padding, 1, 1)
            x = F.batch_norm(x, self.running_mean, self.running_var,
                            self.bn_weight, self.bn_bias, False, 0.1, self.eps)
            return F.relu(x, inplace=True)

class OptimizedDenseLayer(nn.Module):
    def __init__(self, in_features: int, growth_rate: int):
        super(OptimizedDenseLayer, self).__init__()
        self.conv = FusedConvBNReLU(in_features, growth_rate, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        new_feature = self.conv(x)
        new_feature = self.dropout(new_feature)
        return new_feature

class DenseBlockNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlockNew, self).__init__()
        self.layers = nn.ModuleList([
            OptimizedDenseLayer(num_input_features + i * growth_rate, growth_rate)
            for i in range(num_layers)
        ])

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)
        return x

class TransitionLayerNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayerNew, self).__init__()
        self.bn = FusedBatchNormReLU(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelNew, self).__init__()

        # Initial convolution with fused BN+ReLU
        self.init_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.init_bn = FusedBatchNormReLU(64)
        self.init_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        num_features = 64
        block_layers = [6, 12, 24, 16]

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlockNew(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayerNew(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        self.final_bn = FusedBatchNormReLU(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.init_pool(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.final_bn(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x