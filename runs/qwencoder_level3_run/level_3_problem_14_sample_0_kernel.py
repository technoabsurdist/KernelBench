import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused BatchNorm + ReLU + Conv2d
fused_bn_relu_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

__global__ void fused_bn_relu_conv_kernel(
    const float* input,
    const float* weight,
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
    int kernel_size
) {
    int out_ch = blockIdx.x;
    int batch_idx = blockIdx.y;
    int out_h = threadIdx.x;
    int out_w = threadIdx.y;
    
    if (out_ch >= out_channels || batch_idx >= batch_size || out_h >= in_height || out_w >= in_width) return;
    
    int padding = (kernel_size - 1) / 2;
    float eps = 1e-5;
    
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        float sum = 0.0f;
        
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int in_h = out_h + kh - padding;
                int in_w = out_w + kw - padding;
                
                if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                    int input_idx = batch_idx * (in_channels * in_height * in_width) + 
                                   in_ch * (in_height * in_width) + 
                                   in_h * in_width + in_w;
                                   
                    // BatchNorm: (x - mean) / sqrt(var + eps)
                    float normalized = (input[input_idx] - running_mean[in_ch]) / sqrtf(running_var[in_ch] + eps);
                    // Scale and shift: gamma * normalized + beta
                    float bn_out = gamma[in_ch] * normalized + beta[in_ch];
                    // ReLU
                    float relu_out = fmaxf(bn_out, 0.0f);
                    
                    int weight_idx = out_ch * (in_channels * kernel_size * kernel_size) + 
                                    in_ch * (kernel_size * kernel_size) + 
                                    kh * kernel_size + kw;
                                    
                    sum += relu_out * weight[weight_idx];
                }
            }
        }
        
        int output_idx = batch_idx * (out_channels * in_height * in_width) + 
                        out_ch * (in_height * in_width) + 
                        out_h * in_width + out_w;
        output[output_idx] += sum;
    }
}

torch::Tensor fused_bn_relu_conv_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    auto output = torch::zeros({batch_size, out_channels, in_height, in_width}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    dim3 grid(out_channels, batch_size);
    dim3 block(in_height, in_width);
    
    fused_bn_relu_conv_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
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
        kernel_size
    );
    
    return output;
}
"""

fused_bn_relu_conv_cpp_source = """
torch::Tensor fused_bn_relu_conv_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta
);
"""

# Compile the inline CUDA code for fused operation
fused_bn_relu_conv = load_inline(
    name="fused_bn_relu_conv",
    cpp_sources=fused_bn_relu_conv_cpp_source,
    cuda_sources=fused_bn_relu_conv_source,
    functions=["fused_bn_relu_conv_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for channel concatenation
concat_channels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void concat_channels_kernel(
    const float* input1,
    const float* input2,
    float* output,
    int batch_size,
    int channels1,
    int channels2,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * (channels1 + channels2) * height * width;
    
    if (idx < total_elements) {
        int batch_idx = idx / ((channels1 + channels2) * height * width);
        int remaining = idx % ((channels1 + channels2) * height * width);
        int channel_idx = remaining / (height * width);
        int spatial_idx = remaining % (height * width);
        
        if (channel_idx < channels1) {
            int input_idx = batch_idx * (channels1 * height * width) + 
                           channel_idx * (height * width) + spatial_idx;
            output[idx] = input1[input_idx];
        } else {
            int input_idx = batch_idx * (channels2 * height * width) + 
                           (channel_idx - channels1) * (height * width) + spatial_idx;
            output[idx] = input2[input_idx];
        }
    }
}

torch::Tensor concat_channels_cuda(torch::Tensor input1, torch::Tensor input2) {
    auto batch_size = input1.size(0);
    auto channels1 = input1.size(1);
    auto channels2 = input2.size(1);
    auto height = input1.size(2);
    auto width = input1.size(3);
    
    auto output = torch::zeros({batch_size, channels1 + channels2, height, width}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_elements = batch_size * (channels1 + channels2) * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    concat_channels_kernel<<<num_blocks, block_size>>>(
        input1.data_ptr<float>(),
        input2.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels1,
        channels2,
        height,
        width
    );
    
    return output;
}
"""

concat_channels_cpp_source = """
torch::Tensor concat_channels_cuda(torch::Tensor input1, torch::Tensor input2);
"""

# Compile the inline CUDA code for channel concatenation
concat_channels = load_inline(
    name="concat_channels",
    cpp_sources=concat_channels_cpp_source,
    cuda_sources=concat_channels_source,
    functions=["concat_channels_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class FusedLayer(nn.Module):
    def __init__(self, in_features: int, growth_rate: int):
        super(FusedLayer, self).__init__()
        self.in_features = in_features
        self.growth_rate = growth_rate
        
        # BatchNorm parameters
        self.running_mean = nn.Parameter(torch.zeros(in_features))
        self.running_var = nn.Parameter(torch.ones(in_features))
        self.gamma = nn.Parameter(torch.ones(in_features))
        self.beta = nn.Parameter(torch.zeros(in_features))
        
        # Conv2d weight
        self.weight = nn.Parameter(torch.randn(growth_rate, in_features, 3, 3) * 0.1)
        
        # Initialize parameters
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        return fused_bn_relu_conv.fused_bn_relu_conv_cuda(
            x, self.weight, self.running_mean, self.running_var, self.gamma, self.beta
        )

class ModelNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(ModelNew, self).__init__()
        self.num_layers = num_layers
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate
        
        layers = []
        for i in range(num_layers):
            layers.append(FusedLayer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)
        
        self.concat_op = concat_channels

    def forward(self, x):
        features = [x]
        current_x = x
        
        for i, layer in enumerate(self.layers):
            new_feature = layer(current_x)
            features.append(new_feature)
            
            # Concatenate all features
            if i == 0:
                current_x = self.concat_op.concat_channels_cuda(current_x, new_feature)
            else:
                # For subsequent layers, we need to concatenate with all previous features
                temp = features[0]
                for j in range(1, len(features)):
                    temp = self.concat_op.concat_channels_cuda(temp, features[j])
                current_x = temp
                
        return current_x

batch_size = 10
num_layers = 6
num_input_features = 32
growth_rate = 32
height, width = 224, 224

def get_inputs():
    return [torch.rand(batch_size, num_input_features, height, width).cuda()]

def get_init_inputs():
    return [num_layers, num_input_features, growth_rate]