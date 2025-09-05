import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused BatchNorm2d + Softmax
fused_bn_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void fused_bn_softmax_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        int w = idx % width;
        int h = (idx / width) % height;
        int c = (idx / (width * height)) % channels;
        int b = idx / (channels * width * height);
        
        // BatchNorm computation
        float mean = running_mean[c];
        float var = running_var[c];
        float scale = gamma[c];
        float shift = beta[c];
        
        float normalized = (input[idx] - mean) / sqrtf(var + eps);
        float bn_out = normalized * scale + shift;
        
        // Softmax computation along width dimension (last dimension)
        // First pass: find max for numerical stability
        float max_val = -INFINITY;
        int base_idx = b * channels * height * width + c * height * width + h * width;
        for (int w_i = 0; w_i < width; w_i++) {
            float val = (input[base_idx + w_i] - mean) / sqrtf(var + eps);
            val = val * scale + shift;
            max_val = fmaxf(max_val, val);
        }
        
        // Second pass: compute exp sum
        float sum_exp = 0.0f;
        for (int w_i = 0; w_i < width; w_i++) {
            float val = (input[base_idx + w_i] - mean) / sqrtf(var + eps);
            val = val * scale + shift;
            sum_exp += expf(val - max_val);
        }
        
        // Final computation
        output[idx] = expf(bn_out - max_val) / sum_exp;
    }
}

torch::Tensor fused_bn_softmax_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (batch_size * channels * height * width + block_size - 1) / block_size;
    
    fused_bn_softmax_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width, eps
    );
    
    return output;
}
"""

fused_bn_softmax_cpp_source = """
torch::Tensor fused_bn_softmax_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
);
"""

# Custom CUDA kernel for optimized concatenation
concat_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void concat_kernel(
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
        int w = idx % width;
        int h = (idx / width) % height;
        int c = (idx / (width * height)) % (channels1 + channels2);
        int b = idx / ((channels1 + channels2) * width * height);
        
        if (c < channels1) {
            int src_idx = b * channels1 * height * width + c * height * width + h * width + w;
            output[idx] = input1[src_idx];
        } else {
            int c2 = c - channels1;
            int src_idx = b * channels2 * height * width + c2 * height * width + h * width + w;
            output[idx] = input2[src_idx];
        }
    }
}

torch::Tensor concat_cuda(torch::Tensor input1, torch::Tensor input2) {
    auto batch_size = input1.size(0);
    auto channels1 = input1.size(1);
    auto channels2 = input2.size(1);
    auto height = input1.size(2);
    auto width = input1.size(3);
    
    auto output = torch::empty({batch_size, channels1 + channels2, height, width}, input1.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * (channels1 + channels2) * height * width + block_size - 1) / block_size;
    
    concat_kernel<<<num_blocks, block_size>>>(
        input1.data_ptr<float>(),
        input2.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels1, channels2, height, width
    );
    
    return output;
}
"""

concat_cpp_source = "torch::Tensor concat_cuda(torch::Tensor input1, torch::Tensor input2);"

# Compile the inline CUDA code
fused_bn_softmax = load_inline(
    name="fused_bn_softmax",
    cpp_sources=fused_bn_softmax_cpp_source,
    cuda_sources=fused_bn_softmax_source,
    functions=["fused_bn_softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

concat_op = load_inline(
    name="concat_op",
    cpp_sources=concat_cpp_source,
    cuda_sources=concat_source,
    functions=["concat_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class DoubleConvOptimized(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.fused_bn_softmax = fused_bn_softmax

    def forward(self, x):
        x = self.conv1(x)
        x = self.fused_bn_softmax.fused_bn_softmax_cuda(
            x, 
            self.bn1.weight,
            self.bn1.bias,
            self.bn1.running_mean,
            self.bn1.running_var,
            self.bn1.eps
        )
        x = self.conv2(x)
        x = self.fused_bn_softmax.fused_bn_softmax_cuda(
            x,
            self.bn2.weight,
            self.bn2.bias,
            self.bn2.running_mean,
            self.bn2.running_var,
            self.bn2.eps
        )
        return x

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super(ModelNew, self).__init__()
        self.encoder1 = DoubleConvOptimized(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConvOptimized(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = DoubleConvOptimized(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = DoubleConvOptimized(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConvOptimized(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConvOptimized(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConvOptimized(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConvOptimized(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConvOptimized(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
        self.concat_op = concat_op

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = self.concat_op.concat_cuda(dec4, enc4)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = self.concat_op.concat_cuda(dec3, enc3)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = self.concat_op.concat_cuda(dec2, enc2)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = self.concat_op.concat_cuda(dec1, enc1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)

batch_size = 8
in_channels = 8
out_channels = 4
height = 64
width = 512
features = 64

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, features]