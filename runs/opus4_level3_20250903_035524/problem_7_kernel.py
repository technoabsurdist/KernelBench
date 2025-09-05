import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Fused Conv2d + ReLU kernel
conv_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cudnn.h>

__global__ void add_relu_bias_kernel(float* output, const float* bias, int channels, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = channels * spatial_size;
    
    if (idx < total_size) {
        int c = idx / spatial_size;
        float val = output[idx] + bias[c];
        output[idx] = fmaxf(0.0f, val);
    }
}

torch::Tensor conv2d_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                               int stride_h, int stride_w, int pad_h, int pad_w) {
    // Use PyTorch's cudnn conv2d, then apply fused bias+relu
    auto output = torch::cudnn_convolution(input, weight, {pad_h, pad_w}, {stride_h, stride_w}, {1, 1}, 1, false, true);
    
    int batch_size = output.size(0);
    int channels = output.size(1);
    int height = output.size(2);
    int width = output.size(3);
    int spatial_size = height * width;
    
    const int block_size = 256;
    const int num_blocks = (batch_size * channels * spatial_size + block_size - 1) / block_size;
    
    for (int b = 0; b < batch_size; b++) {
        add_relu_bias_kernel<<<num_blocks, block_size>>>(
            output[b].data_ptr<float>(), bias.data_ptr<float>(), channels, spatial_size);
    }
    
    return output;
}
"""

conv_relu_cpp_source = "torch::Tensor conv2d_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride_h, int stride_w, int pad_h, int pad_w);"

# Optimized concatenation kernel for inception modules
concat_4way_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void concat_4way_kernel(
    const float* in1, const float* in2, const float* in3, const float* in4,
    float* out, 
    int batch_size, int height, int width,
    int c1, int c2, int c3, int c4) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = height * width;
    int total_channels = c1 + c2 + c3 + c4;
    int total_size = batch_size * total_channels * spatial_size;
    
    if (idx < total_size) {
        int b = idx / (total_channels * spatial_size);
        int rem = idx % (total_channels * spatial_size);
        int c = rem / spatial_size;
        int s = rem % spatial_size;
        
        if (c < c1) {
            out[idx] = in1[b * c1 * spatial_size + c * spatial_size + s];
        } else if (c < c1 + c2) {
            out[idx] = in2[b * c2 * spatial_size + (c - c1) * spatial_size + s];
        } else if (c < c1 + c2 + c3) {
            out[idx] = in3[b * c3 * spatial_size + (c - c1 - c2) * spatial_size + s];
        } else {
            out[idx] = in4[b * c4 * spatial_size + (c - c1 - c2 - c3) * spatial_size + s];
        }
    }
}

torch::Tensor concat_4way_cuda(torch::Tensor in1, torch::Tensor in2, torch::Tensor in3, torch::Tensor in4) {
    int batch_size = in1.size(0);
    int height = in1.size(2);
    int width = in1.size(3);
    int c1 = in1.size(1);
    int c2 = in2.size(1);
    int c3 = in3.size(1);
    int c4 = in4.size(1);
    
    auto out = torch::empty({batch_size, c1 + c2 + c3 + c4, height, width}, in1.options());
    
    int total_size = batch_size * (c1 + c2 + c3 + c4) * height * width;
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    concat_4way_kernel<<<num_blocks, block_size>>>(
        in1.data_ptr<float>(), in2.data_ptr<float>(), in3.data_ptr<float>(), in4.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size, height, width, c1, c2, c3, c4);
    
    return out;
}
"""

concat_4way_cpp_source = "torch::Tensor concat_4way_cuda(torch::Tensor in1, torch::Tensor in2, torch::Tensor in3, torch::Tensor in4);"

# Optimized adaptive average pooling
adaptive_avgpool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void adaptive_avgpool_1x1_kernel(const float* input, float* output, 
                                            int batch_size, int channels, 
                                            int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels;
    
    if (idx < total_size) {
        int b = idx / channels;
        int c = idx % channels;
        
        float sum = 0.0f;
        int offset = b * channels * height * width + c * height * width;
        
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                sum += input[offset + h * width + w];
            }
        }
        
        output[b * channels + c] = sum / (height * width);
    }
}

torch::Tensor adaptive_avgpool_1x1_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    auto output = torch::empty({batch_size, channels, 1, 1}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * channels + block_size - 1) / block_size;
    
    adaptive_avgpool_1x1_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, channels, height, width);
    
    return output;
}
"""

adaptive_avgpool_cpp_source = "torch::Tensor adaptive_avgpool_1x1_cuda(torch::Tensor input);"

# Load custom CUDA kernels
conv_relu_cuda = load_inline(
    name="conv_relu_cuda",
    cpp_sources=conv_relu_cpp_source,
    cuda_sources=conv_relu_source,
    functions=["conv2d_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

concat_4way = load_inline(
    name="concat_4way",
    cpp_sources=concat_4way_cpp_source,
    cuda_sources=concat_4way_source,
    functions=["concat_4way_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

adaptive_avgpool = load_inline(
    name="adaptive_avgpool",
    cpp_sources=adaptive_avgpool_cpp_source,
    cuda_sources=adaptive_avgpool_source,
    functions=["adaptive_avgpool_1x1_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class InceptionModuleOptimized(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(InceptionModuleOptimized, self).__init__()
        
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
        
        self.concat_op = concat_4way
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        return self.concat_op.concat_4way_cuda(branch1x1, branch3x3, branch5x5, branch_pool)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception3a = InceptionModuleOptimized(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModuleOptimized(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception4a = InceptionModuleOptimized(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModuleOptimized(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModuleOptimized(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModuleOptimized(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModuleOptimized(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception5a = InceptionModuleOptimized(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModuleOptimized(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool_op = adaptive_avgpool
        self.fc = nn.Linear(1024, num_classes)
        
        self.conv_relu_op = conv_relu_cuda
    
    def forward(self, x):
        x = self.conv_relu_op.conv2d_relu_cuda(x, self.conv1.weight, self.conv1.bias, 2, 2, 3, 3)
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.conv_relu_op.conv2d_relu_cuda(x, self.conv3.weight, self.conv3.bias, 1, 1, 1, 1)
        x = self.maxpool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avgpool_op.adaptive_avgpool_1x1_cuda(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def get_inputs():
    batch_size = 10
    input_channels = 3
    height = 224
    width = 224
    return [torch.rand(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [1000]