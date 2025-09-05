import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused convolution operations
inception_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void conv1x1_kernel(const float* input, const float* weight, float* output,
                               int batch_size, int in_channels, int out_channels,
                               int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * height * width;
    
    if (idx < total_elements) {
        int w = idx % width;
        int h = (idx / width) % height;
        int out_c = (idx / (width * height)) % out_channels;
        int b = idx / (width * height * out_channels);
        
        float sum = 0.0f;
        for (int in_c = 0; in_c < in_channels; in_c++) {
            sum += input[((b * in_channels + in_c) * height + h) * width + w] * 
                   weight[out_c * in_channels + in_c];
        }
        output[idx] = sum;
    }
}

__global__ void conv3x3_kernel(const float* input, const float* weight, float* output,
                               int batch_size, int in_channels, int out_channels,
                               int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * height * width;
    
    if (idx < total_elements) {
        int w = idx % width;
        int h = (idx / width) % height;
        int out_c = (idx / (width * height)) % out_channels;
        int b = idx / (width * height * out_channels);
        
        float sum = 0.0f;
        for (int in_c = 0; in_c < in_channels; in_c++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int ih = h + kh - 1;
                    int iw = w + kw - 1;
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        float val = input[((b * in_channels + in_c) * height + ih) * width + iw];
                        float wgt = weight[((out_c * in_channels + in_c) * 3 + kh) * 3 + kw];
                        sum += val * wgt;
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

__global__ void conv5x5_kernel(const float* input, const float* weight, float* output,
                               int batch_size, int in_channels, int out_channels,
                               int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * height * width;
    
    if (idx < total_elements) {
        int w = idx % width;
        int h = (idx / width) % height;
        int out_c = (idx / (width * height)) % out_channels;
        int b = idx / (width * height * out_channels);
        
        float sum = 0.0f;
        for (int in_c = 0; in_c < in_channels; in_c++) {
            for (int kh = 0; kh < 5; kh++) {
                for (int kw = 0; kw < 5; kw++) {
                    int ih = h + kh - 2;
                    int iw = w + kw - 2;
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        float val = input[((b * in_channels + in_c) * height + ih) * width + iw];
                        float wgt = weight[(((out_c * in_channels + in_c) * 5 + kh) * 5) + kw];
                        sum += val * wgt;
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

__global__ void maxpool3x3_kernel(const float* input, float* output,
                                  int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        int w = idx % width;
        int h = (idx / width) % height;
        int c = (idx / (width * height)) % channels;
        int b = idx / (width * height * channels);
        
        float max_val = -1e38f;
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                int ih = h + kh - 1;
                int iw = w + kw - 1;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    float val = input[((b * channels + c) * height + ih) * width + iw];
                    max_val = fmaxf(max_val, val);
                }
            }
        }
        output[idx] = max_val;
    }
}

torch::Tensor inception_branch1x1_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_elements = batch_size * out_channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv1x1_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, out_channels, height, width
    );
    
    return output;
}

torch::Tensor inception_branch3x3_cuda(torch::Tensor input, torch::Tensor weight_reduce, 
                                       torch::Tensor weight_conv) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto reduce_channels = weight_reduce.size(0);
    auto out_channels = weight_conv.size(0);
    
    // First 1x1 convolution
    auto reduced = torch::zeros({batch_size, reduce_channels, height, width}, 
                                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_elements = batch_size * reduce_channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv1x1_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight_reduce.data_ptr<float>(),
        reduced.data_ptr<float>(), batch_size, in_channels, reduce_channels, height, width
    );
    
    // Then 3x3 convolution
    auto output = torch::zeros({batch_size, out_channels, height, width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    total_elements = batch_size * out_channels * height * width;
    num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv3x3_kernel<<<num_blocks, block_size>>>(
        reduced.data_ptr<float>(), weight_conv.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, reduce_channels, out_channels, height, width
    );
    
    return output;
}

torch::Tensor inception_branch5x5_cuda(torch::Tensor input, torch::Tensor weight_reduce, 
                                       torch::Tensor weight_conv) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto reduce_channels = weight_reduce.size(0);
    auto out_channels = weight_conv.size(0);
    
    // First 1x1 convolution
    auto reduced = torch::zeros({batch_size, reduce_channels, height, width}, 
                                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_elements = batch_size * reduce_channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv1x1_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight_reduce.data_ptr<float>(),
        reduced.data_ptr<float>(), batch_size, in_channels, reduce_channels, height, width
    );
    
    // Then 5x5 convolution
    auto output = torch::zeros({batch_size, out_channels, height, width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    total_elements = batch_size * out_channels * height * width;
    num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv5x5_kernel<<<num_blocks, block_size>>>(
        reduced.data_ptr<float>(), weight_conv.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, reduce_channels, out_channels, height, width
    );
    
    return output;
}

torch::Tensor inception_branch_pool_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    
    // Max pooling
    auto pooled = torch::zeros({batch_size, in_channels, height, width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_elements = batch_size * in_channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    maxpool3x3_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), pooled.data_ptr<float>(),
        batch_size, in_channels, height, width
    );
    
    // 1x1 convolution
    auto output = torch::zeros({batch_size, out_channels, height, width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    total_elements = batch_size * out_channels * height * width;
    num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv1x1_kernel<<<num_blocks, block_size>>>(
        pooled.data_ptr<float>(), weight.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, out_channels, height, width
    );
    
    return output;
}
"""

inception_cpp_source = """
torch::Tensor inception_branch1x1_cuda(torch::Tensor input, torch::Tensor weight);
torch::Tensor inception_branch3x3_cuda(torch::Tensor input, torch::Tensor weight_reduce, torch::Tensor weight_conv);
torch::Tensor inception_branch5x5_cuda(torch::Tensor input, torch::Tensor weight_reduce, torch::Tensor weight_conv);
torch::Tensor inception_branch_pool_cuda(torch::Tensor input, torch::Tensor weight);
"""

# Compile the inline CUDA code for inception operations
inception_ops = load_inline(
    name="inception_ops",
    cpp_sources=inception_cpp_source,
    cuda_sources=inception_cuda_source,
    functions=["inception_branch1x1_cuda", "inception_branch3x3_cuda", 
               "inception_branch5x5_cuda", "inception_branch_pool_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        """
        :param in_channels: Number of input channels
        :param out_1x1: Number of output channels for the 1x1 convolution
        :param reduce_3x3: Number of output channels for the 1x1 reduction before 3x3 convolution
        :param out_3x3: Number of output channels for the 3x3 convolution
        :param reduce_5x5: Number of output channels for the 1x1 reduction before 5x5 convolution
        :param out_5x5: Number of output channels for the 5x5 convolution
        :param pool_proj: Number of output channels for the pooling projection
        """
        super(ModelNew, self).__init__()
        
        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # 3x3 convolution branch
        self.branch3x3_reduce = nn.Conv2d(in_channels, reduce_3x3, kernel_size=1)
        self.branch3x3_conv = nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        
        # 5x5 convolution branch
        self.branch5x5_reduce = nn.Conv2d(in_channels, reduce_5x5, kernel_size=1)
        self.branch5x5_conv = nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        
        # Max pooling branch
        self.branch_pool_conv = nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        
        # Load custom CUDA operations
        self.inception_ops = inception_ops
    
    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        # 1x1 convolution branch
        branch1x1 = self.inception_ops.inception_branch1x1_cuda(x, self.branch1x1.weight)
        if self.branch1x1.bias is not None:
            branch1x1 += self.branch1x1.bias.view(1, -1, 1, 1)
        
        # 3x3 convolution branch
        branch3x3 = self.inception_ops.inception_branch3x3_cuda(
            x, self.branch3x3_reduce.weight, self.branch3x3_conv.weight)
        if self.branch3x3_conv.bias is not None:
            branch3x3 += self.branch3x3_conv.bias.view(1, -1, 1, 1)
        
        # 5x5 convolution branch
        branch5x5 = self.inception_ops.inception_branch5x5_cuda(
            x, self.branch5x5_reduce.weight, self.branch5x5_conv.weight)
        if self.branch5x5_conv.bias is not None:
            branch5x5 += self.branch5x5_conv.bias.view(1, -1, 1, 1)
        
        # Max pooling branch
        branch_pool = self.inception_ops.inception_branch_pool_cuda(x, self.branch_pool_conv.weight)
        if self.branch_pool_conv.bias is not None:
            branch_pool += self.branch_pool_conv.bias.view(1, -1, 1, 1)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)