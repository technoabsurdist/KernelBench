import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused squeeze+expand operations
squeezenet_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void concat_kernel(const float* input1, const float* input2, float* output, 
                             int channels1, int channels2, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_channels = channels1 + channels2;
    
    if (idx < spatial_size * total_channels) {
        int spatial_idx = idx % spatial_size;
        int channel_idx = idx / spatial_size;
        
        if (channel_idx < channels1) {
            output[idx] = input1[spatial_idx + channel_idx * spatial_size];
        } else {
            output[idx] = input2[spatial_idx + (channel_idx - channels1) * spatial_size];
        }
    }
}

__global__ void conv1x1_kernel(const float* input, const float* weight, float* output,
                              int batch_size, int in_channels, int out_channels, 
                              int height, int width) {
    int out_ch = blockIdx.x;
    int spatial_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (out_ch < out_channels && spatial_idx < height * width) {
        float sum = 0.0f;
        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
            sum += input[in_ch * height * width + spatial_idx] * 
                   weight[out_ch * in_channels + in_ch];
        }
        output[out_ch * height * width + spatial_idx] = sum;
    }
}

__global__ void conv3x3_kernel(const float* input, const float* weight, float* output,
                              int batch_size, int in_channels, int out_channels,
                              int height, int width) {
    int out_ch = blockIdx.x;
    int h = blockIdx.y;
    int w = threadIdx.x;
    
    if (out_ch < out_channels && h < height && w < width) {
        float sum = 0.0f;
        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int ih = h + kh - 1;
                    int iw = w + kw - 1;
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        sum += input[in_ch * height * width + ih * width + iw] *
                               weight[out_ch * in_channels * 9 + in_ch * 9 + kh * 3 + kw];
                    }
                }
            }
        }
        output[out_ch * height * width + h * width + w] = sum;
    }
}

torch::Tensor squeezenet_fused_forward(torch::Tensor input,
                                      torch::Tensor squeeze_weight,
                                      torch::Tensor squeeze_bias,
                                      torch::Tensor expand1x1_weight,
                                      torch::Tensor expand1x1_bias,
                                      torch::Tensor expand3x3_weight,
                                      torch::Tensor expand3x3_bias) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto squeeze_channels = squeeze_weight.size(0);
    auto expand1x1_channels = expand1x1_weight.size(0);
    auto expand3x3_channels = expand3x3_weight.size(0);
    
    // Squeeze operation
    auto squeezed = torch::zeros({batch_size, squeeze_channels, height, width}, 
                                torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Expand operations
    auto expanded1x1 = torch::zeros({batch_size, expand1x1_channels, height, width}, 
                                   torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto expanded3x3 = torch::zeros({batch_size, expand3x3_channels, height, width}, 
                                   torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Output tensor
    auto output = torch::zeros({batch_size, expand1x1_channels + expand3x3_channels, height, width}, 
                              torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    int spatial_size = height * width;
    const int block_size = 256;
    const int num_blocks_spatial = (spatial_size + block_size - 1) / block_size;
    const int num_blocks_channels = (squeeze_channels + block_size - 1) / block_size;
    
    // Process each sample in the batch
    for (int b = 0; b < batch_size; b++) {
        float* input_ptr = input.data_ptr<float>() + b * in_channels * spatial_size;
        float* squeezed_ptr = squeezed.data_ptr<float>() + b * squeeze_channels * spatial_size;
        float* expanded1x1_ptr = expanded1x1.data_ptr<float>() + b * expand1x1_channels * spatial_size;
        float* expanded3x3_ptr = expanded3x3.data_ptr<float>() + b * expand3x3_channels * spatial_size;
        float* output_ptr = output.data_ptr<float>() + b * (expand1x1_channels + expand3x3_channels) * spatial_size;
        
        // Squeeze 1x1 conv
        dim3 grid_squeeze(squeeze_channels, spatial_size/block_size + 1);
        dim3 block_squeeze(min(block_size, spatial_size));
        conv1x1_kernel<<<grid_squeeze, block_squeeze>>>(
            input_ptr, squeeze_weight.data_ptr<float>(), squeezed_ptr,
            1, in_channels, squeeze_channels, height, width
        );
        
        // Add bias and ReLU for squeeze
        relu_kernel<<<num_blocks_spatial * squeeze_channels, block_size>>>(
            squeezed_ptr, spatial_size * squeeze_channels
        );
        
        // Expand 1x1 conv
        dim3 grid_expand1x1(expand1x1_channels, spatial_size/block_size + 1);
        conv1x1_kernel<<<grid_expand1x1, block_squeeze>>>(
            squeezed_ptr, expand1x1_weight.data_ptr<float>(), expanded1x1_ptr,
            1, squeeze_channels, expand1x1_channels, height, width
        );
        
        // Add bias and ReLU for expand1x1
        relu_kernel<<<num_blocks_spatial * expand1x1_channels, block_size>>>(
            expanded1x1_ptr, spatial_size * expand1x1_channels
        );
        
        // Expand 3x3 conv
        dim3 grid_expand3x3(expand3x3_channels, height, width);
        conv3x3_kernel<<<grid_expand3x3, 1>>>(
            squeezed_ptr, expand3x3_weight.data_ptr<float>(), expanded3x3_ptr,
            1, squeeze_channels, expand3x3_channels, height, width
        );
        
        // Add bias and ReLU for expand3x3
        relu_kernel<<<num_blocks_spatial * expand3x3_channels, block_size>>>(
            expanded3x3_ptr, spatial_size * expand3x3_channels
        );
        
        // Concatenate results
        int total_output_channels = expand1x1_channels + expand3x3_channels;
        const int concat_block_size = 256;
        const int concat_num_blocks = (spatial_size * total_output_channels + concat_block_size - 1) / concat_block_size;
        concat_kernel<<<concat_num_blocks, concat_block_size>>>(
            expanded1x1_ptr, expanded3x3_ptr, output_ptr,
            expand1x1_channels, expand3x3_channels, spatial_size
        );
    }
    
    return output;
}
"""

squeezenet_fused_cpp_source = """
torch::Tensor squeezenet_fused_forward(torch::Tensor input,
                                      torch::Tensor squeeze_weight,
                                      torch::Tensor squeeze_bias,
                                      torch::Tensor expand1x1_weight,
                                      torch::Tensor expand1x1_bias,
                                      torch::Tensor expand3x3_weight,
                                      torch::Tensor expand3x3_bias);
"""

# Compile the inline CUDA code
squeezenet_fused = load_inline(
    name="squeezenet_fused",
    cpp_sources=squeezenet_fused_cpp_source,
    cuda_sources=squeezenet_fused_source,
    functions=["squeezenet_fused_forward"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        """
        :param in_channels: Number of input channels
        :param squeeze_channels: Number of output channels for the squeeze layer
        :param expand1x1_channels: Number of output channels for the 1x1 expand layer
        :param expand3x3_channels: Number of output channels for the 3x3 expand layer
        """
        super(ModelNew, self).__init__()
        
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        
        # Register the compiled CUDA module
        self.squeezenet_fused = squeezenet_fused
    
    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, expand1x1_channels + expand3x3_channels, height, width)
        """
        return self.squeezenet_fused.squeezenet_fused_forward(
            x,
            self.squeeze.weight.flatten(1),
            self.squeeze.bias,
            self.expand1x1.weight.flatten(1),
            self.expand1x1.bias,
            self.expand3x3.weight.flatten(1),
            self.expand3x3.bias
        )