import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused conv3d output processing with division
conv_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_div_kernel(float* output, const float divisor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = output[idx] / divisor;
    }
}

void conv_div_cuda(torch::Tensor output, float divisor) {
    int size = output.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    conv_div_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), 
        divisor, 
        size
    );
}
"""

# Custom CUDA kernel for fused maxpool + global avg pool
fused_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void fused_maxpool_avgpool_kernel(
    const float* input, float* output,
    int batch, int channels, int D, int H, int W,
    int pool_d, int pool_h, int pool_w) {
    
    int b = blockIdx.x;
    int c = blockIdx.y;
    
    if (b >= batch || c >= channels) return;
    
    int out_D = D / pool_d;
    int out_H = H / pool_h;
    int out_W = W / pool_w;
    
    float sum = 0.0f;
    int count = 0;
    
    for (int od = 0; od < out_D; od++) {
        for (int oh = 0; oh < out_H; oh++) {
            for (int ow = 0; ow < out_W; ow++) {
                float max_val = -FLT_MAX;
                
                for (int pd = 0; pd < pool_d; pd++) {
                    for (int ph = 0; ph < pool_h; ph++) {
                        for (int pw = 0; pw < pool_w; pw++) {
                            int id = od * pool_d + pd;
                            int ih = oh * pool_h + ph;
                            int iw = ow * pool_w + pw;
                            
                            if (id < D && ih < H && iw < W) {
                                int idx = b * (channels * D * H * W) + 
                                         c * (D * H * W) + 
                                         id * (H * W) + 
                                         ih * W + iw;
                                max_val = fmaxf(max_val, input[idx]);
                            }
                        }
                    }
                }
                
                sum += max_val;
                count++;
            }
        }
    }
    
    output[b * channels + c] = sum / count;
}

torch::Tensor fused_maxpool_avgpool_cuda(
    torch::Tensor input, int pool_d, int pool_h, int pool_w) {
    
    auto batch = input.size(0);
    auto channels = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    
    auto output = torch::zeros({batch, channels, 1, 1, 1}, input.options());
    
    dim3 blocks(batch, channels);
    
    fused_maxpool_avgpool_kernel<<<blocks, 1>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch, channels, D, H, W,
        pool_d, pool_h, pool_w
    );
    
    return output;
}
"""

# Custom CUDA kernel for fused bias addition and sum
bias_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bias_sum_kernel(
    const float* input, const float* bias, float* output,
    int batch, int channels) {
    
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b < batch) {
        float sum = 0.0f;
        for (int c = 0; c < channels; c++) {
            sum += input[b * channels + c] + bias[c];
        }
        output[b] = sum;
    }
}

torch::Tensor bias_sum_cuda(torch::Tensor input, torch::Tensor bias) {
    auto batch = input.size(0);
    auto channels = input.size(1);
    
    auto output = torch::zeros({batch}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch + block_size - 1) / block_size;
    
    bias_sum_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch, channels
    );
    
    return output;
}
"""

conv_div_cpp_source = "void conv_div_cuda(torch::Tensor output, float divisor);"
fused_pool_cpp_source = "torch::Tensor fused_maxpool_avgpool_cuda(torch::Tensor input, int pool_d, int pool_h, int pool_w);"
bias_sum_cpp_source = "torch::Tensor bias_sum_cuda(torch::Tensor input, torch::Tensor bias);"

# Compile the inline CUDA code
conv_div = load_inline(
    name="conv_div",
    cpp_sources=conv_div_cpp_source,
    cuda_sources=conv_div_source,
    functions=["conv_div_cuda"],
    verbose=True,
)

fused_pool = load_inline(
    name="fused_pool",
    cpp_sources=fused_pool_cpp_source,
    cuda_sources=fused_pool_source,
    functions=["fused_maxpool_avgpool_cuda"],
    verbose=True,
)

bias_sum = load_inline(
    name="bias_sum",
    cpp_sources=bias_sum_cpp_source,
    cuda_sources=bias_sum_source,
    functions=["bias_sum_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernels for fused operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.pool_size = pool_size
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        
        self.conv_div = conv_div
        self.fused_pool = fused_pool
        self.bias_sum = bias_sum

    def forward(self, x):
        x = self.conv(x)
        
        # Fused division
        self.conv_div.conv_div_cuda(x, self.divisor)
        
        # Fused maxpool + global avg pool
        x = self.fused_pool.fused_maxpool_avgpool_cuda(
            x, self.pool_size[0], self.pool_size[1], self.pool_size[2]
        )
        
        # Reshape for bias_sum kernel
        x = x.view(x.size(0), x.size(1))
        bias_flat = self.bias.view(-1)
        
        # Fused bias addition and sum
        x = self.bias_sum.bias_sum_cuda(x, bias_flat)
        
        return x

batch_size   = 128  
in_channels  = 8            
out_channels = 16  
depth = 16; height = width = 64 
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]