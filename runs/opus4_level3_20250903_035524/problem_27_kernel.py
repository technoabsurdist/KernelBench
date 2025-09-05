import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Fused BatchNorm2d + ReLU CUDA kernel
fused_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void fused_bn_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ output,
    int N, int C, int H, int W, float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;
    
    if (idx < total_elements) {
        int c = (idx / (H * W)) % C;
        
        float mean = running_mean[c];
        float var = running_var[c];
        float scale = gamma[c];
        float bias = beta[c];
        
        float normalized = (input[idx] - mean) / sqrtf(var + eps);
        float result = normalized * scale + bias;
        
        // Fused ReLU
        output[idx] = fmaxf(result, 0.0f);
    }
}

torch::Tensor fused_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps) {
    
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    
    auto output = torch::zeros_like(input);
    
    int total_elements = N * C * H * W;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_bn_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, eps
    );
    
    return output;
}
"""

fused_bn_relu_cpp_source = """
torch::Tensor fused_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps);
"""

# Global Average Pooling CUDA kernel
global_avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void global_avg_pool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W) {
    
    int batch = blockIdx.x;
    int channel = blockIdx.y;
    
    if (batch < N && channel < C) {
        float sum = 0.0f;
        int offset = batch * C * H * W + channel * H * W;
        
        for (int i = threadIdx.x; i < H * W; i += blockDim.x) {
            sum += input[offset + i];
        }
        
        __shared__ float shared_sum[256];
        shared_sum[threadIdx.x] = sum;
        __syncthreads();
        
        // Reduction within block
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            }
            __syncthreads();
        }
        
        if (threadIdx.x == 0) {
            output[batch * C + channel] = shared_sum[0] / (H * W);
        }
    }
}

torch::Tensor global_avg_pool_cuda(torch::Tensor input) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    
    auto output = torch::zeros({N, C}, input.options());
    
    dim3 grid(N, C);
    dim3 block(256);
    
    global_avg_pool_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W
    );
    
    return output;
}
"""

global_avg_pool_cpp_source = """
torch::Tensor global_avg_pool_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
fused_bn_relu = load_inline(
    name="fused_bn_relu",
    cpp_sources=fused_bn_relu_cpp_source,
    cuda_sources=fused_bn_relu_source,
    functions=["fused_bn_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

global_avg_pool = load_inline(
    name="global_avg_pool",
    cpp_sources=global_avg_pool_cpp_source,
    cuda_sources=global_avg_pool_source,
    functions=["global_avg_pool_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class FusedBatchNormReLU(nn.Module):
    def __init__(self, num_features):
        super(FusedBatchNormReLU, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = 1e-5
        self.momentum = 0.1
        
    def forward(self, x):
        if self.training:
            # Use standard BatchNorm for training (to compute running stats)
            mean = x.mean(dim=[0, 2, 3])
            var = x.var(dim=[0, 2, 3], unbiased=False)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            x = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
            return F.relu(x)
        else:
            # Use fused kernel for inference
            return fused_bn_relu.fused_bn_relu_cuda(
                x.contiguous(), 
                self.weight, 
                self.bias, 
                self.running_mean, 
                self.running_var,
                self.eps
            )

class ModelNew(nn.Module):
    def __init__(self, input_channels, stages, block_widths, output_classes):
        super(ModelNew, self).__init__()
        
        self.stages = stages
        self.block_widths = block_widths
        
        layers = []
        current_channels = input_channels
        
        for i in range(stages):
            layers.append(self._make_stage(current_channels, block_widths[i]))
            current_channels = block_widths[i]
        
        self.feature_extractor = nn.Sequential(*layers)
        self.fc = nn.Linear(block_widths[-1], output_classes)
        self.global_avg_pool = global_avg_pool
    
    def _make_stage(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            FusedBatchNormReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            FusedBatchNormReLU(out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.global_avg_pool.global_avg_pool_cuda(x.contiguous())
        x = self.fc(x)
        return x