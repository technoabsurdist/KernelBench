import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Fused BatchNorm + ReLU kernel
fused_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void fused_bn_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ output,
    int N, int C, int HW,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * HW;
    
    if (idx < total_elements) {
        int c = (idx / HW) % C;
        
        float mean = running_mean[c];
        float var = running_var[c];
        float scale = gamma[c];
        float bias = beta[c];
        
        float x = input[idx];
        float normalized = (x - mean) / sqrtf(var + eps);
        float bn_out = normalized * scale + bias;
        output[idx] = fmaxf(bn_out, 0.0f);  // ReLU
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
    auto HW = H * W;
    
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (N * C * HW + threads - 1) / threads;
    
    fused_bn_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, HW, eps
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

# Optimized concatenation kernel for DenseBlock
concat_dense_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void concat_dense_kernel(
    float** inputs,
    float* output,
    int* channel_offsets,
    int num_inputs,
    int N, int H, int W,
    int total_channels) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * total_channels * H * W;
    
    if (idx < total_elements) {
        int n = idx / (total_channels * H * W);
        int c = (idx / (H * W)) % total_channels;
        int hw = idx % (H * W);
        
        // Find which input tensor this channel belongs to
        int input_idx = 0;
        int local_c = c;
        for (int i = 0; i < num_inputs; i++) {
            if (local_c < channel_offsets[i]) {
                input_idx = i;
                break;
            }
            local_c -= channel_offsets[i];
        }
        
        int input_idx_flat = n * channel_offsets[input_idx] * H * W + local_c * H * W + hw;
        output[idx] = inputs[input_idx][input_idx_flat];
    }
}

torch::Tensor concat_dense_cuda(std::vector<torch::Tensor> inputs) {
    int num_inputs = inputs.size();
    auto N = inputs[0].size(0);
    auto H = inputs[0].size(2);
    auto W = inputs[0].size(3);
    
    std::vector<int> channel_counts;
    int total_channels = 0;
    for (const auto& inp : inputs) {
        channel_counts.push_back(inp.size(1));
        total_channels += inp.size(1);
    }
    
    auto output = torch::empty({N, total_channels, H, W}, inputs[0].options());
    
    // Allocate device memory for pointers and offsets
    float** d_inputs;
    int* d_offsets;
    cudaMalloc(&d_inputs, num_inputs * sizeof(float*));
    cudaMalloc(&d_offsets, num_inputs * sizeof(int));
    
    std::vector<float*> h_inputs;
    for (const auto& inp : inputs) {
        h_inputs.push_back(inp.data_ptr<float>());
    }
    
    cudaMemcpy(d_inputs, h_inputs.data(), num_inputs * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, channel_counts.data(), num_inputs * sizeof(int), cudaMemcpyHostToDevice);
    
    const int threads = 256;
    const int blocks = (N * total_channels * H * W + threads - 1) / threads;
    
    concat_dense_kernel<<<blocks, threads>>>(
        d_inputs, output.data_ptr<float>(), d_offsets,
        num_inputs, N, H, W, total_channels
    );
    
    cudaFree(d_inputs);
    cudaFree(d_offsets);
    
    return output;
}
"""

concat_dense_cpp_source = """
torch::Tensor concat_dense_cuda(std::vector<torch::Tensor> inputs);
"""

# Load custom kernels
fused_bn_relu = load_inline(
    name="fused_bn_relu",
    cpp_sources=fused_bn_relu_cpp_source,
    cuda_sources=fused_bn_relu_source,
    functions=["fused_bn_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

concat_dense = load_inline(
    name="concat_dense",
    cpp_sources=concat_dense_cpp_source,
    cuda_sources=concat_dense_source,
    functions=["concat_dense_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class FusedBatchNormReLU(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = 1e-5
        self.momentum = 0.1
        
    def forward(self, x):
        if self.training:
            mean = x.mean([0, 2, 3])
            var = x.var([0, 2, 3], unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            x = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
            return F.relu(x, inplace=True)
        else:
            return fused_bn_relu.fused_bn_relu_cuda(
                x, self.weight, self.bias, 
                self.running_mean, self.running_var, self.eps
            )

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            FusedBatchNormReLU(in_features),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            if self.training:
                x = torch.cat(features, 1)
            else:
                x = concat_dense.concat_dense_cuda(features)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer, self).__init__()
        self.bn_relu = FusedBatchNormReLU(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn_relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelNew, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_relu1 = FusedBatchNormReLU(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        num_features = 64
        block_layers = [6, 12, 24, 16]

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn_relu1(x)
        x = self.maxpool(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.final_bn(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_inputs():
    batch_size = 10
    height, width = 224, 224
    return [torch.rand(batch_size, 3, height, width)]

def get_init_inputs():
    return [32, 10]