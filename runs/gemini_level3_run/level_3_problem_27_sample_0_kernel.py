import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for Fused BatchNorm+ReLU and Global Average Pooling
custom_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel 1: Fused BatchNorm2d + ReLU (for inference)
__global__ void fused_bn_relu_kernel(
    const float* x, 
    float* y, 
    const float* gamma, 
    const float* beta, 
    const float* running_mean, 
    const float* running_var, 
    float eps, 
    int total_elements,
    int C,
    int HW) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < total_elements) {
        // Calculate channel index 'c' from the flat index
        int c = (index / HW) % C;

        // Get the input value
        float x_val = x[index];

        // Get batch norm parameters for the current channel
        float mean = running_mean[c];
        float var = running_var[c];
        float g = gamma[c];
        float b = beta[c];

        // Apply BatchNorm
        float inv_std = rsqrtf(var + eps);
        float bn_val = g * (x_val - mean) * inv_std + b;

        // Apply ReLU
        y[index] = fmaxf(0.0f, bn_val);
    }
}

torch::Tensor fused_bn_relu_cuda(
    torch::Tensor x, 
    torch::Tensor gamma, 
    torch::Tensor beta, 
    torch::Tensor running_mean, 
    torch::Tensor running_var, 
    double eps) {
    
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);

    auto y = torch::empty_like(x);
    
    const int total_elements = N * C * H * W;
    const int block_size = 1024;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_bn_relu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        total_elements,
        C,
        H * W
    );
    
    return y;
}

// Kernel 2: Global Average Pooling
__global__ void global_avg_pool_kernel(const float* input, float* output, int map_size) {
    extern __shared__ float sdata[];

    int map_idx = blockIdx.x; // Each block processes one feature map (for a given n and c)
    int tid = threadIdx.x;

    const float* map_start = input + map_idx * map_size;
    
    // Each thread sums a portion of the feature map
    float thread_sum = 0.0f;
    for (int i = tid; i < map_size; i += blockDim.x) {
        thread_sum += map_start[i];
    }
    sdata[tid] = thread_sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final result (average) to global memory
    if (tid == 0) {
        output[map_idx] = sdata[0] / map_size;
    }
}

torch::Tensor global_avg_pool_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);

    auto output = torch::empty({N, C}, x.options());

    const int num_maps = N * C;
    const int map_size = H * W;
    const int block_size = 256; // A common choice, can be tuned
    
    // Shared memory size: block_size * sizeof(float)
    size_t shared_mem_size = block_size * sizeof(float);

    global_avg_pool_kernel<<<num_maps, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        map_size
    );

    return output;
}
"""

custom_ops_cpp_source = """
torch::Tensor fused_bn_relu_cuda(
    torch::Tensor x, 
    torch::Tensor gamma, 
    torch::Tensor beta, 
    torch::Tensor running_mean, 
    torch::Tensor running_var, 
    double eps);

torch::Tensor global_avg_pool_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code
# This will be compiled on the first run and cached for subsequent runs.
custom_ops = load_inline(
    name="custom_regnet_ops",
    cpp_sources=custom_ops_cpp_source,
    cuda_sources=custom_ops_source,
    functions=["fused_bn_relu_cuda", "global_avg_pool_cuda"],
    verbose=True,
)


class CustomStage(nn.Module):
    """
    A custom stage block that replaces the sequence of
    BatchNorm2d -> ReLU with a single fused CUDA kernel.
    This is only valid during inference (model.eval()).
    """
    def __init__(self, in_channels, out_channels):
        super(CustomStage, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        # Fused BN + ReLU
        x = custom_ops.fused_bn_relu_cuda(
            x, self.bn1.weight, self.bn1.bias, 
            self.bn1.running_mean, self.bn1.running_var, self.bn1.eps
        )
        
        # Block 2
        x = self.conv2(x)
        # Fused BN + ReLU
        x = custom_ops.fused_bn_relu_cuda(
            x, self.bn2.weight, self.bn2.bias, 
            self.bn2.running_mean, self.bn2.running_var, self.bn2.eps
        )
        
        x = self.pool(x)
        return x


class ModelNew(nn.Module):
    def __init__(self, input_channels, stages, block_widths, output_classes):
        """
        :param input_channels: int, Number of input channels for the first layer
        :param stages: int, Number of stages in the RegNet architecture
        :param block_widths: List[int], Width (number of channels) for each block in the stages
        :param output_classes: int, Number of output classes for classification
        """
        super(ModelNew, self).__init__()

        self.stages = stages
        self.block_widths = block_widths
        
        layers = []
        current_channels = input_channels
        
        # Construct the stages with their respective blocks using the custom stage
        for i in range(stages):
            layers.append(CustomStage(current_channels, block_widths[i]))
            current_channels = block_widths[i]
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Final fully connected layer for classification
        self.fc = nn.Linear(block_widths[-1], output_classes)

    def forward(self, x):
        """
        Forward pass through the optimized RegNet model.
        :param x: torch.Tensor of shape (batch_size, input_channels, height, width)
        :return: torch.Tensor of shape (batch_size, output_classes)
        """
        x = self.feature_extractor(x)
        # Use custom Global Average Pooling kernel
        x = custom_ops.global_avg_pool_cuda(x)
        x = self.fc(x)
        return x