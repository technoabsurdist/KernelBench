import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Fused BatchNorm + Add + ReLU
# This kernel replaces the sequence:
#   out = bn3(out)
#   out += identity
#   out = relu(out)
# with a single, more efficient operation. This reduces memory bandwidth
# by avoiding writing intermediate tensors (the output of bn3 and the addition)
# to global memory, and also reduces kernel launch overhead.
fused_bn_add_relu_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Template to handle both float and half precision data
template <typename T>
__global__ void fused_bn_add_relu_kernel(
    const T* conv_out,
    const T* identity,
    const float* bn_weight,     // gamma
    const float* bn_bias,       // beta
    const float* bn_running_mean,
    const float* bn_running_var,
    const float bn_eps,
    T* final_out,
    const int N, const int C, const int H, const int W) {

    // Using a grid-stride loop to process all elements, regardless of grid size
    const int total_elements = N * C * H * W;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_elements;
         i += gridDim.x * blockDim.x) {

        // Decompose the 1D index `i` to 4D (n, c, h, w) to find the channel index `c`
        int c = (i / (W * H)) % C;

        // 1. Read input values from global memory
        T x = conv_out[i];
        T id_val = identity[i];

        // 2. Read channel-specific BatchNorm parameters
        float mean = bn_running_mean[c];
        float var = bn_running_var[c];
        float gamma = bn_weight[c];
        float beta = bn_bias[c];

        // 3. Perform the fused operations in registers
        // 3a. BatchNorm (inference mode)
        // Calculation is done in float for precision, then cast back to T
        float inv_std = rsqrtf(var + bn_eps);
        T bn_out = (T)(gamma * ((float)x - mean) * inv_std + beta);

        // 3b. Element-wise Add
        T sum_val = bn_out + id_val;

        // 3c. ReLU
        T final_val = sum_val > (T)0.0f ? sum_val : (T)0.0f;

        // 4. Write the final result to global memory
        final_out[i] = final_val;
    }
}

// C++ wrapper function that will be called from Python
torch::Tensor fused_bn_add_relu_cuda(
    torch::Tensor conv_out,
    torch::Tensor identity,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    double bn_eps) {

    // Get tensor dimensions
    const int N = conv_out.size(0);
    const int C = conv_out.size(1);
    const int H = conv_out.size(2);
    const int W = conv_out.size(3);

    // Create an empty output tensor with the same properties as the input
    auto final_out = torch::empty_like(conv_out);

    // Configure kernel launch parameters
    const int total_elements = N * C * H * W;
    const int block_size = 256;
    // Use a modest number of blocks to ensure good occupancy and allow the grid-stride loop to work
    const int num_blocks = std::min((total_elements + block_size - 1) / block_size, 4096);

    // Dispatch the templated kernel based on the input tensor's data type (float or half)
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(conv_out.scalar_type(), "fused_bn_add_relu_kernel", ([&] {
        fused_bn_add_relu_kernel<scalar_t><<<num_blocks, block_size>>>(
            conv_out.data_ptr<scalar_t>(),
            identity.data_ptr<scalar_t>(),
            bn_weight.data_ptr<float>(),
            bn_bias.data_ptr<float>(),
            bn_running_mean.data_ptr<float>(),
            bn_running_var.data_ptr<float>(),
            (float)bn_eps,
            final_out.data_ptr<scalar_t>(),
            N, C, H, W
        );
    }));

    return final_out;
}
"""

fused_bn_add_relu_cpp_source = """
torch::Tensor fused_bn_add_relu_cuda(
    torch::Tensor conv_out,
    torch::Tensor identity,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    double bn_eps);
"""

class BottleneckNew(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, fused_op=None):
        super(BottleneckNew, self).__init__()
        if fused_op is None:
            raise ValueError("A fused CUDA operator must be provided.")
        self.fused_op = fused_op

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Replace the standard PyTorch sequence with our single fused kernel call
        # Original:
        #   out = self.bn3(out)
        #   out += identity
        #   out = self.relu(out)
        # Fused:
        out = self.fused_op(
            out,
            identity,
            self.bn3.weight,
            self.bn3.bias,
            self.bn3.running_mean,
            self.bn3.running_var,
            self.bn3.eps
        )

        return out

class ModelNew(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super(ModelNew, self).__init__()

        # Compile the inline CUDA code when the model is instantiated.
        # This module will be passed to the BottleneckNew blocks.
        self.fused_op_module = load_inline(
            name="fused_bn_add_relu",
            cpp_sources=fused_bn_add_relu_cpp_source,
            cuda_sources=fused_bn_add_relu_source,
            functions=["fused_bn_add_relu_cuda"],
            verbose=False, # Set to True for debugging compilation
        )

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = BottleneckNew  # Use the new Bottleneck with the fused op

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # Pass the compiled CUDA function to the BottleneckNew constructor
        layers.append(block(self.in_channels, out_channels, stride, downsample, fused_op=self.fused_op_module.fused_bn_add_relu_cuda))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, fused_op=self.fused_op_module.fused_bn_add_relu_cuda))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x