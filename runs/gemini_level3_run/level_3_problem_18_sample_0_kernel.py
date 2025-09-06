import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# 1. Define the custom CUDA kernel source
# This kernel performs a fused Conv2d + ReLU operation.
# It's a naive direct convolution implementation for demonstration.
# Grid: (W_out, H_out, N) -> One block per output pixel in the batch
# Block: (C_out) -> One thread per output channel
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Helper for 4D tensor access in NCHW format
#define GET_4D_INDEX(n, c, h, w, C, H, W) (((n) * (C) + (c)) * (H) + (h)) * (W) + (w)

__global__ void conv2d_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int H_out, const int W_out,
    const int K, const int stride, const int padding
) {
    // Calculate global thread indices from grid and block dimensions
    const int w_out = blockIdx.x;
    const int h_out = blockIdx.y;
    const int n = blockIdx.z;
    const int c_out = threadIdx.x;

    // Check if we are within the bounds of the output tensor
    if (w_out >= W_out || h_out >= H_out || n >= N || c_out >= C_out) {
        return;
    }

    // Calculate starting position in the input tensor for this output pixel
    const int h_start = h_out * stride - padding;
    const int w_start = w_out * stride - padding;

    // Accumulator for the convolution, initialized with bias
    float acc = bias[c_out];

    // Loop over input channels
    for (int c_in = 0; c_in < C_in; ++c_in) {
        // Loop over kernel dimensions
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                const int h_in = h_start + kh;
                const int w_in = w_start + kw;

                // Boundary check for valid input pixels (handles padding)
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    // Input tensor index
                    int input_idx = GET_4D_INDEX(n, c_in, h_in, w_in, C_in, H_in, W_in);
                    // Weight tensor index
                    int weight_idx = GET_4D_INDEX(c_out, c_in, kh, kw, C_in, K, K);
                    
                    acc += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Apply ReLU activation function
    float result = fmaxf(0.f, acc);

    // Output tensor index
    int output_idx = GET_4D_INDEX(n, c_out, h_out, w_out, C_out, H_out, W_out);
    output[output_idx] = result;
}

// C++ wrapper function to be called from Python
torch::Tensor conv2d_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor");
    TORCH_CHECK(bias.dim() == 1, "Bias must be a 1D tensor");

    // Get tensor dimensions
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    const int C_out = weight.size(0);
    const int K = weight.size(2); // Kernel size

    TORCH_CHECK(C_in == weight.size(1), "Input channels must match weight's input channels");
    TORCH_CHECK(C_out == bias.size(0), "Output channels must match bias size");
    TORCH_CHECK(K == weight.size(3), "Kernel must be square");

    // Calculate output dimensions
    const int H_out = (H_in + 2 * padding - K) / stride + 1;
    const int W_out = (W_in + 2 * padding - K) / stride + 1;

    // Create the output tensor
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    // Configure and launch the kernel
    dim3 threads(C_out);
    dim3 blocks(W_out, H_out, N);
    
    TORCH_CHECK(C_out <= 1024, "Number of output channels exceeds max CUDA block size of 1024");

    conv2d_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K, stride, padding
    );
    
    // Check for any CUDA errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

cpp_source = "torch::Tensor conv2d_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding);"

# 2. Compile the CUDA kernel inline
fused_conv_relu = load_inline(
    name="fused_conv_relu",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv2d_relu_cuda"],
    verbose=False,
)

# 3. Define the new FireModule using the custom CUDA operator
class FireModuleNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModuleNew, self).__init__()
        
        # Store the compiled CUDA function
        self.fused_op = fused_conv_relu

        # Manually define parameters for each conceptual layer
        self.squeeze_weight = nn.Parameter(torch.empty(squeeze_channels, in_channels, 1, 1))
        self.squeeze_bias = nn.Parameter(torch.empty(squeeze_channels))
        
        self.expand1x1_weight = nn.Parameter(torch.empty(expand1x1_channels, squeeze_channels, 1, 1))
        self.expand1x1_bias = nn.Parameter(torch.empty(expand1x1_channels))
        
        self.expand3x3_weight = nn.Parameter(torch.empty(expand3x3_channels, squeeze_channels, 3, 3))
        self.expand3x3_bias = nn.Parameter(torch.empty(expand3x3_channels))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Replicate the default initialization of nn.Conv2d
        nn.init.kaiming_uniform_(self.squeeze_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.expand1x1_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.expand3x3_weight, a=math.sqrt(5))
        
        for w, b in [(self.squeeze_weight, self.squeeze_bias), 
                     (self.expand1x1_weight, self.expand1x1_bias), 
                     (self.expand3x3_weight, self.expand3x3_bias)]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(b, -bound, bound)

    def forward(self, x):
        # Call the custom fused Conv2d+ReLU kernel for each operation
        x = self.fused_op.conv2d_relu_cuda(x, self.squeeze_weight, self.squeeze_bias, stride=1, padding=0)
        
        out_1x1 = self.fused_op.conv2d_relu_cuda(x, self.expand1x1_weight, self.expand1x1_bias, stride=1, padding=0)
        out_3x3 = self.fused_op.conv2d_relu_cuda(x, self.expand3x3_weight, self.expand3x3_bias, stride=1, padding=1)
        
        return torch.cat([out_1x1, out_3x3], 1)

# 4. Define the final ModelNew architecture, replacing FireModule with FireModuleNew
class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # The feature extractor part of the network
        # We keep standard PyTorch operators for layers that are not part of the FireModule,
        # as they are already highly optimized (e.g., using cuDNN).
        # The main optimization comes from replacing the most frequent block (FireModule).
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModuleNew(96, 16, 64, 64),
            FireModuleNew(128, 16, 64, 64),
            FireModuleNew(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModuleNew(256, 32, 128, 128),
            FireModuleNew(256, 48, 192, 192),
            FireModuleNew(384, 48, 192, 192),
            FireModuleNew(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModuleNew(512, 64, 256, 256),
        )
        
        # The classifier part of the network
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)