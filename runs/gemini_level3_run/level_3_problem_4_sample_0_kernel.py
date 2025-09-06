import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for fused operations
# 1. Fused ReLU + MaxPool2d
# 2. Fused Linear + ReLU
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

// Fused Kernel: Applies ReLU activation and then performs 2x2 Max Pooling with stride 2
__global__ void fused_relu_maxpool2d_kernel(const float* input, float* output, int N, int C, int H_in, int W_in) {
    int h_out = (H_in + 1) / 2;
    int w_out = (W_in + 1) / 2;

    // Calculate the output index for this thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_elements = N * C * h_out * w_out;

    if (idx < total_out_elements) {
        // Decompose the 1D index into 4D coordinates (n, c, h, w) for the output tensor
        int w_idx_out = idx % w_out;
        int h_idx_out = (idx / w_out) % h_out;
        int c_idx = (idx / (w_out * h_out)) % C;
        int n_idx = idx / (C * w_out * h_out);

        // Determine the 2x2 window in the input tensor
        int h_start = h_idx_out * 2;
        int w_start = w_idx_out * 2;

        float max_val = -FLT_MAX;

        // Iterate over the 2x2 window
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                int current_h = h_start + i;
                int current_w = w_start + j;

                if (current_h < H_in && current_w < W_in) {
                    // Calculate the 1D index for the input tensor
                    int input_idx = n_idx * (C * H_in * W_in) + c_idx * (H_in * W_in) + current_h * W_in + current_w;
                    
                    // Apply ReLU on the fly and find the max
                    float val = fmaxf(0.0f, input[input_idx]);
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
        }
        output[idx] = max_val;
    }
}

// Fused Kernel: Performs Linear transformation (Y = XA^T + B) and applies ReLU
__global__ void fused_linear_relu_kernel(const float* input, const float* weight, const float* bias, float* output, int M, int N, int K) {
    // Each thread computes one element of the output matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Corresponds to M
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Corresponds to N

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += input[row * K + k] * weight[col * K + k];
        }
        sum += bias[col];
        output[row * N + col] = fmaxf(0.0f, sum);
    }
}

// C++ wrapper for fused_relu_maxpool2d
torch::Tensor fused_relu_maxpool2d_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");

    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H_in = input.size(2);
    const auto W_in = input.size(3);

    const int H_out = (H_in + 1) / 2;
    const int W_out = (W_in + 1) / 2;

    auto output = torch::zeros({N, C, H_out, W_out}, input.options());
    
    const int total_elements = N * C * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_relu_maxpool2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        N, C, H_in, W_in
    );
    
    return output;
}

// C++ wrapper for fused_linear_relu
torch::Tensor fused_linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(), "All inputs must be CUDA tensors");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Input and weight must be contiguous");
    TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, "Input and weight must be 2D tensors");

    const auto M = input.size(0);
    const auto K = input.size(1);
    const auto N = weight.size(0);
    TORCH_CHECK(weight.size(1) == K, "Matrix dimensions are not compatible for multiplication");

    auto output = torch::zeros({M, N}, input.options());

    const dim3 threads(16, 16);
    const dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    fused_linear_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    return output;
}
"""

cpp_source = """
torch::Tensor fused_relu_maxpool2d_cuda(torch::Tensor input);
torch::Tensor fused_linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
custom_fused_ops = load_inline(
    name="custom_fused_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_relu_maxpool2d_cuda", "fused_linear_relu_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, num_classes):
        """
        LeNet-5 architecture with custom fused CUDA kernels.

        :param num_classes: The number of output classes.
        """
        super(ModelNew, self).__init__()
        
        # Convolutional layers (weights are managed by PyTorch)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # Fully connected layers (weights are managed by PyTorch)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
    
    def forward(self, x):
        """
        Forward pass of the LeNet-5 model using custom kernels.

        :param x: The input tensor, shape (batch_size, 1, 32, 32)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # First convolutional layer followed by fused ReLU + MaxPool
        x = self.conv1(x)
        x = custom_fused_ops.fused_relu_maxpool2d_cuda(x)
        
        # Second convolutional layer followed by fused ReLU + MaxPool
        x = self.conv2(x)
        x = custom_fused_ops.fused_relu_maxpool2d_cuda(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 16*5*5)
        
        # First fully connected layer with fused Linear + ReLU
        x = custom_fused_ops.fused_linear_relu_cuda(x, self.fc1.weight, self.fc1.bias)
        
        # Second fully connected layer with fused Linear + ReLU
        x = custom_fused_ops.fused_linear_relu_cuda(x, self.fc2.weight, self.fc2.bias)
        
        # Final fully connected layer (no ReLU, so use standard PyTorch op)
        x = self.fc3(x)
        
        return x

# Test code for the LeNet-5 model (larger batch & image)
batch_size = 4096
num_classes = 20

def get_inputs():
    return [torch.rand(batch_size, 1, 32, 32).cuda()]

def get_init_inputs():
    return [num_classes]