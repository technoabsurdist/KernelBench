import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for a fused Linear + ReLU operation
fused_linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For fmaxf

// TILE_WIDTH determines the size of the tile processed by each thread block.
// A tile size of 16x16 is a safe choice that works well on many GPUs
// and avoids excessive shared memory usage.
#define TILE_WIDTH 16

__global__ void linear_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int M, int N, int K) {
    // Kernel to compute: output = relu(input @ weight.T + bias)
    // - input:  [M, K]
    // - weight: [N, K]
    // - bias:   [N]
    // - output: [M, N]

    // Shared memory for tiles of input and weight matrices
    __shared__ float s_input[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_weight[TILE_WIDTH][TILE_WIDTH];

    // Thread's position in the output matrix
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Thread's position within the thread block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Accumulator for the dot product
    float acc = 0.0f;

    // Loop over tiles of the K dimension
    for (int tile_k = 0; tile_k < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++tile_k) {
        // --- Load input tile into shared memory ---
        int input_k = tile_k * TILE_WIDTH + tx;
        if (row < M && input_k < K) {
            s_input[ty][tx] = input[row * K + input_k];
        } else {
            s_input[ty][tx] = 0.0f; // Pad with zero if out of bounds
        }

        // --- Load weight tile into shared memory (transposed) ---
        // We load weight[col, k] into s_weight[k, tx] to allow for
        // coalesced memory access and avoid shared memory bank conflicts during computation.
        int weight_row = col;
        int weight_col = tile_k * TILE_WIDTH + ty;
        if (weight_row < N && weight_col < K) {
            s_weight[ty][tx] = weight[weight_row * K + weight_col];
        } else {
            s_weight[ty][tx] = 0.0f; // Pad with zero
        }

        __syncthreads(); // Wait for all threads in the block to finish loading

        // --- Compute partial dot product from tiles ---
        for (int k = 0; k < TILE_WIDTH; ++k) {
            acc += s_input[ty][k] * s_weight[k][tx];
        }

        __syncthreads(); // Wait for all threads to finish computation before loading next tile
    }

    // --- Finalize: add bias and apply ReLU ---
    if (row < M && col < N) {
        acc += bias[col];
        output[row * N + col] = fmaxf(0.0f, acc); // In-place ReLU
    }
}

// C++ wrapper function to be called from Python
torch::Tensor linear_relu_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");

    // Ensure contiguous memory layout for performance
    auto input_contiguous = input.contiguous();
    
    // Get dimensions
    int M = input_contiguous.size(0);
    int K = input_contiguous.size(1);
    int N = weight.size(0);

    // Dimension compatibility checks
    TORCH_CHECK(K == weight.size(1), "Input and weight dimensions are incompatible for matmul");
    TORCH_CHECK(N == bias.size(0), "Weight and bias dimensions are incompatible");

    // Allocate output tensor
    auto output = torch::zeros({M, N}, input_contiguous.options());

    // Configure kernel launch
    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (M + TILE_WIDTH - 1) / TILE_WIDTH
    );

    // Launch the kernel
    linear_relu_kernel<<<blocks, threads>>>(
        input_contiguous.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );
    
    // Check for any errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

fused_linear_relu_cpp_source = (
    "torch::Tensor linear_relu_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# JIT compile the CUDA/C++ code
fused_linear_relu_op = load_inline(
    name="fused_linear_relu_op",
    cpp_sources=fused_linear_relu_cpp_source,
    cuda_sources=fused_linear_relu_source,
    functions=["linear_relu_forward_cuda"],
    verbose=False,
)


class FusedLinearReLU(nn.Module):
    """
    A custom nn.Module that replaces nn.Linear followed by nn.ReLU
    with a single fused CUDA kernel.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Standard PyTorch initialization for nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Call the custom CUDA operator
        return fused_linear_relu_op.linear_relu_forward_cuda(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias=True'.format(
            self.in_features, self.out_features
        )


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Initialize the VGG16 model with custom fused operators.
        
        :param num_classes: The number of output classes (default is 1000 for ImageNet)
        """
        super(ModelNew, self).__init__()
        
        # The feature extractor remains the same, as custom Conv2d is complex
        # and PyTorch's cuDNN implementation is highly optimized.
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # The classifier is optimized by replacing Linear+ReLU pairs with our custom fused layer.
        self.classifier = nn.Sequential(
            # Replace nn.Linear + nn.ReLU
            FusedLinearReLU(512 * 7 * 7, 4096),
            nn.Dropout(p=0.0),
            # Replace nn.Linear + nn.ReLU
            FusedLinearReLU(4096, 4096),
            nn.Dropout(p=0.0),
            # The final linear layer has no subsequent ReLU, so we leave it as is.
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass of the VGG16 model.
        
        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x