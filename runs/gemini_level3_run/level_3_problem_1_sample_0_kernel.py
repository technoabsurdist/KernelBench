import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernels and C++ wrapper functions
fused_linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_WIDTH 32

// Kernel for Fused Linear (Matmul + Bias) + ReLU
__global__ void fused_linear_relu_kernel(
    const float* x, const float* weight, const float* bias,
    float* out, int M, int N, int K) {

    // Shared memory for tiles of input x and weight matrix
    __shared__ float x_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float weight_tile[TILE_WIDTH][TILE_WIDTH];

    // Thread indices to calculate the output element's row and column
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Accumulator for the output element, stored in a register
    float acc = 0.0f;

    // Loop over the tiles of the input matrices
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Load a tile of x from global memory to shared memory
        int x_col = t * TILE_WIDTH + threadIdx.x;
        if (row < M && x_col < K) {
            x_tile[threadIdx.y][threadIdx.x] = x[row * K + x_col];
        } else {
            x_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load a tile of weight.T from global memory to shared memory
        // The weight matrix is (N, K), we access it as if it's transposed
        int w_row = t * TILE_WIDTH + threadIdx.y;
        if (w_row < K && col < N) {
            weight_tile[threadIdx.y][threadIdx.x] = weight[col * K + w_row];
        } else {
            weight_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads(); // Synchronize to ensure all data is loaded

        // Compute the dot product for the tiles
        for (int i = 0; i < TILE_WIDTH; ++i) {
            acc += x_tile[threadIdx.y][i] * weight_tile[i][threadIdx.x];
        }

        __syncthreads(); // Synchronize before loading the next tile
    }

    // Write the final result to the output tensor in global memory
    if (row < M && col < N) {
        acc += bias[col]; // Add bias
        out[row * N + col] = fmaxf(0.0f, acc); // Apply ReLU
    }
}

// Kernel for Fused Linear (Matmul + Bias) only
__global__ void fused_linear_kernel(
    const float* x, const float* weight, const float* bias,
    float* out, int M, int N, int K) {

    __shared__ float x_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float weight_tile[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        int x_col = t * TILE_WIDTH + threadIdx.x;
        if (row < M && x_col < K) {
            x_tile[threadIdx.y][threadIdx.x] = x[row * K + x_col];
        } else {
            x_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int w_row = t * TILE_WIDTH + threadIdx.y;
        if (w_row < K && col < N) {
            weight_tile[threadIdx.y][threadIdx.x] = weight[col * K + w_row];
        } else {
            weight_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            acc += x_tile[threadIdx.y][i] * weight_tile[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        acc += bias[col]; // Add bias
        out[row * N + col] = acc; // No ReLU
    }
}

// C++ function to launch the fused_linear_relu_kernel
torch::Tensor fused_linear_relu_forward(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(x.size(1) == weight.size(1), "x and weight feature dimensions must match");
    TORCH_CHECK(weight.size(0) == bias.size(0), "weight and bias output dimensions must match");

    const int M = x.size(0); // batch_size
    const int N = weight.size(0); // output_features
    const int K = x.size(1); // input_features

    auto out = torch::empty({M, N}, x.options());

    const dim3 threads(TILE_WIDTH, TILE_WIDTH);
    const dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    fused_linear_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N, K
    );
    return out;
}

// C++ function to launch the fused_linear_kernel
torch::Tensor fused_linear_forward(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(x.size(1) == weight.size(1), "x and weight feature dimensions must match");
    TORCH_CHECK(weight.size(0) == bias.size(0), "weight and bias output dimensions must match");

    const int M = x.size(0);
    const int N = weight.size(0);
    const int K = x.size(1);

    auto out = torch::empty({M, N}, x.options());

    const dim3 threads(TILE_WIDTH, TILE_WIDTH);
    const dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    fused_linear_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N, K
    );
    return out;
}
"""

fused_linear_cpp_source = """
torch::Tensor fused_linear_relu_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
torch::Tensor fused_linear_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code using torch.utils.cpp_extension
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_linear_cpp_source,
    cuda_sources=fused_linear_source,
    functions=["fused_linear_relu_forward", "fused_linear_forward"],
    verbose=True,
)

# Define a custom autograd Function for the fused Linear + ReLU operation
class FusedLinearReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        # Call the custom CUDA kernel for the forward pass
        out = fused_ops.fused_linear_relu_forward(x, weight, bias)
        # Save tensors needed for the backward pass
        ctx.save_for_backward(x, weight, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, out = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None

        # Gradient for ReLU: derivative is 1 for positive values, 0 otherwise
        grad_output_relu = grad_output.clone()
        grad_output_relu[out == 0] = 0

        # Gradients for Linear layer, computed using standard PyTorch ops
        if ctx.needs_input_grad[0]:
            grad_x = grad_output_relu.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output_relu.t().matmul(x)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output_relu.sum(0)

        return grad_x, grad_weight, grad_bias

# Define a custom autograd Function for the fused Linear operation
class FusedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        # Call the custom CUDA kernel for the forward pass
        out = fused_ops.fused_linear_forward(x, weight, bias)
        # Save tensors needed for the backward pass
        ctx.save_for_backward(x, weight)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None

        # Gradients for Linear layer, computed using standard PyTorch ops
        if ctx.needs_input_grad[0]:
            grad_x = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(x)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_x, grad_weight, grad_bias


class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()
        
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        
        all_sizes = [input_size] + layer_sizes + [output_size]
        
        # Create and initialize parameters for each layer
        for i in range(len(all_sizes) - 1):
            in_features = all_sizes[i]
            out_features = all_sizes[i+1]
            
            weight = nn.Parameter(torch.Tensor(out_features, in_features))
            bias = nn.Parameter(torch.Tensor(out_features))
            
            # Initialize parameters similar to nn.Linear's default (Kaiming uniform)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias, -bound, bound)
            
            self.weights.append(weight)
            self.biases.append(bias)
            
        self.num_hidden_layers = len(layer_sizes)

    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Ensure input is on the same device and dtype as model parameters
        x = x.to(self.weights[0].device, self.weights[0].dtype)

        # Apply fused Linear + ReLU for all hidden layers
        for i in range(self.num_hidden_layers):
            x = FusedLinearReLUFunction.apply(x, self.weights[i], self.biases[i])
        
        # Apply the final fused Linear layer (without ReLU)
        x = FusedLinearFunction.apply(x, self.weights[-1], self.biases[-1])
        
        return x