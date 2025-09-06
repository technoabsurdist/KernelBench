import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Fused Linear + ReLU
# This kernel performs a tiled matrix multiplication using shared memory and applies the ReLU activation.
# It computes: out = ReLU(x @ weight.T + bias)
# This fusion saves memory bandwidth by avoiding the creation of an intermediate tensor for the linear layer's output.
fused_linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Define the tile size for shared memory matrix multiplication.
// A tile size of 16x16 is a common choice that balances parallelism and resource usage.
#define TILE_WIDTH 16

__global__ void fused_linear_relu_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int M, // batch_size (rows of x)
    int N, // out_features (rows of weight)
    int K  // in_features (cols of x / cols of weight)
) {
    // Shared memory tiles for sub-matrices of x and weight.T
    __shared__ float tile_x[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_w_T[TILE_WIDTH][TILE_WIDTH];

    // Identify the row and column of the output matrix element to be computed by this thread.
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Accumulator for the dot product, stored in a register.
    float acc = 0.0f;

    // Loop over the tiles along the K dimension.
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // --- Load tiles into shared memory ---

        // Load a tile from the input matrix x.
        // Each thread loads one element: x[row, t*TILE_WIDTH + threadIdx.x]
        int x_col = t * TILE_WIDTH + threadIdx.x;
        if (row < M && x_col < K) {
            tile_x[threadIdx.y][threadIdx.x] = x[row * K + x_col];
        } else {
            tile_x[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load a tile from the transposed weight matrix (weight.T).
        // Each thread loads one element: weight.T[t*TILE_WIDTH + threadIdx.y, col]
        // which is equivalent to weight[col, t*TILE_WIDTH + threadIdx.y]
        int w_T_row = t * TILE_WIDTH + threadIdx.y;
        if (w_T_row < K && col < N) {
            tile_w_T[threadIdx.y][threadIdx.x] = weight[col * K + w_T_row];
        } else {
            tile_w_T[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure all threads in the block have finished loading their data.
        __syncthreads();

        // --- Compute dot product for the tiles ---
        // Each thread contributes to one output element by multiplying a row from tile_x
        // with a column from tile_w_T.
        for (int k = 0; k < TILE_WIDTH; ++k) {
            acc += tile_x[threadIdx.y][k] * tile_w_T[k][threadIdx.x];
        }

        // Synchronize to ensure all threads have finished with the current tiles
        // before the next iteration loads new data into shared memory.
        __syncthreads();
    }

    // --- Finalize and write the output ---
    // Check if the thread is within the bounds of the output matrix.
    if (row < M && col < N) {
        // Add the bias term.
        acc += bias[col];
        // Apply the ReLU activation function.
        out[row * N + col] = fmaxf(acc, 0.0f);
    }
}

torch::Tensor fused_linear_relu_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias
) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Input weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Input bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input x must be a float32 tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Input weight must be a float32 tensor");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Input bias must be a float32 tensor");
    TORCH_CHECK(x.dim() == 2, "Input x must be 2D");
    TORCH_CHECK(weight.dim() == 2, "Input weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "Input bias must be 1D");
    TORCH_CHECK(x.size(1) == weight.size(1), "x.size(1) and weight.size(1) must match");
    TORCH_CHECK(weight.size(0) == bias.size(0), "weight.size(0) and bias.size(0) must match");

    const auto M = x.size(0); // batch_size
    const auto N = weight.size(0); // out_features
    const auto K = x.size(1); // in_features

    auto out = torch::empty({M, N}, x.options());

    const dim3 threads(TILE_WIDTH, TILE_WIDTH);
    const dim3 blocks(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (M + TILE_WIDTH - 1) / TILE_WIDTH
    );

    fused_linear_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N, K
    );
    
    // Check for CUDA errors after kernel launch for robust error handling.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
"""

fused_linear_relu_cpp_source = """
torch::Tensor fused_linear_relu_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
"""

# JIT compile the CUDA kernel. This will be done only once when the module is imported.
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_linear_relu_cpp_source,
    cuda_sources=fused_linear_relu_source,
    functions=["fused_linear_relu_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: The number of output classes (default is 1000 for ImageNet)
        """
        super(ModelNew, self).__init__()
        
        # --- Convolutional Layers (unchanged) ---
        # Replacing these with custom kernels is complex and unlikely to beat cuDNN
        # without significant effort (e.g., Winograd, FFT convolutions).
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # --- Fully Connected Layers ---
        # We will replace the fc + relu pattern with our custom fused kernel.
        # The nn.Linear modules are kept to store the weights and biases.
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        # self.relu6 is no longer needed as it's fused into the fc1 call.
        self.dropout1 = nn.Dropout(p=0.0)
        
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        # self.relu7 is no longer needed as it's fused into the fc2 call.
        self.dropout2 = nn.Dropout(p=0.0)
        
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # --- Convolutional part (unchanged) ---
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool3(x)
        
        x = torch.flatten(x, 1)
        
        # --- Fully connected part (optimized) ---
        
        # Use the custom fused kernel for fc1 + relu6
        x = fused_op.fused_linear_relu_cuda(x, self.fc1.weight, self.fc1.bias)
        x = self.dropout1(x)
        
        # Use the custom fused kernel for fc2 + relu7
        x = fused_op.fused_linear_relu_cuda(x, self.fc2.weight, self.fc2.bias)
        x = self.dropout2(x)
        
        # The final linear layer has no activation, so we use the standard module.
        x = self.fc3(x)
        
        return x