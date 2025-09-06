import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    """
    Optimized model that fuses the Linear layer, ReLU, and division into a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        # We still need the nn.Linear layer to hold the parameters (weight and bias)
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor

        # Define the C++ and CUDA source code for the fused kernel
        fused_linear_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <cmath>

        __global__ void fused_linear_relu_div_kernel(
            const float* x,
            const float* weight,
            const float* bias,
            float* out,
            const float divisor,
            const int M,
            const int N,
            const int K) {

            // Each thread computes one element of the output matrix `out` (shape M x N)
            // The operation is: out = relu((x @ weight.T + bias) / divisor)
            // x: M x K
            // weight: N x K
            // bias: N
            // out: M x N

            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < M && col < N) {
                float sum = 0.0f;
                // Perform dot product
                for (int k = 0; k < K; ++k) {
                    sum += x[row * K + k] * weight[col * K + k];
                }

                // Add bias
                sum += bias[col];
                // Divide by divisor
                sum /= divisor;
                // Apply ReLU
                out[row * N + col] = fmaxf(0.0f, sum);
            }
        }

        torch::Tensor fused_linear_relu_div_cuda(
            torch::Tensor x,
            torch::Tensor weight,
            torch::Tensor bias,
            double divisor) {

            TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
            TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
            TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
            TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
            TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
            TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

            const int M = x.size(0);
            const int K = x.size(1);
            const int N = weight.size(0);

            TORCH_CHECK(K == weight.size(1), "Input feature size must match weight's input feature size");
            TORCH_CHECK(N == bias.size(0), "Weight's output feature size must match bias size");

            auto out = torch::empty({M, N}, x.options());

            const dim3 threads(16, 16);
            const dim3 blocks(
                (N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y
            );

            fused_linear_relu_div_kernel<<<blocks, threads>>>(
                x.data_ptr<float>(),
                weight.data_ptr<float>(),
                bias.data_ptr<float>(),
                out.data_ptr<float>(),
                static_cast<float>(divisor),
                M, N, K
            );

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
            }

            return out;
        }
        """

        fused_linear_cpp_source = """
        torch::Tensor fused_linear_relu_div_cuda(
            torch::Tensor x,
            torch::Tensor weight,
            torch::Tensor bias,
            double divisor);
        """

        # Compile the inline CUDA code
        self.fused_op = load_inline(
            name="fused_linear_op",
            cpp_sources=fused_linear_cpp_source,
            cuda_sources=fused_linear_source,
            functions=["fused_linear_relu_div_cuda"],
            verbose=False,
        )

    def forward(self, x):
        # Call the single fused kernel with the input and the layer's parameters
        return self.fused_op.fused_linear_relu_div_cuda(
            x, self.linear.weight, self.linear.bias, self.divisor
        )