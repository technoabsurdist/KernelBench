import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA and C++ source code for the fused ConvTranspose3d + ReLU kernel
fused_conv_transpose3d_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel for ConvTranspose3d + ReLU
// This implementation is specific to stride=1, padding=0, dilation=1.
// It computes the "full" cross-correlation, which is equivalent to PyTorch's
// ConvTranspose3d with these parameters.
__global__ void conv_transpose3d_relu_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    float* __restrict__ output,
    const int N, const int C_in, const int D_in, const int H_in, const int W_in,
    const int C_out, const int D_out, const int H_out, const int W_out,
    const int kD, const int kH, const int kW) {

    // Calculate output indices from thread and block IDs
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int z_idx = blockIdx.z;

    if (ow >= W_out || oh >= H_out) {
        return;
    }

    // Unpack z_idx to get batch, output channel, and depth indices
    const int od = z_idx % D_out;
    const int c_out = (z_idx / D_out) % C_out;
    const int n = z_idx / (D_out * C_out);

    float acc = 0.0f;

    // Iterate over input channels and kernel dimensions
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kd = 0; kd < kD; ++kd) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    // Calculate corresponding input coordinates for the "full" correlation
                    const int id = od - kd;
                    const int ih = oh - kh;
                    const int iw = ow - kw;

                    // Check if the input coordinates are within bounds
                    if (id >= 0 && id < D_in &&
                        ih >= 0 && ih < H_in &&
                        iw >= 0 && iw < W_in) {

                        // Calculate flat indices for input and weight tensors
                        // Assumes contiguous tensors (checked in the C++ wrapper)
                        const long long input_idx = (long long)n * C_in * D_in * H_in * W_in +
                                                    (long long)c_in * D_in * H_in * W_in +
                                                    (long long)id * H_in * W_in +
                                                    (long long)ih * W_in +
                                                    iw;

                        // PyTorch weight layout for ConvTranspose3d: (C_in, C_out, kD, kH, kW)
                        const long long weight_idx = (long long)c_in * C_out * kD * kH * kW +
                                                     (long long)c_out * kD * kH * kW +
                                                     (long long)kd * kH * kW +
                                                     (long long)kh * kW +
                                                     kw;

                        acc += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Apply ReLU activation
    const float result = fmaxf(acc, 0.0f);

    // Calculate flat index for output tensor and write the result
    const long long output_idx = (long long)n * C_out * D_out * H_out * W_out +
                                 (long long)c_out * D_out * H_out * W_out +
                                 (long long)od * H_out * W_out +
                                 (long long)oh * W_out +
                                 ow;
    output[output_idx] = result;
}

// C++ wrapper function to be bound with PyTorch
torch::Tensor conv_transpose3d_relu_cuda(torch::Tensor input, torch::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "Weight must be a float32 tensor");

    // Ensure tensors are contiguous for simple indexing
    input = input.contiguous();
    weight = weight.contiguous();

    // Get tensor dimensions
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    // Weight dimensions: (C_in, C_out, kD, kH, kW)
    TORCH_CHECK(weight.dim() == 5, "Weight must be a 5D tensor");
    TORCH_CHECK(weight.size(0) == C_in, "Weight C_in dimension mismatch");
    const int C_out = weight.size(1);
    const int kD = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);

    // Calculate output dimensions (for stride=1, padding=0)
    const int D_out = D_in + kD - 1;
    const int H_out = H_in + kH - 1;
    const int W_out = W_in + kW - 1;

    // Create the output tensor
    auto output = torch::empty({N, C_out, D_out, H_out, W_out}, input.options());

    // Define kernel launch configuration
    const int THREADS_X = 16;
    const int THREADS_Y = 16;
    const dim3 threads(THREADS_X, THREADS_Y, 1);
    const dim3 blocks((W_out + THREADS_X - 1) / THREADS_X,
                      (H_out + THREADS_Y - 1) / THREADS_Y,
                      (long long)N * C_out * D_out);
    
    // Launch the kernel
    conv_transpose3d_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        kD, kH, kW
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error: ", cudaGetErrorString(err));

    return output;
}
"""

fused_conv_transpose3d_relu_cpp_source = """
torch::Tensor conv_transpose3d_relu_cuda(torch::Tensor input, torch::Tensor weight);
"""

# JIT compile the custom CUDA kernel
fused_conv_relu = load_inline(
    name="fused_conv_relu",
    cpp_sources=fused_conv_transpose3d_relu_cpp_source,
    cuda_sources=fused_conv_transpose3d_relu_source,
    functions=["conv_transpose3d_relu_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses the transposed 3D convolution and ReLU activation
    into a single custom CUDA kernel. The Group Normalization is applied afterwards
    using the standard PyTorch operator.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        
        # The custom kernel assumes stride=1, padding=0, and bias=False, which
        # matches the defaults and the provided architecture.
        if bias:
            raise NotImplementedError("Custom kernel does not support bias=True")

        # We instantiate the original ConvTranspose3d layer primarily to manage
        # the weight parameter (initialization, device placement, etc.).
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        
        # The GroupNorm layer remains unchanged.
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

        # Store the compiled custom operator.
        self.fused_conv_relu_op = fused_conv_relu

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        # Apply the fused ConvTranspose3d + ReLU operation using the custom kernel.
        # We pass the input tensor and the weight parameter from our conv_transpose layer.
        x = self.fused_conv_relu_op.conv_transpose3d_relu_cuda(x, self.conv_transpose.weight)
        
        # Apply the standard GroupNorm operation.
        x = self.group_norm(x)
        return x