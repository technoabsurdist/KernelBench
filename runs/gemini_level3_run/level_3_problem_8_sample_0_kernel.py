import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for fused operations
# We will create two fused kernels:
# 1. fused_bn_relu: Combines BatchNorm and ReLU. Used after the first convolution.
# 2. fused_bn_add_relu: Combines BatchNorm, residual addition, and ReLU. Used after the second convolution.
# This approach reduces memory bandwidth by eliminating intermediate tensors and reduces kernel launch overhead.

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Grid-stride loop helper function to ensure all elements are processed
#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void fused_bn_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bn_weight, // gamma
    const float* __restrict__ bn_bias,   // beta
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    const float bn_eps,
    float* __restrict__ out,
    const int C, const int H, const int W) {

    const int total_elements = C * H * W * blockDim.y; // N is blockDim.y
    const int H_W = H * W;

    CUDA_KERNEL_LOOP(i, total_elements) {
        // Calculate channel index
        const int c = (i / H_W) % C;

        // Load BN params for this channel
        const float gamma = bn_weight[c];
        const float beta = bn_bias[c];
        const float mean = bn_mean[c];
        const float var = bn_var[c];

        // Compute inv_stddev
        const float inv_stddev = rsqrtf(var + bn_eps);

        // Apply BN
        float val = gamma * (x[i] - mean) * inv_stddev + beta;

        // Apply ReLU
        out[i] = fmaxf(0.f, val);
    }
}

__global__ void fused_bn_add_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ identity,
    const float* __restrict__ bn_weight, // gamma
    const float* __restrict__ bn_bias,   // beta
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    const float bn_eps,
    float* __restrict__ out,
    const int C, const int H, const int W) {

    const int total_elements = C * H * W * blockDim.y; // N is blockDim.y
    const int H_W = H * W;

    CUDA_KERNEL_LOOP(i, total_elements) {
        // Calculate channel index
        const int c = (i / H_W) % C;

        // Load BN params for this channel
        const float gamma = bn_weight[c];
        const float beta = bn_bias[c];
        const float mean = bn_mean[c];
        const float var = bn_var[c];

        // Compute inv_stddev
        const float inv_stddev = rsqrtf(var + bn_eps);

        // Apply BN
        float val = gamma * (x[i] - mean) * inv_stddev + beta;

        // Add identity
        val += identity[i];

        // Apply ReLU
        out[i] = fmaxf(0.f, val);
    }
}

torch::Tensor fused_bn_relu_cuda(
    torch::Tensor x,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    double bn_eps) {

    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    // Add more checks for other tensors as needed

    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);
    const auto total_elements = x.numel();

    auto out = torch::empty_like(x);

    const int block_size_x = 256;
    const int grid_size_x = (total_elements + block_size_x - 1) / block_size_x;
    
    dim3 threads(block_size_x, N); // Use blockDim.y for batch size N
    dim3 blocks(grid_size_x / N); // Adjust grid size accordingly

    fused_bn_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        static_cast<float>(bn_eps),
        out.data_ptr<float>(),
        C, H, W
    );
    return out;
}

torch::Tensor fused_bn_add_relu_cuda(
    torch::Tensor x,
    torch::Tensor identity,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    double bn_eps) {

    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(identity.is_contiguous(), "Identity tensor must be contiguous");
    // Add more checks for other tensors as needed

    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);
    const auto total_elements = x.numel();

    auto out = torch::empty_like(x);

    const int block_size_x = 256;
    const int grid_size_x = (total_elements + block_size_x - 1) / block_size_x;

    dim3 threads(block_size_x, N); // Use blockDim.y for batch size N
    dim3 blocks(grid_size_x / N); // Adjust grid size accordingly

    fused_bn_add_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        identity.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        static_cast<float>(bn_eps),
        out.data_ptr<float>(),
        C, H, W
    );
    return out;
}
"""

cpp_source = """
torch::Tensor fused_bn_relu_cuda(torch::Tensor x, torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var, double bn_eps);
torch::Tensor fused_bn_add_relu_cuda(torch::Tensor x, torch::Tensor identity, torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var, double bn_eps);
"""

# Compile the inline CUDA code
# This might take a moment the first time it's run.
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_bn_relu_cuda", "fused_bn_add_relu_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride for the first convolutional layer
        """
        super(ModelNew, self).__init__()
        # Define the layers exactly as in the original model to load weights correctly
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # The original ReLU is no longer needed as it's fused
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # The downsample path is left unchanged for simplicity, though it could also be optimized
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        else:
            self.downsample = None
            
        self.stride = stride

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        # Ensure model is in eval mode for custom kernels to work correctly with running stats
        if self.training:
            raise RuntimeError("ModelNew with custom CUDA kernels only supports inference mode (eval()).")

        identity = x

        # --- First fused block: Conv -> BN -> ReLU ---
        out = self.conv1(x)
        # Instead of bn1(out) and relu(out), call the single fused kernel
        out = fused_ops.fused_bn_relu_cuda(
            out,
            self.bn1.weight,
            self.bn1.bias,
            self.bn1.running_mean,
            self.bn1.running_var,
            self.bn1.eps,
        )

        # --- Second block ---
        out = self.conv2(out)

        # --- Downsample path (unchanged) ---
        if self.downsample is not None:
            identity = self.downsample(x)

        # --- Second fused block: BN -> Add -> ReLU ---
        # Instead of bn2(out), out += identity, relu(out), call the single fused kernel
        out = fused_ops.fused_bn_add_relu_cuda(
            out,
            identity,
            self.bn2.weight,
            self.bn2.bias,
            self.bn2.running_mean,
            self.bn2.running_var,
            self.bn2.eps,
        )

        return out