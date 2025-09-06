import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused BatchNorm2d (inference) and Tanh
fused_batch_norm_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel for applying BatchNorm2d (in inference mode) followed by Tanh activation
__global__ void batch_norm_tanh_kernel(
    const float* x,
    const float* mean,
    const float* var,
    const float* weight,
    const float* bias,
    float* out,
    int N, int C, int H, int W,
    float eps) {

    int plane_size = H * W;
    int total_size = N * C * plane_size;

    // Using a grid-stride loop to ensure all elements are processed
    // regardless of the number of blocks launched.
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_size; idx += blockDim.x * gridDim.x) {
        // Calculate the channel index to fetch the correct per-channel parameters
        int c = (idx / plane_size) % C;

        // Load per-channel parameters
        float mean_val = mean[c];
        float var_val = var[c];
        float weight_val = weight[c];
        float bias_val = bias[c];

        // Compute the inverse standard deviation
        float inv_std = rsqrtf(var_val + eps);

        // Apply the BatchNorm transformation
        float normalized = (x[idx] - mean_val) * inv_std;
        float scaled = normalized * weight_val + bias_val;
        
        // Apply the Tanh activation and store the result
        out[idx] = tanhf(scaled);
    }
}

// C++ wrapper function that will be called from Python
torch::Tensor batch_norm_tanh_cuda(
    torch::Tensor x,
    torch::Tensor mean,
    torch::Tensor var,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps) {

    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(mean.is_cuda() && mean.is_contiguous(), "Mean must be a contiguous CUDA tensor");
    TORCH_CHECK(var.is_cuda() && var.is_contiguous(), "Var must be a contiguous CUDA tensor");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(), "Weight must be a contiguous CUDA tensor");
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous(), "Bias must be a contiguous CUDA tensor");

    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);
    const auto total_size = x.numel();

    // Allocate output tensor
    auto out = torch::empty_like(x);

    // Kernel launch configuration
    const int block_size = 256;
    // Use a heuristic for the number of blocks, can be tuned for specific hardware
    const int num_blocks = std::min((int)((total_size + block_size - 1) / block_size), 4096);

    // Launch the kernel
    batch_norm_tanh_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        static_cast<float>(eps)
    );
    
    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

# C++ source for the function signature
fused_batch_norm_tanh_cpp_source = (
    "torch::Tensor batch_norm_tanh_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, torch::Tensor weight, torch::Tensor bias, double eps);"
)

# Compile the inline CUDA code
fused_batch_norm_tanh = load_inline(
    name="fused_batch_norm_tanh",
    cpp_sources=fused_batch_norm_tanh_cpp_source,
    cuda_sources=fused_batch_norm_tanh_source,
    functions=["batch_norm_tanh_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that replaces BatchNorm2d and Tanh with a single fused CUDA kernel.
    The other operators (ConvTranspose2d, MaxPool2d, GroupNorm) remain as standard PyTorch layers
    as they are already highly optimized in cuDNN. The main benefit here comes from reducing
    kernel launch overhead and memory bandwidth by fusing the element-wise operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        
        # We still instantiate the BatchNorm2d layer to hold its parameters and buffers
        # (weight, bias, running_mean, running_var), which are needed by our custom kernel.
        self.batch_norm = nn.BatchNorm2d(out_channels)
        
        # IMPORTANT: The custom kernel implements the inference-time behavior of BatchNorm.
        # We must set the module to eval mode to ensure running_mean and running_var are used.
        self.batch_norm.eval()
        
        # The Tanh activation is now part of the fused kernel, so we don't need a separate nn.Tanh layer.
        
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        
        # Replace the sequence of batch_norm and tanh with a single call to our fused CUDA kernel.
        # We pass the necessary parameters and buffers from our self.batch_norm layer.
        x = fused_batch_norm_tanh.batch_norm_tanh_cuda(
            x,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.eps
        )
        
        x = self.max_pool(x)
        x = self.group_norm(x)
        return x

batch_size = 512
in_channels  = 64  
out_channels = 128  
height = width = 2048  
kernel_size  = 5
stride       = 1  
padding      = 1
groups       = 8
num_groups   = 8
height, width = 32, 32

def get_inputs():
    # Ensure input is on CUDA device
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, num_groups]