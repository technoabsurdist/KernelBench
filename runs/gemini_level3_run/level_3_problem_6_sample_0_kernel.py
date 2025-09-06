import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for concatenating 4 tensors along the channel dimension
channel_cat_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void channel_cat_4_kernel(
    const float* in1, const float* in2, const float* in3, const float* in4,
    float* out,
    const int N, const int H, const int W,
    const int C1, const int C2, const int C3, const int C4) {

    const int C_out = C1 + C2 + C3 + C4;
    const long long total_size = (long long)N * C_out * H * W;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_size) return;

    // Calculate N, C, H, W coordinates from linear index
    const int HW = H * W;
    const long long C_out_HW = (long long)C_out * HW;

    const int w = idx % W;
    const int h = (idx / W) % H;
    const int c_out = (idx / HW) % C_out;
    const int n = idx / C_out_HW;

    float val;
    if (c_out < C1) {
        // Read from in1
        const int c_in = c_out;
        const long long C1_HW = (long long)C1 * HW;
        const long long in_idx = (long long)n * C1_HW + (long long)c_in * HW + (long long)h * W + w;
        val = in1[in_idx];
    } else if (c_out < C1 + C2) {
        // Read from in2
        const int c_in = c_out - C1;
        const long long C2_HW = (long long)C2 * HW;
        const long long in_idx = (long long)n * C2_HW + (long long)c_in * HW + (long long)h * W + w;
        val = in2[in_idx];
    } else if (c_out < C1 + C2 + C3) {
        // Read from in3
        const int c_in = c_out - (C1 + C2);
        const long long C3_HW = (long long)C3 * HW;
        const long long in_idx = (long long)n * C3_HW + (long long)c_in * HW + (long long)h * W + w;
        val = in3[in_idx];
    } else {
        // Read from in4
        const int c_in = c_out - (C1 + C2 + C3);
        const long long C4_HW = (long long)C4 * HW;
        const long long in_idx = (long long)n * C4_HW + (long long)c_in * HW + (long long)h * W + w;
        val = in4[in_idx];
    }

    out[idx] = val;
}

torch::Tensor channel_cat_4_cuda(
    torch::Tensor in1, torch::Tensor in2,
    torch::Tensor in3, torch::Tensor in4) {

    TORCH_CHECK(in1.dim() == 4, "Input 1 must be a 4D tensor");
    TORCH_CHECK(in2.dim() == 4, "Input 2 must be a 4D tensor");
    TORCH_CHECK(in3.dim() == 4, "Input 3 must be a 4D tensor");
    TORCH_CHECK(in4.dim() == 4, "Input 4 must be a 4D tensor");

    TORCH_CHECK(in1.scalar_type() == torch::kFloat32, "Inputs must be float32 tensors");
    TORCH_CHECK(in2.scalar_type() == torch::kFloat32, "Inputs must be float32 tensors");
    TORCH_CHECK(in3.scalar_type() == torch::kFloat32, "Inputs must be float32 tensors");
    TORCH_CHECK(in4.scalar_type() == torch::kFloat32, "Inputs must be float32 tensors");

    TORCH_CHECK(in1.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(in2.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(in3.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(in4.is_cuda(), "Inputs must be CUDA tensors");

    const auto N = in1.size(0);
    const auto H = in1.size(2);
    const auto W = in1.size(3);

    TORCH_CHECK(in2.size(0) == N && in2.size(2) == H && in2.size(3) == W, "All inputs must have the same N, H, W dimensions");
    TORCH_CHECK(in3.size(0) == N && in3.size(2) == H && in3.size(3) == W, "All inputs must have the same N, H, W dimensions");
    TORCH_CHECK(in4.size(0) == N && in4.size(2) == H && in4.size(3) == W, "All inputs must have the same N, H, W dimensions");

    const auto C1 = in1.size(1);
    const auto C2 = in2.size(1);
    const auto C3 = in3.size(1);
    const auto C4 = in4.size(1);
    const auto C_out = C1 + C2 + C3 + C4;

    auto out = torch::empty({N, C_out, H, W}, in1.options());
    const long long total_size = out.numel();
    if (total_size == 0) {
        return out;
    }

    const int block_size = 1024;
    const int num_blocks = (total_size + block_size - 1) / block_size;

    channel_cat_4_kernel<<<num_blocks, block_size>>>(
        in1.contiguous().data_ptr<float>(), in2.contiguous().data_ptr<float>(),
        in3.contiguous().data_ptr<float>(), in4.contiguous().data_ptr<float>(),
        out.data_ptr<float>(),
        N, H, W, C1, C2, C3, C4
    );
    
    return out;
}
"""

channel_cat_cpp_source = """
torch::Tensor channel_cat_4_cuda(
    torch::Tensor in1, torch::Tensor in2,
    torch::Tensor in3, torch::Tensor in4);
"""

# Compile the inline CUDA code
channel_cat_module = load_inline(
    name="channel_cat_module",
    cpp_sources=channel_cat_cpp_source,
    cuda_sources=channel_cat_source,
    functions=["channel_cat_4_cuda"],
    verbose=True,
)

class Model(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        """
        :param in_channels: Number of input channels
        :param out_1x1: Number of output channels for the 1x1 convolution
        :param reduce_3x3: Number of output channels for the 1x1 reduction before 3x3 convolution
        :param out_3x3: Number of output channels for the 3x3 convolution
        :param reduce_5x5: Number of output channels for the 1x1 reduction before 5x5 convolution
        :param out_5x5: Number of output channels for the 5x5 convolution
        :param pool_proj: Number of output channels for the pooling projection
        """
        super(Model, self).__init__()
        
        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        # 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        # Max pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
    
    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

class ModelNew(Model):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        """
        Initializes the Inception module with a custom CUDA kernel for the final concatenation.
        """
        super().__init__(in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj)
        self.channel_cat = channel_cat_module.channel_cat_4_cuda

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        # Use the custom CUDA kernel for concatenation instead of torch.cat
        return self.channel_cat(branch1x1, branch3x3, branch5x5, branch_pool)