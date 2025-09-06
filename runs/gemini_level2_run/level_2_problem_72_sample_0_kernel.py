import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused BatchNorm3d (inference) and AvgPool3d
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Helper for indexing 5D tensors (N, C, D, H, W)
#define TENSOR_IDX(n, c, d, h, w, C, D, H, W) \
    ((((((n) * (C) + (c)) * (D) + (d)) * (H) + (h)) * (W) + (w)))

__global__ void batch_norm_avg_pool_3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ output,
    const float eps,
    const int N, const int C, const int iD, const int iH, const int iW,
    const int oD, const int oH, const int oW,
    const int k, const int s) {

    // Calculate output indices from thread/block indices
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc_od_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (ow >= oW || oh >= oH || nc_od_idx >= N * C * oD) {
        return;
    }

    // De-flatten the nc_od_idx to get batch, channel, and output depth indices
    const int od = nc_od_idx % oD;
    const int nc_idx = nc_od_idx / oD;
    const int c = nc_idx % C;
    const int n = nc_idx / C;

    // Get batch norm parameters for the current channel
    const float mean = running_mean[c];
    const float var = running_var[c];
    const float g = gamma[c];
    const float b = beta[c];

    // Pre-calculate the scale and shift for batch norm to reduce operations in the loop
    // y = (x - mean) / sqrt(var + eps) * gamma + beta
    // y = x * [gamma / sqrt(var + eps)] + [beta - mean * gamma / sqrt(var + eps)]
    // y = x * scale + shift
    const float inv_std = 1.0f / sqrtf(var + eps);
    const float scale = g * inv_std;
    const float shift = b - mean * scale;

    // Determine the start of the pooling window in the input tensor
    const int id_start = od * s;
    const int ih_start = oh * s;
    const int iw_start = ow * s;

    float sum = 0.0f;
    for (int kd = 0; kd < k; ++kd) {
        for (int kh = 0; kh < k; ++kh) {
            for (int kw = 0; kw < k; ++kw) {
                const int id = id_start + kd;
                const int ih = ih_start + kh;
                const int iw = iw_start + kw;

                const long long input_idx = TENSOR_IDX(n, c, id, ih, iw, C, iD, iH, iW);
                const float val = input[input_idx];

                // Apply batch norm transformation
                const float norm_val = val * scale + shift;

                sum += norm_val;
            }
        }
    }

    const float pool_size = (float)(k * k * k);
    const float avg = sum / pool_size;

    const long long output_idx = TENSOR_IDX(n, c, od, oh, ow, C, oD, oH, oW);
    output[output_idx] = avg;
}

torch::Tensor batch_norm_avg_pool_3d_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps,
    int kernel_size,
    int stride) {

    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 5, "Input must be a 5D tensor");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be a 1D tensor");
    TORCH_CHECK(beta.dim() == 1, "beta must be a 1D tensor");
    TORCH_CHECK(running_mean.dim() == 1, "running_mean must be a 1D tensor");
    TORCH_CHECK(running_var.dim() == 1, "running_var must be a 1D tensor");

    const int N = input.size(0);
    const int C = input.size(1);
    const int iD = input.size(2);
    const int iH = input.size(3);
    const int iW = input.size(4);

    // Calculate output dimensions for average pooling
    const int oD = (iD - kernel_size) / stride + 1;
    const int oH = (iH - kernel_size) / stride + 1;
    const int oW = (iW - kernel_size) / stride + 1;

    auto output = torch::zeros({N, C, oD, oH, oW}, input.options());

    // Use a 3D grid and 3D blocks for better mapping to the problem
    const dim3 threads(8, 8, 4); // 256 threads per block
    const dim3 blocks(
        (oW + threads.x - 1) / threads.x,
        (oH + threads.y - 1) / threads.y,
        (N * C * oD + threads.z - 1) / threads.z
    );

    batch_norm_avg_pool_3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(eps),
        N, C, iD, iH, iW,
        oD, oH, oW,
        kernel_size, stride
    );
    
    // Check for any kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

fused_op_cpp_source = """
torch::Tensor batch_norm_avg_pool_3d_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps,
    int kernel_size,
    int stride);
"""

# JIT compile the custom CUDA kernel
fused_batch_norm_avg_pool_3d = load_inline(
    name="fused_batch_norm_avg_pool_3d",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["batch_norm_avg_pool_3d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    An optimized model that fuses the 3D batch normalization and the first 
    average pooling layer into a single custom CUDA kernel.
    This custom kernel is designed for inference-time optimization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        
        # The second average pooling layer remains a standard PyTorch operator.
        # The default stride for AvgPool3d is equal to the kernel_size.
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2, stride=2)

        # Manually define the parameters and buffers for the fused BatchNorm3d layer.
        # This mimics the behavior of a standard nn.BatchNorm3d layer with
        # affine=True and track_running_stats=True.
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer('bn_running_mean', torch.zeros(out_channels))
        self.register_buffer('bn_running_var', torch.ones(out_channels))
        self.bn_eps = 1e-5  # Default epsilon for nn.BatchNorm3d

        # Store the loaded custom CUDA function
        self.fused_op = fused_batch_norm_avg_pool_3d

    def forward(self, x):
        x = self.conv_transpose(x)
        
        # Call the custom CUDA kernel to perform fused BatchNorm3d and AvgPool3d.
        # We use kernel_size=2 and stride=2 to match the original avg_pool1.
        x = self.fused_op.batch_norm_avg_pool_3d_cuda(
            x,
            self.bn_weight,
            self.bn_bias,
            self.bn_running_mean,
            self.bn_running_var,
            self.bn_eps,
            2,  # kernel_size for the first avg_pool
            2   # stride for the first avg_pool
        )
        
        x = self.avg_pool2(x)
        return x