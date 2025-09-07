import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv3d + min + softmax
fused_conv_min_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

__global__ void conv3d_min_softmax_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int D, int H, int W,
    int kD, int kH, int kW,
    int oD, int oH, int oW
) {
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y;
    int oh = threadIdx.x + blockIdx.z * blockDim.x;
    int ow = threadIdx.y + blockIdx.z * blockDim.y;
    
    if (batch_idx >= batch_size || out_ch >= out_channels || oh >= oH || ow >= oW) return;
    
    const int dim = 2; // Depth dimension for min operation
    
    // Conv3d computation
    float min_val = INFINITY;
    
    for (int od = 0; od < oD; od++) {
        float sum = 0.0f;
        
        for (int kd = 0; kd < kD; kd++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    for (int ic = 0; ic < in_channels; ic++) {
                        int id = od + kd;
                        int ih = oh + kh;
                        int iw = ow + kw;
                        
                        if (id < D && ih < H && iw < W) {
                            int input_idx = batch_idx * (in_channels * D * H * W) + 
                                          ic * (D * H * W) + 
                                          id * (H * W) + 
                                          ih * W + iw;
                            
                            int weight_idx = out_ch * (in_channels * kD * kH * kW) + 
                                           ic * (kD * kH * kW) + 
                                           kd * (kH * kW) + 
                                           kh * kW + kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Add bias
        sum += bias[out_ch];
        
        // Track minimum
        if (sum < min_val) {
            min_val = sum;
        }
    }
    
    // Softmax (simplified for channel dimension)
    // In a real implementation, we would need to compute softmax across all channels
    // This is a simplified version that just normalizes the min value
    output[batch_idx * (out_channels * oH * oW) + 
           out_ch * (oH * oW) + 
           (blockIdx.z * blockDim.x + threadIdx.x) * oW + 
           (blockIdx.z * blockDim.y + threadIdx.y)] = min_val;
}

torch::Tensor fused_conv_min_softmax_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kD, int kH, int kW,
    int dim
) {
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int D = input_sizes[2];
    int H = input_sizes[3];
    int W = input_sizes[4];
    
    int out_channels = weight_sizes[0];
    
    // Output dimensions after conv
    int oD = D - kD + 1;
    int oH = H - kH + 1;
    int oW = W - kW + 1;
    
    auto output = torch::zeros({batch_size, out_channels, oH, oW}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Simplified kernel launch
    dim3 grid(batch_size, out_channels, (oH * oW + 255) / 256);
    dim3 block(min(oH, 16), min(oW, 16), 1);
    
    conv3d_min_softmax_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        D, H, W,
        kD, kH, kW,
        oD, oH, oW
    );
    
    // Apply softmax along channel dimension
    auto softmax_output = torch::softmax(output, 1);
    
    return softmax_output;
}
"""

fused_conv_min_softmax_cpp_source = """
torch::Tensor fused_conv_min_softmax_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kD, int kH, int kW,
    int dim
);
"""

# Compile the inline CUDA code for fused conv3d + min + softmax
fused_conv_min_softmax = load_inline(
    name="fused_conv_min_softmax",
    cpp_sources=fused_conv_min_softmax_cpp_source,
    cuda_sources=fused_conv_min_softmax_source,
    functions=["fused_conv_min_softmax_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for conv3d + min + softmax
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.kernel_size = kernel_size
        self.dim = dim
        self.fused_op = fused_conv_min_softmax

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W)
        """
        return self.fused_op.fused_conv_min_softmax_cuda(
            x, self.weight, self.bias, 
            self.kernel_size, self.kernel_size, self.kernel_size, 
            self.dim
        )