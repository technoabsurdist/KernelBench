import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for depthwise convolution
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void depthwise_conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    bool has_bias
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_ch = blockIdx.z * blockDim.z + threadIdx.z;

    if (out_x >= out_width || out_y >= out_height || out_ch >= out_channels)
        return;

    float sum = 0.0f;
    for (int ky = 0; ky < kernel_h; ky++) {
        for (int kx = 0; kx < kernel_w; kx++) {
            int in_y = out_y * stride_h - padding_h + ky * dilation_h;
            int in_x = out_x * stride_w - padding_w + kx * dilation_w;

            if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                int input_idx = ((/*batch*/ 0 * in_channels + out_ch) * in_height + in_y) * in_width + in_x;
                int weight_idx = (out_ch * kernel_h + ky) * kernel_w + kx;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    if (has_bias) {
        sum += bias[out_ch];
    }

    int output_idx = ((/*batch*/ 0 * out_channels + out_ch) * out_height + out_y) * out_width + out_x;
    output[output_idx] = sum;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    
    auto out_channels = weight.size(0);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);
    
    auto out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    auto out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    if (batch_size == 0) return output;
    
    dim3 block(16, 16, 4);
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        (out_channels + block.z - 1) / block.z
    );
    
    bool has_bias = bias.defined() && bias.numel() > 0;
    
    for (int batch = 0; batch < batch_size; batch++) {
        depthwise_conv2d_kernel<<<grid, block>>>(
            input[batch].data_ptr<float>(),
            weight.data_ptr<float>(),
            has_bias ? bias.data_ptr<float>() : nullptr,
            output[batch].data_ptr<float>(),
            1,
            in_channels,
            in_height,
            in_width,
            out_channels,
            out_height,
            out_width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            has_bias
        );
    }
    
    return output;
}
"""

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w
);
"""

# Compile the inline CUDA code for depthwise convolution
depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        self.bias_flag = bias
        
        # For depthwise convolution, out_channels should equal in_channels
        assert out_channels == in_channels, "For depthwise convolution, out_channels must equal in_channels"
        assert groups == in_channels, "For depthwise convolution, groups must equal in_channels"
        
        # Initialize weight parameters
        self.weight = nn.Parameter(torch.randn(out_channels, 1, kernel_size_h, kernel_size_w))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.depthwise_conv2d = depthwise_conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depthwise_conv2d.depthwise_conv2d_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.empty(0, device=x.device),
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.dilation_h,
            self.dilation_w
        )