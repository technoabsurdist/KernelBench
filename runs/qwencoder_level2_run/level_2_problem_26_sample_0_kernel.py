import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv3d_transpose + add + hardswish
conv3d_transpose_add_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void conv3d_transpose_add_hardswish_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* add_input,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w
) {
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y;
    int out_d = blockIdx.z / (output_h * output_w);
    int out_h = (blockIdx.z % (output_h * output_w)) / output_w;
    int out_w = blockIdx.z % output_w;
    
    if (batch_idx >= batch_size || out_ch >= out_channels || 
        out_d >= output_d || out_h >= output_h || out_w >= output_w) {
        return;
    }
    
    float sum = 0.0f;
    
    // Conv3d transpose operation
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int kd = 0; kd < kernel_d; kd++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    // Calculate corresponding input position
                    int in_d = out_d - kd + padding_d;
                    int in_h = out_h - kh + padding_h;
                    int in_w = out_w - kw + padding_w;
                    
                    // Check if within valid input range after accounting for stride
                    if (in_d % stride_d == 0 && in_h % stride_h == 0 && in_w % stride_w == 0) {
                        in_d /= stride_d;
                        in_h /= stride_h;
                        in_w /= stride_w;
                        
                        if (in_d >= 0 && in_d < input_d && 
                            in_h >= 0 && in_h < input_h && 
                            in_w >= 0 && in_w < input_w) {
                            
                            int input_idx = batch_idx * (in_channels * input_d * input_h * input_w) +
                                          in_ch * (input_d * input_h * input_w) +
                                          in_d * (input_h * input_w) +
                                          in_h * input_w +
                                          in_w;
                                          
                            int weight_idx = out_ch * (in_channels * kernel_d * kernel_h * kernel_w) +
                                           in_ch * (kernel_d * kernel_h * kernel_w) +
                                           kd * (kernel_h * kernel_w) +
                                           kh * kernel_w +
                                           kw;
                                           
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add bias
    sum += bias[out_ch];
    
    // Add add_input
    int add_input_idx = batch_idx * (out_channels * output_d * output_h * output_w) +
                       out_ch * (output_d * output_h * output_w) +
                       out_d * (output_h * output_w) +
                       out_h * output_w +
                       out_w;
    sum += add_input[add_input_idx];
    
    // Apply HardSwish: x * relu6(x + 3) / 6
    float hardswish_val;
    if (sum <= -3.0f) {
        hardswish_val = 0.0f;
    } else if (sum >= 3.0f) {
        hardswish_val = sum;
    } else {
        hardswish_val = sum * (sum + 3.0f) / 6.0f;
    }
    
    // Write output
    int output_idx = batch_idx * (out_channels * output_d * output_h * output_w) +
                    out_ch * (output_d * output_h * output_w) +
                    out_d * (output_h * output_w) +
                    out_h * output_w +
                    out_w;
    output[output_idx] = hardswish_val;
}

torch::Tensor conv3d_transpose_add_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor add_input,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_d = input.size(2);
    auto input_h = input.size(3);
    auto input_w = input.size(4);
    
    auto out_channels = weight.size(0);
    auto kernel_d = weight.size(2);
    auto kernel_h = weight.size(3);
    auto kernel_w = weight.size(4);
    
    // Calculate output dimensions
    auto output_d = (input_d - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d;
    auto output_h = (input_h - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    auto output_w = (input_w - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;
    
    auto output = torch::zeros({batch_size, out_channels, output_d, output_h, output_w}, 
                              torch::dtype(input.dtype()).device(input.device()));
    
    // Launch kernel
    dim3 grid(batch_size, out_channels, output_d * output_h * output_w);
    dim3 block(1, 1, 1);
    
    conv3d_transpose_add_hardswish_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w
    );
    
    return output;
}
"""

conv3d_transpose_add_hardswish_cpp_source = """
torch::Tensor conv3d_transpose_add_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor add_input,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w
);
"""

# Compile the inline CUDA code
conv3d_transpose_add_hardswish = load_inline(
    name="conv3d_transpose_add_hardswish",
    cpp_sources=conv3d_transpose_add_hardswish_cpp_source,
    cuda_sources=conv3d_transpose_add_hardswish_source,
    functions=["conv3d_transpose_add_hardswish_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model with fused ConvTranspose3d + Add + HardSwish operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        
        # Handle different stride/padding/output_padding formats
        if isinstance(stride, int):
            self.stride_d = self.stride_h = self.stride_w = stride
        else:
            self.stride_d, self.stride_h, self.stride_w = stride
            
        if isinstance(padding, int):
            self.padding_d = self.padding_h = self.padding_w = padding
        else:
            self.padding_d, self.padding_h, self.padding_w = padding
            
        if isinstance(output_padding, int):
            self.output_padding_d = self.output_padding_h = self.output_padding_w = output_padding
        else:
            self.output_padding_d, self.output_padding_h, self.output_padding_w = output_padding

    def forward(self, x, add_input):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
            add_input (torch.Tensor): Input tensor to be added after transposed convolution, of shape (batch_size, out_channels, D_out, H_out, W_out).
        Returns:
            torch.Tensor: Output tensor after fused ConvTranspose3d + Add + HardSwish.
        """
        return conv3d_transpose_add_hardswish.conv3d_transpose_add_hardswish_cuda(
            x, self.weight, self.bias, add_input,
            self.stride_d, self.stride_h, self.stride_w,
            self.padding_d, self.padding_h, self.padding_w,
            self.output_padding_d, self.output_padding_h, self.output_padding_w
        )