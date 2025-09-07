import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_conv_transpose_mish_add_hardtanh_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

__device__ float mish_activation(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

__device__ float hardtanh_activation(float x, float min_val, float max_val) {
    return fminf(fmaxf(x, min_val), max_val);
}

__global__ void fused_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    float add_value,
    float scale
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_height * output_width;
    
    if (out_idx >= total_elements) return;
    
    int tmp = out_idx;
    int w_out = tmp % output_width;
    tmp /= output_width;
    int h_out = tmp % output_height;
    tmp /= output_height;
    int c_out = tmp % out_channels;
    int n = tmp / out_channels;
    
    float sum = (bias != nullptr) ? bias[c_out] : 0.0f;
    
    // Calculate corresponding input region
    int h_out_with_padding = h_out + padding;
    int w_out_with_padding = w_out + padding;
    
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h_out_with_padding - kh;
                int w_in = w_out_with_padding - kw;
                
                if (h_in % stride == 0 && w_in % stride == 0) {
                    h_in /= stride;
                    w_in /= stride;
                    
                    if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                        int input_idx = ((n * in_channels + c_in) * input_height + h_in) * input_width + w_in;
                        int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Apply Mish activation
    sum = mish_activation(sum);
    
    // Add value
    sum += add_value;
    
    // Apply Hardtanh activation
    sum = hardtanh_activation(sum, -1.0f, 1.0f);
    
    // Scale
    sum *= scale;
    
    output[out_idx] = sum;
}

torch::Tensor fused_conv_transpose_mish_add_hardtanh_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    float add_value,
    float scale
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_channels = weight.size(1);
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    int total_elements = batch_size * out_channels * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = min((total_elements + block_size - 1) / block_size, 65535);
    
    fused_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        add_value,
        scale
    );
    
    return output;
}
"""

fused_conv_transpose_mish_add_hardtanh_scale_cpp_source = """
torch::Tensor fused_conv_transpose_mish_add_hardtanh_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    float add_value,
    float scale
);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_conv_transpose_mish_add_hardtanh_scale",
    cpp_sources=fused_conv_transpose_mish_add_hardtanh_scale_cpp_source,
    cuda_sources=fused_conv_transpose_mish_add_hardtanh_scale_source,
    functions=["fused_conv_transpose_mish_add_hardtanh_scale_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for ConvTranspose2d + Mish + Add + Hardtanh + Scale
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.add_value = add_value
        self.scale = scale
        
        # Create the transposed convolution layer to get weights and bias
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        
        # Reference to the fused operation
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_conv_transpose_mish_add_hardtanh_scale_cuda(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.add_value,
            self.scale
        )

batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 128  
kernel_size  = 3
stride       = 2  
padding      = 1
output_padding = 1
add_value = 0.5
scale = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]