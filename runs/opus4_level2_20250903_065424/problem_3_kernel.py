import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused add + layer normalization
add_layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<typename T>
__global__ void add_layernorm_kernel(
    const T* __restrict__ input,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T* __restrict__ output,
    float add_val,
    int batch_size,
    int num_channels,
    int spatial_size,
    float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_channels * spatial_size;
    
    if (idx < total_elements) {
        int c = (idx / spatial_size) % num_channels;
        
        // Add scalar
        float val = static_cast<float>(input[idx]) + add_val;
        
        // Compute mean and variance for this channel across spatial dimensions
        float mean = 0.0f;
        float var = 0.0f;
        
        int spatial_start = (idx / (num_channels * spatial_size)) * num_channels * spatial_size + c * spatial_size;
        
        for (int i = 0; i < spatial_size; i++) {
            mean += static_cast<float>(input[spatial_start + i]) + add_val;
        }
        mean /= spatial_size;
        
        for (int i = 0; i < spatial_size; i++) {
            float diff = static_cast<float>(input[spatial_start + i]) + add_val - mean;
            var += diff * diff;
        }
        var /= spatial_size;
        
        // Apply layer norm
        float norm_val = (val - mean) / sqrtf(var + eps);
        output[idx] = static_cast<T>(norm_val * static_cast<float>(gamma[c]) + static_cast<float>(beta[c]));
    }
}

torch::Tensor add_layernorm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float add_val, float eps) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto spatial_size = depth * height * width;
    
    auto output = torch::empty_like(input);
    
    int total_elements = batch_size * num_channels * spatial_size;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "add_layernorm", ([&] {
        add_layernorm_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            add_val,
            batch_size,
            num_channels,
            spatial_size,
            eps
        );
    }));
    
    return output;
}
"""

add_layernorm_cpp_source = "torch::Tensor add_layernorm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float add_val, float eps);"

# Custom CUDA kernel for fused average pooling + GELU
avgpool_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math_constants.h>

template<typename T>
__device__ __forceinline__ T gelu_func(T x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    float x_float = static_cast<float>(x);
    float cdf = 0.5f * (1.0f + tanhf(sqrt_2_over_pi * (x_float + 0.044715f * x_float * x_float * x_float)));
    return static_cast<T>(x_float * cdf);
}

template<typename T>
__global__ void avgpool3d_gelu_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int batch_size, int channels,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_size * channels * D_out * H_out * W_out;
    
    if (idx < total_output) {
        int w_out = idx % W_out;
        int h_out = (idx / W_out) % H_out;
        int d_out = (idx / (W_out * H_out)) % D_out;
        int c = (idx / (W_out * H_out * D_out)) % channels;
        int b = idx / (channels * D_out * H_out * W_out);
        
        float sum = 0.0f;
        int count = 0;
        
        for (int kd = 0; kd < kernel_d; kd++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int d_in = d_out * stride_d - pad_d + kd;
                    int h_in = h_out * stride_h - pad_h + kh;
                    int w_in = w_out * stride_w - pad_w + kw;
                    
                    if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int input_idx = b * (channels * D_in * H_in * W_in) +
                                       c * (D_in * H_in * W_in) +
                                       d_in * (H_in * W_in) +
                                       h_in * W_in +
                                       w_in;
                        sum += static_cast<float>(input[input_idx]);
                        count++;
                    }
                }
            }
        }
        
        float avg_val = (count > 0) ? (sum / count) : 0.0f;
        output[idx] = gelu_func(static_cast<T>(avg_val));
    }
}

torch::Tensor avgpool3d_gelu_cuda(torch::Tensor input, int kernel_d, int kernel_h, int kernel_w) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto D_in = input.size(2);
    auto H_in = input.size(3);
    auto W_in = input.size(4);
    
    int stride_d = kernel_d, stride_h = kernel_h, stride_w = kernel_w;
    int pad_d = 0, pad_h = 0, pad_w = 0;
    
    int D_out = (D_in + 2 * pad_d - kernel_d) / stride_d + 1;
    int H_out = (H_in + 2 * pad_h - kernel_h) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - kernel_w) / stride_w + 1;
    
    auto output = torch::empty({batch_size, channels, D_out, H_out, W_out}, input.options());
    
    int total_output = batch_size * channels * D_out * H_out * W_out;
    const int threads = 256;
    const int blocks = (total_output + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "avgpool3d_gelu", ([&] {
        avgpool3d_gelu_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, channels,
            D_in, H_in, W_in,
            D_out, H_out, W_out,
            kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w
        );
    }));
    
    return output;
}
"""

avgpool_gelu_cpp_source = "torch::Tensor avgpool3d_gelu_cuda(torch::Tensor input, int kernel_d, int kernel_h, int kernel_w);"

# Compile the inline CUDA code
add_layernorm = load_inline(
    name="add_layernorm",
    cpp_sources=add_layernorm_cpp_source,
    cuda_sources=add_layernorm_source,
    functions=["add_layernorm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

avgpool_gelu = load_inline(
    name="avgpool_gelu",
    cpp_sources=avgpool_gelu_cpp_source,
    cuda_sources=avgpool_gelu_source,
    functions=["avgpool3d_gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernels for fused operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        
        # LayerNorm parameters
        self.norm_gamma = nn.Parameter(torch.ones(norm_shape))
        self.norm_beta = nn.Parameter(torch.zeros(norm_shape))
        self.norm_eps = 1e-5
        
        self.pool_kernel_size = pool_kernel_size
        
        self.add_layernorm = add_layernorm
        self.avgpool_gelu = avgpool_gelu

    def forward(self, x):
        x = self.conv_transpose(x)
        
        # Fused add + layer normalization
        x = self.add_layernorm.add_layernorm_cuda(
            x, self.norm_gamma, self.norm_beta, 
            self.sum_weight.item(), self.norm_eps
        )
        
        # Fused average pooling + GELU
        x = self.avgpool_gelu.avgpool3d_gelu_cuda(
            x, self.pool_kernel_size[0], self.pool_kernel_size[1], self.pool_kernel_size[2]
        )
        
        return x

batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = (1, 1, 1)
output_padding = (1, 1, 1)
sum_weight = 1.0
norm_shape = (out_channels,)
pool_kernel_size = (2, 2, 2)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size]