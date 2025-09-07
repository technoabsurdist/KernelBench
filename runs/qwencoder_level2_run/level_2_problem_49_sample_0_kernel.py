import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv3d_transpose + softmax + sigmoid
conv3d_transpose_softmax_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA kernel for sigmoid
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// CUDA kernel for softmax (online softmax for numerical stability)
__device__ float safe_exp(float x, float max_val) {
    return expf(x - max_val);
}

// Fused kernel for conv3d_transpose + softmax + sigmoid
__global__ void conv3d_transpose_softmax_sigmoid_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    if (idx >= total_elements) return;
    
    // Decode output indices
    int temp = idx;
    int w_idx = temp % output_w;
    temp /= output_w;
    int h_idx = temp % output_h;
    temp /= output_h;
    int d_idx = temp % output_d;
    temp /= output_d;
    int c_out = temp % out_channels;
    int b_idx = temp / out_channels;
    
    // Calculate corresponding input region
    int kernel_radius = kernel_size / 2;
    float sum = 0.0f;
    
    // Conv3d transpose calculation
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    // Calculate input position
                    int in_d = d_idx - kd + kernel_radius;
                    int in_h = h_idx - kh + kernel_radius;
                    int in_w = w_idx - kw + kernel_radius;
                    
                    // Check bounds with stride and padding
                    if (in_d % stride == 0 && in_h % stride == 0 && in_w % stride == 0) {
                        in_d /= stride;
                        in_h /= stride;
                        in_w /= stride;
                        
                        if (in_d >= 0 && in_d < input_d && 
                            in_h >= 0 && in_h < input_h && 
                            in_w >= 0 && in_w < input_w) {
                            
                            // Get input value
                            int input_idx = b_idx * (in_channels * input_d * input_h * input_w) +
                                          c_in * (input_d * input_h * input_w) +
                                          in_d * (input_h * input_w) +
                                          in_h * input_w +
                                          in_w;
                            
                            // Get weight value
                            int weight_idx = c_in * (out_channels * kernel_size * kernel_size * kernel_size) +
                                           c_out * (kernel_size * kernel_size * kernel_size) +
                                           kd * (kernel_size * kernel_size) +
                                           kh * kernel_size +
                                           kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add bias
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    
    // For simplicity, we'll do a per-channel softmax and sigmoid in a separate step
    output[idx] = sum;
}

// Kernel for fused softmax + sigmoid per channel
__global__ void channel_softmax_sigmoid_kernel(
    float* data,
    int batch_size,
    int channels,
    int spatial_size
) {
    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || channel_idx >= channels) return;
    
    // Shared memory for reduction
    extern __shared__ float shared_data[];
    
    // Calculate channel data start
    int channel_start = batch_idx * (channels * spatial_size) + channel_idx * spatial_size;
    
    // Find max for numerical stability
    float thread_max = -1e30f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        float val = data[channel_start + i];
        thread_max = fmaxf(thread_max, val);
    }
    
    shared_data[tid] = thread_max;
    __syncthreads();
    
    // Reduce to find max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + s]);
        }
        __syncthreads();
    }
    
    float channel_max = shared_data[0];
    __syncthreads();
    
    // Calculate sum for softmax
    float thread_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        float val = data[channel_start + i];
        thread_sum += safe_exp(val, channel_max);
    }
    
    shared_data[tid] = thread_sum;
    __syncthreads();
    
    // Reduce to find sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    
    float channel_sum = shared_data[0];
    __syncthreads();
    
    // Apply softmax and sigmoid
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = channel_start + i;
        float val = data[idx];
        float softmax_val = safe_exp(val, channel_max) / channel_sum;
        data[idx] = sigmoid(softmax_val);
    }
}

torch::Tensor fused_conv3d_transpose_softmax_sigmoid(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding
) {
    // Get dimensions
    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_d = input_sizes[2];
    int input_h = input_sizes[3];
    int input_w = input_sizes[4];
    
    auto weight_sizes = weight.sizes();
    int out_channels = weight_sizes[1];
    int kernel_size = weight_sizes[2];
    
    // Calculate output dimensions
    int output_d = (input_d - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_h = (input_h - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_w = (input_w - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, output_d, output_h, output_w}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Launch conv3d_transpose kernel
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv3d_transpose_softmax_sigmoid_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_size,
        stride,
        padding,
        output_padding
    );
    
    // Launch softmax + sigmoid kernel
    dim3 block_dim(256);
    dim3 grid_dim(batch_size, out_channels);
    int shared_mem_size = 256 * sizeof(float);
    
    channel_softmax_sigmoid_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        output_d * output_h * output_w
    );
    
    return output;
}
"""

conv3d_transpose_softmax_sigmoid_cpp_source = """
torch::Tensor fused_conv3d_transpose_softmax_sigmoid(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding
);
"""

# Compile the inline CUDA code
fused_conv3d_transpose_softmax_sigmoid_op = load_inline(
    name="fused_conv3d_transpose_softmax_sigmoid",
    cpp_sources=conv3d_transpose_softmax_sigmoid_cpp_source,
    cuda_sources=conv3d_transpose_softmax_sigmoid_source,
    functions=["fused_conv3d_transpose_softmax_sigmoid"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model with fused Conv3dTranspose + Softmax + Sigmoid using custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        
        # Weight and bias parameters
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Reference to the fused CUDA operation
        self.fused_op = fused_conv3d_transpose_softmax_sigmoid_op

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        bias = self.bias if self.bias is not None else torch.zeros(self.out_channels, device=x.device)
        return self.fused_op.fused_conv3d_transpose_softmax_sigmoid(
            x, self.weight, bias, self.stride, self.padding, self.output_padding
        )

batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]