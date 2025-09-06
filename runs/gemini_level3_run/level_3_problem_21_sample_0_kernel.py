import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel source for a fused DepthwiseConv2d -> BatchNorm2d -> ReLU6
fused_dw_conv_bn_relu6_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Helper macros for checking tensor properties
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA kernel for Fused DepthwiseConv2d + BatchNorm2d + ReLU6
__global__ void fused_dw_conv_bn_relu6_kernel(
    const float* input,
    const float* conv_weight,
    const float* bn_weight,
    const float* bn_bias,
    const float* bn_mean,
    const float* bn_var,
    float* output,
    const int N, const int C, const int H, const int W,
    const int H_out, const int W_out,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const float bn_eps) {

    // Calculate the global thread index for the output tensor element
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = N * C * H_out * W_out;

    if (index >= num_threads) {
        return;
    }

    // Decode the 1D index to 4D coordinates (n, c, h_out, w_out)
    const int w_out = index % W_out;
    const int h_out = (index / W_out) % H_out;
    const int c = (index / (W_out * H_out)) % C;
    const int n = index / (W_out * H_out * C);

    // Calculate the top-left corner of the receptive field in the input tensor
    const int h_in_start = h_out * stride_h - pad_h;
    const int w_in_start = w_out * stride_w - pad_w;

    float acc = 0.0f;

    // Perform depthwise convolution for the current output element
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            const int h_in = h_in_start + kh;
            const int w_in = w_in_start + kw;

            // Boundary check for padding
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                // Input index: (n, c, h_in, w_in)
                const int input_idx = n * C * H * W + c * H * W + h_in * W + w_in;
                // Weight index: (c, 0, kh, kw)
                const int weight_idx = c * kernel_h * kernel_w + kh * kernel_w + kw;
                acc += input[input_idx] * conv_weight[weight_idx];
            }
        }
    }

    // Apply fused BatchNorm
    // y = gamma * (x - mean) / sqrt(var + eps) + beta
    // y = (gamma / sqrt(var + eps)) * x + (beta - gamma * mean / sqrt(var + eps))
    const float scale = bn_weight[c] / sqrtf(bn_var[c] + bn_eps);
    const float bias = bn_bias[c] - bn_mean[c] * scale;
    acc = acc * scale + bias;

    // Apply ReLU6 activation
    acc = fminf(fmaxf(acc, 0.0f), 6.0f);

    // Write the final result to the output tensor
    output[index] = acc;
}

// C++ wrapper function that will be called from Python
torch::Tensor fused_dw_conv_bn_relu6_cuda(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    double eps) {

    // Input validation
    CHECK_INPUT(input);
    CHECK_INPUT(conv_weight);
    CHECK_INPUT(bn_weight);
    CHECK_INPUT(bn_bias);
    CHECK_INPUT(bn_mean);
    CHECK_INPUT(bn_var);

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    const int kernel_h = conv_weight.size(2);
    const int kernel_w = conv_weight.size(3);

    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int pad_h = padding[0];
    const int pad_w = padding[1];

    // Calculate output dimensions
    const int H_out = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    // Create the output tensor
    auto output = torch::zeros({N, C, H_out, W_out}, input.options());

    const int num_elements = N * C * H_out * W_out;
    if (num_elements == 0) {
        return output;
    }

    // Configure launch parameters
    const int block_size = 1024;
    const int grid_size = (num_elements + block_size - 1) / block_size;

    // Launch the CUDA kernel
    fused_dw_conv_bn_relu6_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        H_out, W_out,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        static_cast<float>(eps)
    );

    // Check for any CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

fused_dw_conv_bn_relu6_cpp_source = """
torch::Tensor fused_dw_conv_bn_relu6_cuda(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    double eps);
"""

# JIT compile the inline CUDA code. This happens once when the Python module is loaded.
fused_op_module = load_inline(
    name="fused_op_module",
    cpp_sources=fused_dw_conv_bn_relu6_cpp_source,
    cuda_sources=fused_dw_conv_bn_relu6_source,
    functions=["fused_dw_conv_bn_relu6_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        MBConv block implementation with a fused CUDA kernel for the depthwise stage.
        The depthwise convolution, batch normalization, and ReLU6 activation are
        replaced by a single, custom high-performance CUDA kernel.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Kernel size for the depthwise convolution.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the intermediate channels.
        """
        super(ModelNew, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        # Expansion phase remains a standard PyTorch module
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        
        # Depthwise convolution phase: We define the original modules to hold the parameters
        # (weights, biases, running stats), but we will use our custom fused kernel in the
        # forward pass instead of calling these modules directly.
        self.depthwise_conv_fused = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            # The ReLU6 is implicitly part of our fused kernel
        )
        
        # Projection phase remains a standard PyTorch module
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Store the compiled fused operator
        self.fused_dw_conv_op = fused_op_module
    
    def forward(self, x):
        """
        Forward pass of the MBConv block using the custom fused CUDA kernel.

        :param x: The input tensor, shape (batch_size, in_channels, H, W)
        :return: The output tensor, shape (batch_size, out_channels, H', W')
        """
        identity = x
        
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        
        # --- Custom Fused Kernel Call ---
        # Extract the layers and their parameters to pass to the kernel
        dw_conv = self.depthwise_conv_fused[0]
        dw_bn = self.depthwise_conv_fused[1]

        # Custom kernels often require contiguous tensors
        x = x.contiguous()

        # Call the C++/CUDA function from our JIT-compiled module
        x = self.fused_dw_conv_op.fused_dw_conv_bn_relu6_cuda(
            x,
            dw_conv.weight,
            dw_bn.weight,
            dw_bn.bias,
            dw_bn.running_mean,
            dw_bn.running_var,
            dw_conv.stride,
            dw_conv.padding,
            dw_bn.eps
        )
        # --- End of Custom Kernel Call ---
        
        x = self.project_conv(x)
        
        if self.use_residual:
            x += identity
        
        return x