import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused transition layer:
# BatchNorm2d -> ReLU -> Conv2d(1x1) -> AvgPool2d(2x2)
fused_transition_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Helper for NCHW tensor indexing
#define TENSOR_IDX(n, c, h, w, C, H, W) (((n) * (C) + (c)) * (H) + (h)) * (W) + (w)
// Helper for 1x1 Conv weight indexing (C_out, C_in)
#define CONV_WEIGHT_IDX(c_out, c_in, C_in) ((c_out) * (C_in) + (c_in))

__global__ void fused_transition_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bn_scale,
    const float* __restrict__ bn_bias,
    const float* __restrict__ conv_weight,
    float* __restrict__ y,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int H_out, const int W_out
) {
    // Each thread computes one output pixel y[n, c_out, h_out, w_out]
    // Grid is launched to cover the output tensor dimensions (N, C_out, H_out, W_out)
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc_out_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (w_out >= W_out || h_out >= H_out || nc_out_idx >= N * C_out) {
        return;
    }

    const int n = nc_out_idx / C_out;
    const int c_out = nc_out_idx % C_out;

    float pool_sum = 0.0f;

    // 2x2 average pooling loop
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int h_in = 2 * h_out + i;
            const int w_in = 2 * w_out + j;

            float conv_sum = 0.0f;
            // 1x1 convolution loop (dot product over C_in)
            for (int c_in = 0; c_in < C_in; ++c_in) {
                // 1. Load input
                const float x_val = x[TENSOR_IDX(n, c_in, h_in, w_in, C_in, H_in, W_in)];

                // 2. Apply BatchNorm
                const float bn_val = x_val * bn_scale[c_in] + bn_bias[c_in];

                // 3. Apply ReLU
                const float relu_val = fmaxf(0.0f, bn_val);

                // 4. Apply 1x1 Conv
                const float conv_w = conv_weight[CONV_WEIGHT_IDX(c_out, c_in, C_in)];
                conv_sum += relu_val * conv_w;
            }
            pool_sum += conv_sum;
        }
    }

    // 5. Finalize AvgPool and write to output
    y[TENSOR_IDX(n, c_out, h_out, w_out, C_out, H_out, W_out)] = pool_sum * 0.25f;
}

torch::Tensor fused_transition_cuda(
    torch::Tensor x,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    double bn_eps,
    torch::Tensor conv_weight
) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(bn_weight.is_cuda(), "bn_weight must be a CUDA tensor");
    TORCH_CHECK(bn_bias.is_cuda(), "bn_bias must be a CUDA tensor");
    TORCH_CHECK(bn_running_mean.is_cuda(), "bn_running_mean must be a CUDA tensor");
    TORCH_CHECK(bn_running_var.is_cuda(), "bn_running_var must be a CUDA tensor");
    TORCH_CHECK(conv_weight.is_cuda(), "conv_weight must be a CUDA tensor");

    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    // Parameters are usually contiguous, but ensure for safety
    bn_weight = bn_weight.contiguous();
    bn_bias = bn_bias.contiguous();
    bn_running_mean = bn_running_mean.contiguous();
    bn_running_var = bn_running_var.contiguous();
    conv_weight = conv_weight.contiguous();

    const auto N = x.size(0);
    const auto C_in = x.size(1);
    const auto H_in = x.size(2);
    const auto W_in = x.size(3);

    const auto C_out = conv_weight.size(0);
    const auto H_out = H_in / 2;
    const auto W_out = W_in / 2;

    // Pre-compute batch norm parameters for inference
    // y = (x - mean) / sqrt(var + eps) * weight + bias
    // y = x * (weight / sqrt(var + eps)) + (bias - mean * weight / sqrt(var + eps))
    // y = x * bn_scale + bn_bias_term
    auto bn_scale = bn_weight / torch::sqrt(bn_running_var + bn_eps);
    auto bn_bias_term = bn_bias - bn_running_mean * bn_scale;

    // Create output tensor
    auto y = torch::empty({N, C_out, H_out, W_out}, x.options());

    // Configure launch parameters
    const dim3 threads(16, 16, 1);
    const dim3 blocks(
        (W_out + threads.x - 1) / threads.x,
        (H_out + threads.y - 1) / threads.y,
        (N * C_out + threads.z - 1) / threads.z
    );

    // Launch kernel
    fused_transition_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        bn_scale.data_ptr<float>(),
        bn_bias_term.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out
    );
    
    // Check for CUDA errors after kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}
"""

fused_transition_cpp_source = """
torch::Tensor fused_transition_cuda(
    torch::Tensor x,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    double bn_eps,
    torch::Tensor conv_weight
);
"""

# JIT compile the CUDA kernel
fused_transition = load_inline(
    name="fused_transition",
    cpp_sources=fused_transition_cpp_source,
    cuda_sources=fused_transition_source,
    functions=["fused_transition_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super(ModelNew, self).__init__()
        # We create the original layers to hold their parameters.
        # This makes it easy to load weights from a standard model.
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        
        # The ReLU and AvgPool operations are parameter-less and are fused into the kernel.

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Downsampled tensor with reduced number of feature maps
        """
        # This custom kernel is for inference only, as it uses the running statistics
        # from BatchNorm2d and does not implement a backward pass.
        if self.training:
            raise RuntimeError("ModelNew with fused CUDA kernel only supports evaluation mode.")

        # Ensure input is contiguous for the CUDA kernel
        x = x.contiguous()
        
        # The conv weight has shape (C_out, C_in, 1, 1). We squeeze it for the kernel.
        conv_weight_squeezed = self.conv.weight.squeeze()

        return fused_transition.fused_transition_cuda(
            x,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.eps,
            conv_weight_squeezed
        )