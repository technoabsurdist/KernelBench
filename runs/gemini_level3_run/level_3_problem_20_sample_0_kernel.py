import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for Fused Conv2D + Bias + Activation
# This kernel is a naive direct convolution implementation for demonstration purposes.
# It supports standard and depthwise convolutions, and optional ReLU6 activation.
fused_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Enum to select activation function
enum class Activation {
    NONE,
    RELU6
};

// Device function to apply activation
template<Activation A>
__device__ inline float apply_activation(float x);

template<>
__device__ inline float apply_activation<Activation::NONE>(float x) {
    return x;
}

template<>
__device__ inline float apply_activation<Activation::RELU6>(float x) {
    return fminf(fmaxf(0.0f, x), 6.0f);
}

// The CUDA kernel for fused convolution
template<Activation A>
__global__ void fused_conv_activation_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int groups) {

    // Using a 1D grid-stride loop to cover all output elements
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N * C_out * H_out * W_out;
         idx += blockDim.x * gridDim.x) {

        // Map 1D index to 4D output tensor coordinates (n, k, y_out, x_out)
        const int x_out = idx % W_out;
        const int y_out = (idx / W_out) % H_out;
        const int k = (idx / (W_out * H_out)) % C_out;
        const int n = idx / (W_out * H_out * C_out);

        float acc = 0.0f;

        const int c_start = (k / (C_out / groups)) * (C_in / groups);
        const int c_end = c_start + (C_in / groups);

        for (int c = c_start; c < c_end; ++c) {
            for (int r = 0; r < K_h; ++r) {
                for (int s = 0; s < K_w; ++s) {
                    const int y_in = y_out * stride_h + r - pad_h;
                    const int x_in = x_out * stride_w + s - pad_w;

                    if (y_in >= 0 && y_in < H_in && x_in >= 0 && x_in < W_in) {
                        // Calculate linear indices for input and weight
                        int input_idx = n * C_in * H_in * W_in + c * H_in * W_in + y_in * W_in + x_in;
                        int weight_idx = k * (C_in / groups) * K_h * K_w + (c % (C_in/groups)) * K_h * K_w + r * K_w + s;
                        acc += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        if (bias) {
            acc += bias[k];
        }

        output[idx] = apply_activation<A>(acc);
    }
}

// C++ wrapper function to be called from Python
torch::Tensor fused_conv_activation_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int groups,
    std::string activation_name) {

    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "Weight must be a float32 tensor");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "Bias must be a float32 tensor");

    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    const int C_out = weight.size(0);
    const int K_h = weight.size(2);
    const int K_w = weight.size(3);

    const int H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
    const int W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (N * C_out * H_out * W_out + block_size - 1) / block_size;
    const int max_grid_size = 4096; // A reasonable limit
    const int grid_size = std::min(num_blocks, max_grid_size);

    if (activation_name == "relu6") {
        fused_conv_activation_kernel<Activation::RELU6><<<grid_size, block_size>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
            N, C_in, H_in, W_in, C_out, H_out, W_out, K_h, K_w,
            stride_h, stride_w, pad_h, pad_w, groups);
    } else { // "none"
        fused_conv_activation_kernel<Activation::NONE><<<grid_size, block_size>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
            N, C_in, H_in, W_in, C_out, H_out, W_out, K_h, K_w,
            stride_h, stride_w, pad_h, pad_w, groups);
    }
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

fused_conv_cpp_source = """
torch::Tensor fused_conv_activation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride_h, int stride_w, int pad_h, int pad_w, int groups,
    std::string activation_name);
"""

# JIT compile the custom CUDA kernel
fused_conv_op = load_inline(
    name="fused_conv_op",
    cpp_sources=fused_conv_cpp_source,
    cuda_sources=fused_conv_source,
    functions=["fused_conv_activation_cuda"],
    verbose=False,
)

class FusedConvBNActivation(nn.Module):
    """
    A module that fuses Conv2d, BatchNorm2d, and an optional activation (ReLU6).
    The BatchNorm parameters are folded into the Conv2d's weights and bias at initialization.
    This is an inference-only optimization.
    """
    def __init__(self, conv, bn, activation="relu6"):
        super().__init__()
        self.stride = conv.stride
        self.padding = conv.padding
        self.groups = conv.groups
        self.activation = activation

        conv_w = conv.weight
        conv_b = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels).to(conv.weight.device)
        
        bn_rm = bn.running_mean
        bn_rv = bn.running_var
        bn_w = bn.weight
        bn_b = bn.bias
        bn_eps = bn.eps

        # Fold BN parameters into conv parameters
        if bn_w is None: bn_w = torch.ones_like(bn_rm)
        if bn_b is None: bn_b = torch.zeros_like(bn_rm)

        bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

        fused_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
        fused_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

        self.fused_weight = nn.Parameter(fused_w, requires_grad=False)
        self.fused_bias = nn.Parameter(fused_b, requires_grad=False)

    def forward(self, x):
        return fused_conv_op.fused_conv_activation_cuda(
            x, self.fused_weight, self.fused_bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.groups, self.activation
        )

class FusedInvertedResidual(nn.Module):
    """
    An implementation of the Inverted Residual Block using fused operators.
    """
    def __init__(self, inp, oup, stride, expand_ratio):
        super(FusedInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
            bn1 = nn.BatchNorm2d(hidden_dim)
            layers.append(FusedConvBNActivation(conv1, bn1, activation="relu6"))
        
        # Depthwise
        conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
        bn2 = nn.BatchNorm2d(hidden_dim)
        layers.append(FusedConvBNActivation(conv2, bn2, activation="relu6"))
        
        # Pointwise projection
        conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        bn3 = nn.BatchNorm2d(oup)
        layers.append(FusedConvBNActivation(conv3, bn3, activation="none"))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Building first layer
        first_conv = nn.Conv2d(3, input_channel, 3, 2, 1, bias=False)
        first_bn = nn.BatchNorm2d(input_channel)
        features = [FusedConvBNActivation(first_conv, first_bn, activation="relu6")]

        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(FusedInvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # Building last several layers
        last_conv = nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False)
        last_bn = nn.BatchNorm2d(last_channel)
        features.append(FusedConvBNActivation(last_conv, last_bn, activation="relu6"))

        # Final layers (kept as standard PyTorch modules)
        features.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )

        # Weight initialization (important for the original unfused model, less so for this inference-only model)
        # We initialize a temporary standard model to extract folded weights from.
        temp_model = self._get_temp_model(num_classes)
        self.load_state_dict_from_unfused(temp_model.state_dict())

    def _get_temp_model(self, num_classes):
        # Helper to create a standard MobileNetV2 for weight initialization
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(num_classes=num_classes)
        return model

    def load_state_dict_from_unfused(self, state_dict):
        # This method is complex because the architecture is different.
        # For this example, we re-initialize the fused modules from scratch based on the unfused state_dict.
        # This is a simplified approach. A robust implementation would map keys.
        
        # Re-create modules with loaded weights
        def _make_divisible(v, divisor, min_value=None):
            if min_value is None: min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v: new_v += divisor
            return new_v

        # Create a standard model to load weights into
        temp_model = self._get_temp_model(self.classifier[1].out_features)
        temp_model.load_state_dict(state_dict)
        temp_model.eval() # Set to eval mode for BN folding

        # Extract layers from the temporary model and create fused modules
        features = []
        # First layer
        features.append(FusedConvBNActivation(temp_model.features[0][0], temp_model.features[0][1], "relu6"))
        
        # Inverted residual blocks
        for i in range(1, 18):
            block = temp_model.features[i]
            inp = block.conv[0][0].in_channels if len(block.conv) > 1 and hasattr(block.conv[0][0], 'in_channels') else block.conv[0].in_channels
            if hasattr(block, 'use_res_connect'): # InvertedResidual block
                stride = block.conv[1][0].stride[0] if len(block.conv) > 1 else block.conv[0].stride[0]
                oup = block.conv[-1][0].out_channels
                expand_ratio = block.conv[0][0].out_channels / inp if len(block.conv) > 1 else 1
                
                fused_block = FusedInvertedResidual(inp, oup, stride, expand_ratio)
                
                # Manually create fused layers from the temp model's layers
                fused_layers = []
                if expand_ratio != 1:
                    fused_layers.append(FusedConvBNActivation(block.conv[0][0], block.conv[0][1], "relu6"))
                    fused_layers.append(FusedConvBNActivation(block.conv[1][0], block.conv[1][1], "relu6"))
                    fused_layers.append(FusedConvBNActivation(block.conv[2], block.conv[3], "none"))
                else:
                    fused_layers.append(FusedConvBNActivation(block.conv[0], block.conv[1], "relu6"))
                    fused_layers.append(FusedConvBNActivation(block.conv[2], block.conv[3], "none"))
                
                fused_block.conv = nn.Sequential(*fused_layers)
                features.append(fused_block)

        # Last conv layer
        features.append(FusedConvBNActivation(temp_model.features[18][0], temp_model.features[18][1], "relu6"))
        features.append(nn.AdaptiveAvgPool2d((1, 1)))
        
        self.features = nn.Sequential(*features)
        self.classifier.load_state_dict(temp_model.classifier.state_dict())


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x