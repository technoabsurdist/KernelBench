import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ReLU6
relu6_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu6_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        data[idx] = fminf(fmaxf(val, 0.0f), 6.0f);
    }
}

torch::Tensor relu6_cuda(torch::Tensor input) {
    auto output = input.clone();
    int size = output.numel();
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    relu6_kernel<<<num_blocks, block_size>>>(output.data_ptr<float>(), size);
    
    return output;
}
"""

relu6_cpp_source = "torch::Tensor relu6_cuda(torch::Tensor input);"

relu6_op = load_inline(
    name="relu6_op",
    cpp_sources=relu6_cpp_source,
    cuda_sources=relu6_source,
    functions=["relu6_cuda"],
    verbose=False,
)

# Custom CUDA kernel for fused Conv1x1 + BatchNorm + ReLU6
fused_conv1x1_bn_relu6_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_bn_relu6_kernel(
    float* output, 
    const float* input,
    const float* gamma,
    const float* beta,
    const float* mean,
    const float* var,
    int batch_size,
    int channels,
    int spatial_size,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * spatial_size;
    
    if (idx < total_size) {
        int c = (idx / spatial_size) % channels;
        
        float val = input[idx];
        float std = sqrtf(var[c] + eps);
        val = (val - mean[c]) / std * gamma[c] + beta[c];
        val = fminf(fmaxf(val, 0.0f), 6.0f);
        output[idx] = val;
    }
}

torch::Tensor fused_conv1x1_bn_relu6_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
) {
    // Perform conv1x1 using cuBLAS
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    auto spatial_size = height * width;
    
    // Reshape for matmul
    auto input_reshaped = input.view({batch_size, in_channels, spatial_size});
    auto input_transposed = input_reshaped.transpose(1, 2).contiguous();
    auto input_2d = input_transposed.view({batch_size * spatial_size, in_channels});
    
    // Perform matmul
    auto output_2d = torch::matmul(input_2d, weight.t());
    auto output_transposed = output_2d.view({batch_size, spatial_size, out_channels});
    auto output_reshaped = output_transposed.transpose(1, 2).contiguous();
    auto output = output_reshaped.view({batch_size, out_channels, height, width});
    
    // Apply fused BN + ReLU6
    auto output_final = torch::empty_like(output);
    int total_size = batch_size * out_channels * spatial_size;
    
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    fused_bn_relu6_kernel<<<num_blocks, block_size>>>(
        output_final.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        batch_size,
        out_channels,
        spatial_size,
        eps
    );
    
    return output_final;
}
"""

fused_conv1x1_bn_relu6_cpp_source = """
torch::Tensor fused_conv1x1_bn_relu6_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
);
"""

fused_conv1x1_bn_relu6_op = load_inline(
    name="fused_conv1x1_bn_relu6_op",
    cpp_sources=fused_conv1x1_bn_relu6_cpp_source,
    cuda_sources=fused_conv1x1_bn_relu6_source,
    functions=["fused_conv1x1_bn_relu6_cuda"],
    verbose=False,
)

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

        class CustomReLU6(nn.Module):
            def forward(self, x):
                return relu6_op.relu6_cuda(x)

        class FusedConv1x1BNReLU6(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
                self.bn = nn.BatchNorm2d(out_channels)
                
            def forward(self, x):
                if self.training:
                    x = self.conv(x)
                    x = self.bn(x)
                    return relu6_op.relu6_cuda(x)
                else:
                    return fused_conv1x1_bn_relu6_op.fused_conv1x1_bn_relu6_cuda(
                        x,
                        self.conv.weight.view(self.conv.weight.size(0), -1),
                        self.bn.weight,
                        self.bn.bias,
                        self.bn.running_mean,
                        self.bn.running_var,
                        self.bn.eps
                    )

        def _inverted_residual_block(inp, oup, stride, expand_ratio):
            hidden_dim = int(inp * expand_ratio)
            use_res_connect = stride == 1 and inp == oup

            layers = []
            if expand_ratio != 1:
                layers.append(FusedConv1x1BNReLU6(inp, hidden_dim))

            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                CustomReLU6(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])

            if use_res_connect:
                return nn.Sequential(*layers), True
            else:
                return nn.Sequential(*layers), False

        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(input_channel),
                    CustomReLU6()]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(_inverted_residual_block(input_channel, output_channel, stride, expand_ratio=t)[0])
                input_channel = output_channel

        features.append(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False))
        features.append(nn.BatchNorm2d(last_channel))
        features.append(CustomReLU6())

        features.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]