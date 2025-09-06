import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define and compile the custom CUDA kernel for Fused BatchNorm2d + ReLU
bn_relu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel for Fused BatchNorm2D + ReLU for inference
__global__ void batch_norm_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ y,
    float eps,
    int total_elements,
    int C,
    int spatial_dim) {

    // Grid-stride loop to process all elements
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_elements;
         i += blockDim.x * gridDim.x) {

        // Get channel index for the current element
        int c = (i / spatial_dim) % C;

        // Pre-calculate inverse standard deviation
        float inv_std = rsqrtf(running_var[c] + eps);

        // Apply BatchNorm: y = (x - mean) / std * weight + bias
        float normalized = (x[i] - running_mean[c]) * inv_std;
        float scaled = normalized * weight[c] + bias[c];

        // Apply ReLU: y = max(0, y)
        y[i] = fmaxf(0.0f, scaled);
    }
}

torch::Tensor batch_norm_relu_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {

    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(), "Weight tensor must be a CUDA tensor and contiguous");
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous(), "Bias tensor must be a CUDA tensor and contiguous");
    TORCH_CHECK(running_mean.is_cuda() && running_mean.is_contiguous(), "Running mean tensor must be a CUDA tensor and contiguous");
    TORCH_CHECK(running_var.is_cuda() && running_var.is_contiguous(), "Running var tensor must be a CUDA tensor and contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be of type float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D");

    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);
    const auto total_elements = x.numel();
    const auto spatial_dim = H * W;

    // Create output tensor
    auto y = torch::empty_like(x);

    // CUDA launch configuration
    const int block_size = 256;
    // Heuristic for grid size, cap to avoid too many blocks for small problems
    const int num_blocks = std::min((total_elements + block_size - 1) / block_size, 4096);

    // Launch kernel
    batch_norm_relu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<float>(eps),
        total_elements,
        C,
        spatial_dim
    );

    // Check for CUDA errors after kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}
"""

bn_relu_cpp_source = """
torch::Tensor batch_norm_relu_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps);
"""

# Compile the inline CUDA code
bn_relu_fused = load_inline(
    name="bn_relu_fused",
    cpp_sources=bn_relu_cpp_source,
    cuda_sources=bn_relu_cuda_source,
    functions=["batch_norm_relu_cuda"],
    verbose=False,
)


class FusedBatchNormReLU(nn.BatchNorm2d):
    """
    A fused BatchNorm2d and ReLU module.
    For inference on CUDA devices, it uses a custom CUDA kernel for speed.
    For training or CPU execution, it falls back to the standard PyTorch implementation
    to ensure correct gradient calculation and running statistics updates.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )

    def forward(self, x):
        if not self.training and self.track_running_stats and x.is_cuda:
            # Use custom CUDA kernel for inference
            return bn_relu_fused.batch_norm_relu_cuda(
                x, self.weight, self.bias, self.running_mean, self.running_var, self.eps
            )
        else:
            # Fallback to PyTorch's native implementation
            out = F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                self.training,
                self.momentum,
                self.eps,
            )
            return F.relu(out, inplace=True)


class DenseBlockNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlockNew, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        """
        Creates a single layer using the FusedBatchNormReLU operator.
        """
        return nn.Sequential(
            FusedBatchNormReLU(in_features),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            # The input to the layer is the concatenation of all previous features.
            # This is handled by the layer's `in_features` matching the concatenated size.
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)


class TransitionLayerNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayerNew, self).__init__()
        self.transition = nn.Sequential(
            FusedBatchNormReLU(num_input_features),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)


class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelNew, self).__init__()

        # Initial convolution and pooling with fused BN+ReLU
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            FusedBatchNormReLU(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Each dense block is followed by a transition layer, except the last one
        num_features = 64
        block_layers = [6, 12, 48, 32]  # Corresponding layers in DenseNet201

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlockNew(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayerNew(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        # Final batch norm and classifier, now fused
        self.final_bn_relu = FusedBatchNormReLU(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        # Fused final BN and ReLU
        x = self.final_bn_relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x