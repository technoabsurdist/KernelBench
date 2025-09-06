import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --------------------------------------------------------------------------------
# Custom Fused CUDA Kernel for BatchNorm2d + ReLU
# --------------------------------------------------------------------------------

fused_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel for Fused BatchNorm2d + ReLU
// This kernel performs the following operation:
// y = max(0, gamma * (x - E[x]) / sqrt(Var[x] + eps) + beta)
// It assumes NCHW tensor layout and inference mode (using running_mean and running_var).
__global__ void fused_bn_relu_kernel(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int total_elements) {

    // Using a grid-stride loop to handle any number of elements
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x) {
        // To get the channel index, we need tensor dimensions.
        // A simpler way without passing all dims is to compute it based on the fact
        // that the last two dimensions (H, W) are contiguous.
        // However, for BN, all elements in a channel get the same params.
        // We need to know C, H, W to correctly map idx to its channel.
        // Let's assume we get them.
        // int c = (idx / (H * W)) % C; // This would require passing H, W, C.
        // For simplicity and since this is a common pattern, we'll assume the C++ wrapper
        // provides the channel index or we compute it there.
        // Let's stick to the provided example's simplicity and assume the C++ part handles indexing if needed.
        // The provided example is element-wise, so let's make this one work similarly by passing all dims.
        // Let's get C, H, W from the input tensor shape in the C++ wrapper.
        // For now, let's assume we have a way to get the channel index `c`.
        // The C++ wrapper will pass the dimensions.
    }
}

// The C++ wrapper will handle the dimensions properly. Let's rewrite the kernel with full context.
__global__ void fused_bn_relu_kernel_v2(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int total_elements,
    int C,
    int H,
    int W) {

    int hw = H * W;
    int chw = C * H * W;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x) {
        // Decompose the 1D index `idx` to find the channel index `c`
        int c = (idx % chw) / hw;

        // Load BN parameters for the current channel
        float mean = running_mean[c];
        float var = running_var[c];
        float w = weight[c];
        float b = bias[c];

        // Perform BatchNorm
        float inv_std = 1.0f / sqrtf(var + eps);
        float normalized = (input[idx] - mean) * inv_std;
        float scaled = normalized * w + b;

        // Perform ReLU and store the result
        output[idx] = fmaxf(0.0f, scaled);
    }
}


torch::Tensor fused_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {

    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor (NCHW)");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias tensor must be contiguous");
    TORCH_CHECK(running_mean.is_contiguous(), "Running mean tensor must be contiguous");
    TORCH_CHECK(running_var.is_contiguous(), "Running var tensor must be contiguous");


    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);

    auto output = torch::empty_like(input);
    int total_elements = input.numel();

    if (total_elements == 0) {
        return output;
    }

    const int block_size = 256;
    // Heuristic for grid size
    const int num_blocks = std::min((total_elements + block_size - 1) / block_size, 4096);

    fused_bn_relu_kernel_v2<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        total_elements,
        C, H, W
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

fused_bn_relu_cpp_source = """
torch::Tensor fused_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps);
"""

# JIT compile the custom CUDA kernel
fused_bn_relu = load_inline(
    name="fused_bn_relu",
    cpp_sources=fused_bn_relu_cpp_source,
    cuda_sources=fused_bn_relu_source,
    functions=["fused_bn_relu_cuda"],
    verbose=False,
)

# --------------------------------------------------------------------------------
# Custom nn.Module using the fused kernel
# --------------------------------------------------------------------------------

class FusedBatchNormReLU(nn.Module):
    """
    A custom module that fuses BatchNorm2d and ReLU operations using a custom CUDA kernel.
    This module is designed for inference mode.
    """
    def __init__(self, num_features, eps=1e-5):
        super(FusedBatchNormReLU, self).__init__()
        self.num_features = num_features
        self.eps = eps
        
        # Replicate the parameters and buffers of nn.BatchNorm2d
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In a real scenario, training mode would require a separate kernel
        # to calculate batch statistics and update running_mean/running_var.
        # For this optimization problem, we focus on the inference path,
        # which is where custom kernels provide the most benefit.
        if not self.training:
            return fused_bn_relu.fused_bn_relu_cuda(
                x, self.weight, self.bias, self.running_mean, self.running_var, self.eps
            )
        else:
            # Fallback to standard PyTorch ops for training to ensure correctness
            # of batch statistics calculation and gradient computation.
            x = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                             self.training, 0.1, self.eps)
            return F.relu(x, inplace=True)

    def extra_repr(self):
        return '{num_features}, eps={eps}'.format(**self.__dict__)

# --------------------------------------------------------------------------------
# New Architecture with Fused Operators
# --------------------------------------------------------------------------------

class DenseBlockNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlockNew, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        # Replace nn.BatchNorm2d and nn.ReLU with our fused module
        return nn.Sequential(
            FusedBatchNormReLU(in_features),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            # The input to each layer is the concatenation of all previous feature maps
            concatenated_features = torch.cat(features, 1)
            new_feature = layer(concatenated_features)
            features.append(new_feature)
        # The final output of the block is the concatenation of the input and all new features
        return torch.cat(features, 1)


class TransitionLayerNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayerNew, self).__init__()
        # Replace nn.BatchNorm2d and nn.ReLU with our fused module
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
        block_layers = [6, 12, 24, 16]  # Corresponding layers in DenseNet121

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

        # Final batch norm and classifier, with fused BN+ReLU
        self.final_op = FusedBatchNormReLU(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        # In the original DenseNet, the input to a dense block is passed through all its layers.
        # The implementation in the prompt is slightly different, where the concatenated
        # features are passed to the next layer. We will keep that logic.
        # Let's correct the DenseBlockNew forward pass to match the original prompt's logic.
        
        # Re-implementing DenseBlockNew forward to match the prompt's logic exactly
        # The prompt's logic was:
        # features = [x]
        # for layer in self.layers:
        #     new_feature = layer(x) # x is the concatenated output of previous layers
        #     features.append(new_feature)
        #     x = torch.cat(features, 1)
        # return x
        # This logic is what my DenseBlockNew already does, but let's make it explicit.
        
        # The prompt's DenseBlock forward pass was slightly inefficient.
        # A standard DenseNet layer takes the concatenated features as input.
        # Let's re-verify the prompt's DenseBlock forward pass.
        # features = [x]
        # for layer in self.layers:
        #    new_feature = layer(x) # Here x is the *input* to the layer
        #    features.append(new_feature)
        #    x = torch.cat(features, 1) # Here x is updated for the *next* layer
        # This is not standard DenseNet, but we must replicate it.
        # My `DenseBlockNew` implementation is actually the standard one. Let's fix it to match the prompt.

        # Corrected DenseBlockNew to match the prompt's specific forward pass logic
        for i, block in enumerate(self.dense_blocks):
            # The prompt's forward pass for DenseBlock is a bit unusual.
            # Let's create a temporary forward method here to clarify and then fix the class.
            # The prompt's DenseBlock:
            # features = [x]
            # for layer in self.layers:
            #   new_feature = layer(x) # 'x' is the input to the layer
            #   features.append(new_feature)
            #   x = torch.cat(features, 1) # 'x' is updated for the next layer
            # This is a very memory-intensive pattern. My first implementation of DenseBlockNew
            # was actually the more standard, efficient one. Let's stick to the prompt's logic.
            # My DenseBlockNew needs to be corrected.
            
            # Let's define the correct DenseBlockNew again.
            pass # The one defined above is actually correct for the prompt's logic.
                 # `layer(concatenated_features)` is the standard and what the prompt does with `layer(x)`
                 # after `x = torch.cat(features, 1)`. My implementation is correct.

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        # The original does final_bn -> relu -> pool -> view -> linear
        # We fuse final_bn and relu
        x = self.final_op(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Re-defining DenseBlockNew to be absolutely certain it matches the prompt's logic.
# The prompt's logic:
# def forward(self, x_input):
#     features = [x_input]
#     x_current_layer_input = x_input
#     for layer in self.layers:
#         new_feature = layer(x_current_layer_input)
#         features.append(new_feature)
#         x_current_layer_input = torch.cat(features, 1)
#     return x_current_layer_input
# This is a subtle but important distinction. Let's rewrite DenseBlockNew.

class DenseBlockNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlockNew, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            FusedBatchNormReLU(in_features),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        # This forward pass exactly replicates the logic from the original `DenseBlock`
        features = [x]
        for layer in self.layers:
            # The input to the layer is the concatenation of all previous features
            layer_input = torch.cat(features, 1)
            new_feature = layer(layer_input)
            features.append(new_feature)
        return torch.cat(features, 1)

# The previous ModelNew definition was using the first version of DenseBlockNew.
# Let's redefine ModelNew to ensure it uses the final, correct version.
# (The logic was actually the same, but this makes it clearer).
# The final code is self-contained and correct.