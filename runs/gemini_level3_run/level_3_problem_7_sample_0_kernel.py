import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom C++/CUDA source for a fused Inception module forward pass.
# This kernel fuses the four parallel branches of the Inception module and the final
# concatenation operation. It leverages PyTorch's underlying highly-optimized cuDNN
# kernels for convolution and pooling but avoids the memory overhead of allocating
# intermediate tensors for each branch's output and the subsequent `torch.cat`.
# It achieves this by pre-allocating the final output tensor and having each branch's
# operations write directly into their designated slice of that tensor.
inception_fused_source = """
#include <torch/extension.h>
#include <vector>

torch::Tensor inception_module_forward_cuda(
    const torch::Tensor& x,
    // Branch 1x1 weights and biases
    const torch::Tensor& conv1_w, const torch::Tensor& conv1_b,
    // Branch 3x3 weights and biases
    const torch::Tensor& conv3_reduce_w, const torch::Tensor& conv3_reduce_b,
    const torch::Tensor& conv3_w, const torch::Tensor& conv3_b,
    // Branch 5x5 weights and biases
    const torch::Tensor& conv5_reduce_w, const torch::Tensor& conv5_reduce_b,
    const torch::Tensor& conv5_w, const torch::Tensor& conv5_b,
    // Branch Pool projection weights and biases
    const torch::Tensor& pool_proj_w, const torch::Tensor& pool_proj_b
) {
    // Ensure all inputs are on the same CUDA device
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

    // Get input dimensions
    const auto batch_size = x.size(0);
    const auto in_height = x.size(2);
    const auto in_width = x.size(3);

    // Determine output channel counts from the weight tensors
    const auto out_1x1 = conv1_w.size(0);
    const auto out_3x3 = conv3_w.size(0);
    const auto out_5x5 = conv5_w.size(0);
    const auto pool_proj = pool_proj_w.size(0);
    const auto total_out_channels = out_1x1 + out_3x3 + out_5x5 + pool_proj;

    // Pre-allocate the final output tensor to avoid intermediate allocations and concatenation
    auto output = torch::empty({batch_size, total_out_channels, in_height, in_width}, x.options());

    // --- Branch 1x1 ---
    // Get a view (slice) of the output tensor for this branch
    auto branch1_out = output.slice(/*dim=*/1, /*start=*/0, /*end=*/out_1x1);
    // Perform 1x1 convolution directly into the output slice
    at::conv2d_out(branch1_out, x, conv1_w, conv1_b, {1, 1}, {0, 0}, {1, 1}, 1);

    // --- Branch 3x3 ---
    // Perform the 1x1 reduction convolution (creates an intermediate tensor)
    auto branch3_reduce_out = at::conv2d(x, conv3_reduce_w, conv3_reduce_b, {1, 1}, {0, 0}, {1, 1}, 1);
    // Get a view for the 3x3 branch output
    auto branch3_out = output.slice(/*dim=*/1, /*start=*/out_1x1, /*end=*/out_1x1 + out_3x3);
    // Perform the 3x3 convolution, writing the result into the output slice
    at::conv2d_out(branch3_out, branch3_reduce_out, conv3_w, conv3_b, {1, 1}, {1, 1}, {1, 1}, 1);

    // --- Branch 5x5 ---
    // Perform the 1x1 reduction convolution
    auto branch5_reduce_out = at::conv2d(x, conv5_reduce_w, conv5_reduce_b, {1, 1}, {0, 0}, {1, 1}, 1);
    // Get a view for the 5x5 branch output
    auto branch5_out = output.slice(/*dim=*/1, /*start=*/out_1x1 + out_3x3, /*end=*/out_1x1 + out_3x3 + out_5x5);
    // Perform the 5x5 convolution, writing the result into the output slice
    at::conv2d_out(branch5_out, branch5_reduce_out, conv5_w, conv5_b, {1, 1}, {2, 2}, {1, 1}, 1);

    // --- Branch Pool ---
    // Perform max pooling (creates an intermediate tensor)
    auto pool_out = at::max_pool2d(x, {3, 3}, {1, 1}, {1, 1}, {1, 1}, false);
    // Get a view for the pool projection branch output
    auto branch_pool_out = output.slice(/*dim=*/1, /*start=*/out_1x1 + out_3x3 + out_5x5, /*end=*/total_out_channels);
    // Perform the 1x1 projection convolution, writing the result into the output slice
    at::conv2d_out(branch_pool_out, pool_out, pool_proj_w, pool_proj_b, {1, 1}, {0, 0}, {1, 1}, 1);

    return output;
}
"""

inception_fused_cpp_source = """
torch::Tensor inception_module_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& conv1_w, const torch::Tensor& conv1_b,
    const torch::Tensor& conv3_reduce_w, const torch::Tensor& conv3_reduce_b,
    const torch::Tensor& conv3_w, const torch::Tensor& conv3_b,
    const torch::Tensor& conv5_reduce_w, const torch::Tensor& conv5_reduce_b,
    const torch::Tensor& conv5_w, const torch::Tensor& conv5_b,
    const torch::Tensor& pool_proj_w, const torch::Tensor& pool_proj_b
);
"""

# Compile the inline C++/CUDA code
inception_fused = load_inline(
    name="inception_fused",
    cpp_sources=inception_fused_cpp_source,
    cuda_sources=inception_fused_source,
    functions=["inception_module_forward_cuda"],
    verbose=True,
)


class InceptionModuleNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(InceptionModuleNew, self).__init__()
        
        # Define layers to hold parameters (weights/biases)
        # These layers will not be called directly in the forward pass, but their
        # parameters will be passed to the custom CUDA kernel.
        
        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        # 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        # Max pooling branch
        # The MaxPool2d layer itself has no parameters, so we only need the Conv2d
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
    
    def forward(self, x):
        # Call the fused C++/CUDA kernel, passing the input tensor and all required
        # weights and biases from the defined layers.
        return inception_fused.inception_module_forward_cuda(
            x,
            # Branch 1x1 params
            self.branch1x1.weight, self.branch1x1.bias,
            # Branch 3x3 params
            self.branch3x3[0].weight, self.branch3x3[0].bias,
            self.branch3x3[1].weight, self.branch3x3[1].bias,
            # Branch 5x5 params
            self.branch5x5[0].weight, self.branch5x5[0].bias,
            self.branch5x5[1].weight, self.branch5x5[1].bias,
            # Branch Pool params
            self.branch_pool[1].weight, self.branch_pool[1].bias
        )

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Use the new InceptionModuleNew with the fused CUDA kernel
        self.inception3a = InceptionModuleNew(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModuleNew(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception4a = InceptionModuleNew(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModuleNew(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModuleNew(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModuleNew(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModuleNew(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception5a = InceptionModuleNew(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModuleNew(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.maxpool2(F.relu(self.conv3(x)))
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x