import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ConvTranspose2d + GELU + GroupNorm
fused_convtranspose_gelu_gn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void gelu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        data[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

__global__ void group_norm_kernel(float* data, const float* means, const float* vars, 
                                  const float* weights, const float* biases, 
                                  int batch, int channels, int height, int width, int group_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * channels * height * width;
    
    if (idx < total_elements) {
        int c = (idx / (height * width)) % channels;
        int g = c / group_size;
        
        float x = data[idx];
        float mean = means[g];
        float var = vars[g];
        float weight = weights[c];
        float bias = biases[c];
        
        float normalized = (x - mean) / sqrtf(var + 1e-5f);
        data[idx] = normalized * weight + bias;
    }
}

torch::Tensor fused_convtranspose_gelu_gn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int stride,
    int padding,
    int output_padding) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    
    // ConvTranspose2d
    torch::Tensor output = torch::conv_transpose2d(input, weight, bias, 
                                                   {stride, stride}, 
                                                   {padding, padding}, 
                                                   {output_padding, output_padding});
    
    // GELU activation
    int total_elements = output.numel();
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    gelu_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), total_elements);
    
    // GroupNorm
    auto batch = output.size(0);
    auto channels = output.size(1);
    auto height = output.size(2);
    auto width = output.size(3);
    auto num_groups = gamma.size(0);
    auto group_size = channels / num_groups;
    
    // Calculate means and vars per group
    auto reshaped = output.view({batch, num_groups, group_size, height, width});
    auto means = reshaped.mean({2, 3, 4});
    auto vars = reshaped.var({2, 3, 4}, false);
    
    group_norm_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(),
        means.data_ptr<float>(),
        vars.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        batch, channels, height, width, group_size);
    
    return output;
}
"""

fused_convtranspose_gelu_gn_cpp_source = """
torch::Tensor fused_convtranspose_gelu_gn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int stride,
    int padding,
    int output_padding);
"""

# Compile the inline CUDA code
fused_convtranspose_gelu_gn = load_inline(
    name="fused_convtranspose_gelu_gn",
    cpp_sources=fused_convtranspose_gelu_gn_cpp_source,
    cuda_sources=fused_convtranspose_gelu_gn_source,
    functions=["fused_convtranspose_gelu_gn_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused ConvTranspose2d + GELU + GroupNorm operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.num_groups = num_groups
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize weights using Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        
        self.fused_op = fused_convtranspose_gelu_gn

    def forward(self, x):
        return self.fused_op.fused_convtranspose_gelu_gn_cuda(
            x, self.weight, self.bias, self.gamma, self.beta,
            self.stride, 0, 0)  # padding=0, output_padding=0

batch_size   = 128  
in_channels  = 64  
out_channels = 64  
height = width = 256  
kernel_size  = 3
stride       = 1
groups = 8
num_groups = 8

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]