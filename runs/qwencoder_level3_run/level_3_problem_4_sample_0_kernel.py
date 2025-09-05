import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels
lenet_custom_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Conv + ReLU fusion kernel
__global__ void conv_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int out_height,
    int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx < total_elements) {
        int b = idx / (out_channels * out_height * out_width);
        int c = (idx / (out_height * out_width)) % out_channels;
        int h = (idx / out_width) % out_height;
        int w = idx % out_width;
        
        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int ih = h + kh;
                    int iw = w + kw;
                    if (ih < in_height && iw < in_width) {
                        int input_idx = b * (in_channels * in_height * in_width) + 
                                       ic * (in_height * in_width) + 
                                       ih * in_width + iw;
                        int weight_idx = c * (in_channels * kernel_size * kernel_size) + 
                                        ic * (kernel_size * kernel_size) + 
                                        kh * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        sum += bias[c];
        output[idx] = fmaxf(0.0f, sum);  // ReLU activation
    }
}

// Max pooling kernel
__global__ void max_pool2d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int pool_size,
    int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_height = (in_height - pool_size) / stride + 1;
    int out_width = (in_width - pool_size) / stride + 1;
    int total_elements = batch_size * channels * out_height * out_width;
    
    if (idx < total_elements) {
        int b = idx / (channels * out_height * out_width);
        int c = (idx / (out_height * out_width)) % channels;
        int h = (idx / out_width) % out_height;
        int w = idx % out_width;
        
        float max_val = -1e38f;
        for (int ph = 0; ph < pool_size; ph++) {
            for (int pw = 0; pw < pool_size; pw++) {
                int ih = h * stride + ph;
                int iw = w * stride + pw;
                if (ih < in_height && iw < in_width) {
                    int input_idx = b * (channels * in_height * in_width) + 
                                   c * (in_height * in_width) + 
                                   ih * in_width + iw;
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }
        output[idx] = max_val;
    }
}

// Linear + ReLU fusion kernel
__global__ void linear_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx < total_elements) {
        int b = idx / out_features;
        int out_idx = idx % out_features;
        
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            int input_idx = b * in_features + i;
            int weight_idx = out_idx * in_features + i;
            sum += input[input_idx] * weight[weight_idx];
        }
        sum += bias[out_idx];
        output[idx] = fmaxf(0.0f, sum);  // ReLU activation
    }
}

// Final linear layer kernel
__global__ void linear_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx < total_elements) {
        int b = idx / out_features;
        int out_idx = idx % out_features;
        
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            int input_idx = b * in_features + i;
            int weight_idx = out_idx * in_features + i;
            sum += input[input_idx] * weight[weight_idx];
        }
        sum += bias[out_idx];
        output[idx] = sum;
    }
}

// Conv + ReLU functions
torch::Tensor conv1_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int out_height = in_height - kernel_size + 1;
    int out_width = in_width - kernel_size + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_size,
        out_height,
        out_width
    );
    
    return output;
}

torch::Tensor conv2_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int out_height = in_height - kernel_size + 1;
    int out_width = in_width - kernel_size + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_size,
        out_height,
        out_width
    );
    
    return output;
}

// Max pooling functions
torch::Tensor max_pool2d_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int pool_size = 2;
    int stride = 2;
    int out_height = (in_height - pool_size) / stride + 1;
    int out_width = (in_width - pool_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, channels, out_height, out_width}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    const int total_elements = batch_size * channels * out_height * out_width;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    max_pool2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_height,
        in_width,
        pool_size,
        stride
    );
    
    return output;
}

// Linear + ReLU functions
torch::Tensor fc1_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    const int total_elements = batch_size * out_features;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    linear_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}

torch::Tensor fc2_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    const int total_elements = batch_size * out_features;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    linear_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}

// Final linear function
torch::Tensor fc3_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    const int total_elements = batch_size * out_features;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    linear_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

lenet_custom_cpp_source = """
torch::Tensor conv1_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor conv2_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor max_pool2d_cuda(torch::Tensor input);
torch::Tensor fc1_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor fc2_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor fc3_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
lenet_custom = load_inline(
    name="lenet_custom",
    cpp_sources=lenet_custom_cpp_source,
    cuda_sources=lenet_custom_source,
    functions=["conv1_relu_cuda", "conv2_relu_cuda", "max_pool2d_cuda", 
               "fc1_relu_cuda", "fc2_relu_cuda", "fc3_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, num_classes):
        """
        Optimized LeNet-5 with custom CUDA kernels.
        """
        super(ModelNew, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
        
        # Custom CUDA functions
        self.custom_ops = lenet_custom
    
    def forward(self, x):
        """
        Forward pass with custom CUDA kernels.
        """
        # First convolutional layer with ReLU activation and max pooling
        x = self.custom_ops.conv1_relu_cuda(x, self.conv1.weight, self.conv1.bias)
        x = self.custom_ops.max_pool2d_cuda(x)
        
        # Second convolutional layer with ReLU activation and max pooling
        x = self.custom_ops.conv2_relu_cuda(x, self.conv2.weight, self.conv2.bias)
        x = self.custom_ops.max_pool2d_cuda(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 16*5*5)
        
        # First fully connected layer with ReLU activation
        x = self.custom_ops.fc1_relu_cuda(x, self.fc1.weight, self.fc1.bias)
        
        # Second fully connected layer with ReLU activation
        x = self.custom_ops.fc2_relu_cuda(x, self.fc2.weight, self.fc2.bias)
        
        # Final fully connected layer
        x = self.custom_ops.fc3_cuda(x, self.fc3.weight, self.fc3.bias)
        
        return x

# Test code for the LeNet-5 model (larger batch & image)
batch_size = 4096
num_classes = 20

def get_inputs():
    return [torch.rand(batch_size, 1, 32, 32).cuda()]

def get_init_inputs():
    return [num_classes]