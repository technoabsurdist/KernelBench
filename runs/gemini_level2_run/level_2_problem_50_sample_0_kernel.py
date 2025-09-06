import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrapper for the fused operation
# This kernel fuses four operations: scale, avg_pool, bias_add, and a final scale.
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_scale_avg_pool_bias_scale_kernel(
    const float* input,
    float* output,
    const float* bias,
    const float scale1,
    const float scale2,
    const int N, const int C,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int K, const int S)
{
    // Calculate the global thread index
    const long long total_threads = (long long)N * C * D_out * H_out * W_out;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_threads) {
        return;
    }

    // Map the 1D global index to 5D output coordinates (n, c, d, h, w)
    const int w_out = idx % W_out;
    const int h_out = (idx / W_out) % H_out;
    const int d_out = (idx / (W_out * H_out)) % D_out;
    const int c = (idx / (W_out * H_out * D_out)) % C;
    const int n = idx / (W_out * H_out * D_out * C);

    // Determine the top-left corner of the pooling window in the input tensor
    const int d_start = d_out * S;
    const int h_start = h_out * S;
    const int w_start = w_out * S;

    float sum = 0.0f;
    int count = 0;

    // Iterate over the 3D pooling window
    for (int kd = 0; kd < K; ++kd) {
        const int d_in = d_start + kd;
        if (d_in >= D_in) continue; // Boundary check
        for (int kh = 0; kh < K; ++kh) {
            const int h_in = h_start + kh;
            if (h_in >= H_in) continue; // Boundary check
            for (int kw = 0; kw < K; ++kw) {
                const int w_in = w_start + kw;
                if (w_in >= W_in) continue; // Boundary check

                // Calculate the 1D index for the input tensor element
                const long long input_idx = (long long)n * C * D_in * H_in * W_in +
                                            (long long)c * D_in * H_in * W_in +
                                            (long long)d_in * H_in * W_in +
                                            (long long)h_in * W_in +
                                            w_in;
                
                // FUSION: Apply the first scaling operation while reading from global memory
                sum += input[input_idx] * scale1;
                count++;
            }
        }
    }

    // Finalize the operation for the current output element
    if (count > 0) {
        // Complete the average pooling
        float avg_val = sum / count;
        // FUSION: Add the bias (bias is shape [C, 1, 1, 1], so we just need the C-th element)
        float biased_val = avg_val + bias[c];
        // FUSION: Apply the second scaling operation
        float final_val = biased_val * scale2;
        output[idx] = final_val;
    } else {
        // This case should not be hit with valid pooling parameters but is included for safety
        output[idx] = 0.0f;
    }
}

// C++ wrapper function to be called from Python
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    torch::Tensor scale1,
    torch::Tensor scale2)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 5, "Input must be a 5D tensor");
    TORCH_CHECK(bias.size(0) == input.size(1), "Bias channel dimension mismatch with input");

    // Get input tensor dimensions
    const int N = input.size(0);
    const int C = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    // Hardcode AvgPool3d parameters from the original model
    const int kernel_size = 2;
    const int stride = 2;
    const int padding = 0; // Default for nn.AvgPool3d

    // Calculate output dimensions based on pooling parameters
    const int D_out = (D_in + 2 * padding - kernel_size) / stride + 1;
    const int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    const int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    // Create the output tensor
    auto output = torch::empty({N, C, D_out, H_out, W_out}, input.options());

    const long long total_elements = output.numel();
    if (total_elements == 0) {
        return output;
    }

    // Configure and launch the CUDA kernel
    const int block_size = 256;
    const long long num_blocks = (total_elements + block_size - 1) / block_size;

    fused_scale_avg_pool_bias_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale1.item<float>(), // Pass scalar values to the kernel
        scale2.item<float>(),
        N, C,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        kernel_size, stride
    );
    
    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

# Define the C++ function signature for the JIT compiler
fused_op_cpp_source = """
torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor scale1, torch::Tensor scale2);
"""

# Use load_inline to JIT compile the CUDA/C++ code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that replaces a sequence of four operations (scale, avg_pool, bias_add, scale)
    with a single custom fused CUDA kernel. The complex ConvTranspose3d operation is left to
    PyTorch's highly optimized cuDNN backend.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        # The ConvTranspose3d layer remains unchanged as it's complex and highly optimized.
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        
        # Parameters for the fused operation are kept as nn.Parameter to ensure they are trainable.
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

        # Store the compiled custom fused operator
        self.fused_op = fused_op.fused_op_cuda

    def forward(self, x):
        # 1. Apply the standard, highly-optimized ConvTranspose3d
        x = self.conv_transpose(x)
        
        # 2. Apply the custom fused kernel for the remaining four operations
        x = self.fused_op(x, self.bias, self.scale1, self.scale2)
        
        return x