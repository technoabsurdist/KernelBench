import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel source for ConvTranspose1d
conv_transpose1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose1d_forward_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int L_in,
    int C_out, int L_out,
    int K, int stride, int padding, int dilation
) {
    // Using a grid-stride loop to handle any number of output elements
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < N * C_out * L_out;
         index += blockDim.x * gridDim.x) {

        // De-flatten the 1D index to get (n, c_out, l_out)
        int l_out = index % L_out;
        int c_out = (index / L_out) % C_out;
        int n = index / (L_out * C_out);

        float sum = 0.0f;

        // Iterate over input channels and kernel positions
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int k = 0; k < K; ++k) {
                // This is the core relationship for transposed convolution's "gather" operation
                int l_in_numerator = l_out + padding - k * dilation;

                // An input element contributes to this output element only if it's perfectly divisible by stride
                if (l_in_numerator >= 0 && l_in_numerator % stride == 0) {
                    int l_in = l_in_numerator / stride;

                    // Check if the calculated input position is within bounds
                    if (l_in < L_in) {
                        // Calculate flat indices for input and weight tensors
                        long long input_idx = (long long)n * C_in * L_in + (long long)c_in * L_in + l_in;
                        // PyTorch weight layout for ConvTranspose1d is (C_in, C_out, K)
                        long long weight_idx = (long long)c_in * C_out * K + (long long)c_out * K + k;

                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        // Add bias if it exists (fused operation)
        if (bias != nullptr) {
            sum += bias[c_out];
        }

        output[index] = sum;
    }
}

torch::Tensor conv_transpose1d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias_opt,
    int stride,
    int padding,
    int dilation
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    input = input.contiguous();
    weight = weight.contiguous();

    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(input));

    // Get input dimensions
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int L_in = input.size(2);

    // Get weight dimensions
    TORCH_CHECK(weight.size(0) == C_in, "Weight in_channels mismatch");
    const int C_out = weight.size(1);
    const int K = weight.size(2);

    // Handle optional bias
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C_out, "Bias shape mismatch");
        bias = bias.contiguous();
        bias_ptr = bias.data_ptr<float>();
    }

    // Calculate output size using the standard formula
    const int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length must be positive");

    // Create the output tensor
    auto output = torch::empty({N, C_out, L_out}, input.options());

    // Get raw data pointers
    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Kernel launch configuration
    const long long total_threads = (long long)N * C_out * L_out;
    if (total_threads == 0) {
        return output; // Return empty tensor if no work to do
    }
    const int block_size = 256;
    // Cap grid size at CUDA's limit for a single dimension
    const int num_blocks = std::min((int)((total_threads + block_size - 1) / block_size), 65535);

    // Launch the CUDA kernel
    conv_transpose1d_forward_kernel<<<num_blocks, block_size>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        N, C_in, L_in, C_out, L_out,
        K, stride, padding, dilation
    );

    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

# C++ source for function signature
conv_transpose1d_cpp_source = """
torch::Tensor conv_transpose1d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias_opt,
    int stride,
    int padding,
    int dilation
);
"""

# JIT compile the custom CUDA kernel
custom_conv_transpose1d = load_inline(
    name="custom_conv_transpose1d",
    cpp_sources=conv_transpose1d_cpp_source,
    cuda_sources=conv_transpose1d_source,
    functions=["conv_transpose1d_forward_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution using a custom CUDA kernel.
    The implementation is a "gather" approach where each output element is computed
    independently, summing contributions from relevant input elements. This avoids
    atomic operations and is well-suited for parallelization. Bias addition is fused
    into the same kernel for efficiency.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We instantiate the original PyTorch layer to properly register and manage
        # the weight and bias parameters (e.g., for model.parameters(), .to(device), etc.).
        # We will not use its forward pass.
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        
        # Store hyperparameters to pass them to the custom kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Call the JIT-compiled CUDA function, passing the input tensor and the
        # managed weight and bias parameters from our placeholder layer.
        return custom_conv_transpose1d.conv_transpose1d_forward_cuda(
            x,
            self.conv1d_transpose.weight,
            self.conv1d_transpose.bias,
            self.stride,
            self.padding,
            self.dilation
        )