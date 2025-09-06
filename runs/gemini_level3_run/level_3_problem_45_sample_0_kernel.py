import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------------
# Custom Fused CUDA Kernel for BatchNorm + Softmax
# ----------------------------------------------------------------------------

bn_softmax_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// Helper for reduction in shared memory using warp-level primitives for efficiency
// This is a standard block-wide reduction pattern.
__device__ void block_reduce_sum(float& val, float* sdata) {
    sdata[threadIdx.x] = val;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    val = sdata[0];
}

__device__ void block_reduce_max(float& val, float* sdata) {
    sdata[threadIdx.x] = val;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }
    val = sdata[0];
}

__global__ void bn_softmax_fused_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    int H, int W, int C) {

    // Each block processes one row from the flattened (N, C, H) dimensions
    const int row_idx = blockIdx.x;
    // From the flat row index, deduce the channel index 'c' to get the correct scale/bias
    const int c = (row_idx / H) % C;

    const float* row_in = x + row_idx * W;
    float* row_out = out + row_idx * W;
    const float c_scale = scale[c];
    const float c_bias = bias[c];

    extern __shared__ float sdata[];

    // --- Step 1: Find max value for the row (for stable softmax) ---
    // Each thread finds the max in its own subset of the row
    float thread_max = -FLT_MAX;
    for (int i = threadIdx.x; i < W; i += blockDim.x) {
        thread_max = fmaxf(thread_max, row_in[i] * c_scale + c_bias);
    }
    // Reduce across the block to find the absolute max for the row
    block_reduce_max(thread_max, sdata);
    const float block_max = thread_max;

    // --- Step 2: Calculate sum of exponentials ---
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < W; i += blockDim.x) {
        thread_sum += expf((row_in[i] * c_scale + c_bias) - block_max);
    }
    // Reduce across the block to get the total sum for the row
    block_reduce_sum(thread_sum, sdata);
    const float block_sum = thread_sum;

    // --- Step 3: Final calculation and write to output ---
    // Add a small epsilon to the denominator for numerical stability
    const float inv_sum = 1.0f / (block_sum + 1e-8f);
    for (int i = threadIdx.x; i < W; i += blockDim.x) {
        row_out[i] = expf((row_in[i] * c_scale + c_bias) - block_max) * inv_sum;
    }
}

// C++ wrapper function to be called from PyTorch
torch::Tensor bn_softmax_fused_cuda(
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor bias) {

    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "Input scale must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Input bias must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input x must be a 4D tensor");
    TORCH_CHECK(scale.dim() == 1, "Input scale must be a 1D tensor");
    TORCH_CHECK(bias.dim() == 1, "Input bias must be a 1D tensor");

    // Ensure input is contiguous in memory for correct pointer arithmetic
    auto x_cont = x.contiguous();

    const auto N = x_cont.size(0);
    const auto C = x_cont.size(1);
    const auto H = x_cont.size(2);
    const auto W = x_cont.size(3);

    TORCH_CHECK(scale.size(0) == C, "scale size must match channel dimension");
    TORCH_CHECK(bias.size(0) == C, "bias size must match channel dimension");

    auto out = torch::empty_like(x_cont);

    // Kernel launch configuration
    const int block_size = 256; // A common, reasonable block size
    const int num_rows = N * C * H;
    const dim3 grid_size(num_rows);
    const int shared_mem_size = block_size * sizeof(float);

    // Launch the kernel
    bn_softmax_fused_kernel<<<grid_size, block_size, shared_mem_size>>>(
        x_cont.data_ptr<float>(),
        out.data_ptr<float>(),
        scale.data_ptr<float>(),
        bias.data_ptr<float>(),
        H, W, C
    );

    // Check for any CUDA errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\\n", cudaGetErrorString(err));
    }

    return out;
}
"""

bn_softmax_fused_cpp_source = """
torch::Tensor bn_softmax_fused_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor bias);
"""

# JIT compile the CUDA kernel
bn_softmax_fused = load_inline(
    name="bn_softmax_fused",
    cpp_sources=bn_softmax_fused_cpp_source,
    cuda_sources=bn_softmax_fused_source,
    functions=["bn_softmax_fused_cuda"],
    verbose=False,
)

# ----------------------------------------------------------------------------
# Original Architecture (required for initialization of the new model)
# ----------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Softmax(dim=-1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.double_conv(x)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super(Model, self).__init__()
        self.encoder1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = DoubleConv(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(features * 8, features * 16)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(features * 2, features)
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.final_conv(dec1)

# ----------------------------------------------------------------------------
# New Architecture with Fused Operators
# ----------------------------------------------------------------------------

class FusedDoubleConv(nn.Module):
    """
    Replaces a DoubleConv block. It keeps the Conv2d layers but replaces
    the BatchNorm2d + Softmax sequence with a single fused CUDA kernel.
    This is optimized for inference, as it bakes the batchnorm running stats
    into a static scale and bias.
    """
    def __init__(self, orig_module: DoubleConv):
        super().__init__()
        self.fused_op = bn_softmax_fused

        # Extract conv layers directly from the original module
        self.conv1 = orig_module.double_conv[0]
        self.conv2 = orig_module.double_conv[3]

        # Extract batchnorm layers to get their parameters
        bn1 = orig_module.double_conv[1]
        bn2 = orig_module.double_conv[4]

        # This fusion is for inference, so we use the trained running stats
        bn1.eval()
        bn2.eval()

        # Pre-calculate scale and bias for the fused operation.
        # The BatchNorm operation y = (x - mean) / sqrt(var + eps) * weight + bias
        # can be rewritten as y = x * scale_factor + bias_factor, where:
        # scale_factor = weight / sqrt(var + eps)
        # bias_factor = bias - mean * scale_factor
        with torch.no_grad():
            scale1 = bn1.weight / torch.sqrt(bn1.running_var + bn1.eps)
            bias1 = bn1.bias - bn1.running_mean * scale1
            
            scale2 = bn2.weight / torch.sqrt(bn2.running_var + bn2.eps)
            bias2 = bn2.bias - bn2.running_mean * scale2

        # Register scale and bias as buffers. This ensures they are moved to the
        # correct device (e.g., .cuda()) along with the model.
        self.register_buffer('scale1', scale1)
        self.register_buffer('bias1', bias1)
        self.register_buffer('scale2', scale2)
        self.register_buffer('bias2', bias2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.fused_op.bn_softmax_fused_cuda(x, self.scale1, self.bias1)
        x = self.conv2(x)
        x = self.fused_op.bn_softmax_fused_cuda(x, self.scale2, self.bias2)
        return x


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super(ModelNew, self).__init__()
        
        # Create a temporary instance of the original model to steal its layers
        # and their initialized weights.
        orig_model = Model(in_channels, out_channels, features)

        # Replace all DoubleConv modules with our FusedDoubleConv version
        self.encoder1 = FusedDoubleConv(orig_model.encoder1)
        self.encoder2 = FusedDoubleConv(orig_model.encoder2)
        self.encoder3 = FusedDoubleConv(orig_model.encoder3)
        self.encoder4 = FusedDoubleConv(orig_model.encoder4)
        self.bottleneck = FusedDoubleConv(orig_model.bottleneck)
        self.decoder4 = FusedDoubleConv(orig_model.decoder4)
        self.decoder3 = FusedDoubleConv(orig_model.decoder3)
        self.decoder2 = FusedDoubleConv(orig_model.decoder2)
        self.decoder1 = FusedDoubleConv(orig_model.decoder1)

        # Copy over all other layers (pools, upconvs, etc.) from the original model
        self.pool1 = orig_model.pool1
        self.pool2 = orig_model.pool2
        self.pool3 = orig_model.pool3
        self.pool4 = orig_model.pool4
        self.upconv4 = orig_model.upconv4
        self.upconv3 = orig_model.upconv3
        self.upconv2 = orig_model.upconv2
        self.upconv1 = orig_model.upconv1
        self.final_conv = orig_model.final_conv

    def forward(self, x):
        """
        The forward pass logic is identical to the original model, but it now
        calls the FusedDoubleConv modules instead of the standard ones.
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)