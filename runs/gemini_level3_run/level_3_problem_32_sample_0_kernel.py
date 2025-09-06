import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA and C++ source code for the fused operators
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

// A simple warp-level reduction sum using shuffle instructions
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// A simple block-level reduction sum using shared memory
__device__ inline float block_reduce_sum(float val, float* shared) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    val = (tid < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}


__global__ void add_layer_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float epsilon,
    int D,
    float* __restrict__ y) {

    extern __shared__ float shared_mem[];

    int row_idx = blockIdx.x;
    int base_idx = row_idx * D;

    // Step 1: Fused Add and parallel reduction for mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sum += x[base_idx + i] + residual[base_idx + i];
    }
    sum = block_reduce_sum(sum, shared_mem);

    if (threadIdx.x == 0) {
        shared_mem[0] = sum / D;
    }
    __syncthreads();
    float mean = shared_mem[0];

    // Step 2: Parallel reduction for variance
    float sum_sq_diff = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = x[base_idx + i] + residual[base_idx + i];
        sum_sq_diff += (val - mean) * (val - mean);
    }
    sum_sq_diff = block_reduce_sum(sum_sq_diff, shared_mem);

    if (threadIdx.x == 0) {
        shared_mem[0] = rsqrtf(sum_sq_diff / D + epsilon);
    }
    __syncthreads();
    float inv_std = shared_mem[0];

    // Step 3: Apply normalization, scale, and shift
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = x[base_idx + i] + residual[base_idx + i];
        y[base_idx + i] = (val - mean) * inv_std * gamma[i] + beta[i];
    }
}

__global__ void bias_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    int N, // Inner dimension
    int total_elements,
    float* __restrict__ output) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int col_idx = idx % N;
        output[idx] = fmaxf(0.0f, input[idx] + bias[col_idx]);
    }
}

// C++ Wrapper for Add + LayerNorm
torch::Tensor add_layer_norm_cuda(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor gamma,
    torch::Tensor beta,
    double epsilon) {

    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(residual.is_cuda(), "Input residual must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "Input gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "Input beta must be a CUDA tensor");

    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(residual.is_contiguous(), "Input residual must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "Input gamma must be contiguous");
    TORCH_CHECK(beta.is_contiguous(), "Input beta must be contiguous");
    
    TORCH_CHECK(x.sizes() == residual.sizes(), "x and residual must have the same shape");
    
    const auto last_dim = x.dim() - 1;
    const int D = x.size(last_dim);
    const int B_S = x.numel() / D;

    TORCH_CHECK(gamma.numel() == D, "gamma must have size D");
    TORCH_CHECK(beta.numel() == D, "beta must have size D");

    auto y = torch::empty_like(x);

    const int block_size = 256;
    // One block per row (token embedding)
    const int num_blocks = B_S;
    // Shared memory for reduction (number of warps * sizeof(float))
    const int shared_mem_size = (block_size / 32) * sizeof(float);

    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    add_layer_norm_kernel<<<num_blocks, block_size, shared_mem_size, stream.stream()>>>(
        x.data_ptr<float>(),
        residual.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        static_cast<float>(epsilon),
        D,
        y.data_ptr<float>()
    );
    return y;
}

// C++ Wrapper for Bias + ReLU
torch::Tensor bias_relu_cuda(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");

    const auto last_dim = input.dim() - 1;
    const int N = input.size(last_dim);
    TORCH_CHECK(bias.numel() == N, "Bias size must match input's last dimension");

    auto output = torch::empty_like(input);
    const int total_elements = input.numel();

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    bias_relu_kernel<<<num_blocks, block_size, 0, stream.stream()>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        N,
        total_elements,
        output.data_ptr<float>()
    );
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor add_layer_norm_cuda(torch::Tensor x, torch::Tensor residual, torch::Tensor gamma, torch::Tensor beta, double epsilon);
torch::Tensor bias_relu_cuda(torch::Tensor input, torch::Tensor bias);
"""

# JIT compile the custom CUDA kernels
fused_ops = load_inline(
    name="fused_cvit_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["add_layer_norm_cuda", "bias_relu_cuda"],
    verbose=False,
)


class CustomTransformerEncoderLayer(nn.Module):
    """
    A TransformerEncoderLayer implementation that uses custom fused CUDA kernels
    for Add+LayerNorm and Bias+ReLU operations.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0, batch_first=True):
        super().__init__()
        # The original implementation uses dropout, but it's set to 0.0 in the model.
        # We omit it here for simplicity as it would be a no-op.
        assert dropout == 0.0, "Custom kernel version does not support dropout."
        assert batch_first, "Custom kernel version only supports batch_first=True."

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Replace nn.LayerNorm with custom parameters
        self.norm1_weight = nn.Parameter(torch.ones(d_model))
        self.norm1_bias = nn.Parameter(torch.zeros(d_model))
        self.norm2_weight = nn.Parameter(torch.ones(d_model))
        self.norm2_bias = nn.Parameter(torch.zeros(d_model))
        
        self.layer_norm_eps = 1e-5 # Default epsilon for nn.LayerNorm

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # PyTorch's MHA returns a tuple (output, weights)
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask,
                                      need_weights=False)
        
        # --- Custom Kernel Fusion 1: Add + LayerNorm ---
        x = fused_ops.add_layer_norm_cuda(src, attn_output, self.norm1_weight, self.norm1_bias, self.layer_norm_eps)

        # Feed-forward block
        # We need the matmul result before bias and activation
        ff_matmul_out = F.linear(x, self.linear1.weight)
        
        # --- Custom Kernel Fusion 2: Bias Add + ReLU ---
        ff_activated = fused_ops.bias_relu_cuda(ff_matmul_out, self.linear1.bias)
        
        ff_output = self.linear2(ff_activated)

        # --- Custom Kernel Fusion 3: Add + LayerNorm ---
        out = fused_ops.add_layer_norm_cuda(x, ff_output, self.norm2_weight, self.norm2_bias, self.layer_norm_eps)
        
        return out


class ModelNew(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6, 
                 mlp_ratio=4.0, patch_size=4, in_channels=3, image_size=32):
        """
        Optimized Convolutional Vision Transformer (CViT) implementation
        using custom fused CUDA kernels in the transformer layers.
        """
        super(ModelNew, self).__init__()

        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.linear_proj = nn.Linear(embed_dim * num_patches, embed_dim)

        # Use the custom transformer encoder layer
        self.transformer_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Forward pass of the CViT model.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of shape (B, num_classes)
        """
        B = x.size(0)
        x = self.conv1(x)
        x = x.flatten(start_dim=1)
        x = self.linear_proj(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)

        for layer in self.transformer_layers:
            x = layer(x)

        return self.fc_out(x[:, 0])