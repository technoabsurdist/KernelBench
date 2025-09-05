import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Fused Linear + GELU kernel
fused_linear_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void gelu_kernel(float* output, const float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        output[idx] = x * cdf;
    }
}

torch::Tensor fused_linear_gelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto in_features = input.size(2);
    auto out_features = weight.size(0);
    
    auto input_2d = input.reshape({batch_size * seq_len, in_features});
    auto output = torch::matmul(input_2d, weight.t());
    
    if (bias.defined()) {
        output = output + bias;
    }
    
    const int block_size = 256;
    const int num_blocks = (output.numel() + block_size - 1) / block_size;
    gelu_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(),
        output.data_ptr<float>(),
        output.numel()
    );
    
    return output.reshape({batch_size, seq_len, out_features});
}
"""

fused_linear_gelu_cpp_source = """
torch::Tensor fused_linear_gelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

fused_linear_gelu = load_inline(
    name="fused_linear_gelu",
    cpp_sources=fused_linear_gelu_cpp_source,
    cuda_sources=fused_linear_gelu_source,
    functions=["fused_linear_gelu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Optimized patch extraction and embedding kernel
patch_embed_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void patch_extract_embed_kernel(
    const float* img, const float* weight, const float* bias,
    float* output, int batch_size, int channels, int image_size,
    int patch_size, int num_patches_per_dim, int dim, int patch_dim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_patches_per_dim * num_patches_per_dim * dim;
    
    if (idx < total_elements) {
        int b = idx / (num_patches_per_dim * num_patches_per_dim * dim);
        int remainder = idx % (num_patches_per_dim * num_patches_per_dim * dim);
        int patch_idx = remainder / dim;
        int d = remainder % dim;
        
        int patch_row = patch_idx / num_patches_per_dim;
        int patch_col = patch_idx % num_patches_per_dim;
        
        float sum = 0.0f;
        for (int c = 0; c < channels; c++) {
            for (int py = 0; py < patch_size; py++) {
                for (int px = 0; px < patch_size; px++) {
                    int img_row = patch_row * patch_size + py;
                    int img_col = patch_col * patch_size + px;
                    int img_idx = b * channels * image_size * image_size +
                                  c * image_size * image_size +
                                  img_row * image_size + img_col;
                    int patch_elem_idx = c * patch_size * patch_size + py * patch_size + px;
                    sum += img[img_idx] * weight[d * patch_dim + patch_elem_idx];
                }
            }
        }
        if (bias != nullptr) {
            sum += bias[d];
        }
        output[idx] = sum;
    }
}

torch::Tensor patch_extract_embed_cuda(
    torch::Tensor img, torch::Tensor weight, torch::Tensor bias,
    int patch_size) {
    
    auto batch_size = img.size(0);
    auto channels = img.size(1);
    auto image_size = img.size(2);
    auto dim = weight.size(0);
    auto patch_dim = weight.size(1);
    
    int num_patches_per_dim = image_size / patch_size;
    int num_patches = num_patches_per_dim * num_patches_per_dim;
    
    auto output = torch::zeros({batch_size, num_patches, dim}, img.options());
    
    const int block_size = 256;
    const int num_elements = batch_size * num_patches * dim;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    
    patch_extract_embed_kernel<<<num_blocks, block_size>>>(
        img.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, channels, image_size, patch_size,
        num_patches_per_dim, dim, patch_dim
    );
    
    return output;
}
"""

patch_embed_cpp_source = """
torch::Tensor patch_extract_embed_cuda(torch::Tensor img, torch::Tensor weight, torch::Tensor bias, int patch_size);
"""

patch_embed = load_inline(
    name="patch_embed",
    cpp_sources=patch_embed_cpp_source,
    cuda_sources=patch_embed_source,
    functions=["patch_extract_embed_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Fused position embedding addition
pos_embed_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void pos_embed_add_kernel(float* x, const float* pos_emb, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] += pos_emb[idx];
    }
}

torch::Tensor pos_embed_add_cuda(torch::Tensor x, torch::Tensor pos_embedding) {
    auto batch_size = x.size(0);
    auto seq_len = x.size(1);
    auto dim = x.size(2);
    
    const int block_size = 256;
    const int num_elements = batch_size * seq_len * dim;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    
    auto x_contiguous = x.contiguous();
    auto pos_expanded = pos_embedding.expand({batch_size, seq_len, dim}).contiguous();
    
    pos_embed_add_kernel<<<num_blocks, block_size>>>(
        x_contiguous.data_ptr<float>(),
        pos_expanded.data_ptr<float>(),
        num_elements
    );
    
    return x_contiguous;
}
"""

pos_embed_add_cpp_source = """
torch::Tensor pos_embed_add_cuda(torch::Tensor x, torch::Tensor pos_embedding);
"""

pos_embed_add = load_inline(
    name="pos_embed_add",
    cpp_sources=pos_embed_add_cpp_source,
    cuda_sources=pos_embed_add_source,
    functions=["pos_embed_add_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super(ModelNew, self).__init__()
        
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )
        
        self.to_cls_token = nn.Identity()
        
        # MLP head with custom kernels
        self.mlp_linear1 = nn.Linear(dim, mlp_dim)
        self.mlp_dropout = nn.Dropout(dropout)
        self.mlp_linear2 = nn.Linear(mlp_dim, num_classes)
        
        # Custom CUDA kernels
        self.fused_linear_gelu = fused_linear_gelu
        self.patch_embed = patch_embed
        self.pos_embed_add = pos_embed_add
    
    def forward(self, img):
        p = self.patch_size
        
        # Use custom patch extraction and embedding
        if img.is_cuda:
            x = self.patch_embed.patch_extract_embed_cuda(
                img.contiguous(),
                self.patch_to_embedding.weight,
                self.patch_to_embedding.bias,
                p
            )
        else:
            x = img.unfold(2, p, p).unfold(3, p, p).reshape(img.shape[0], -1, p*p*img.shape[1])
            x = self.patch_to_embedding(x)
        
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Use custom position embedding addition
        if x.is_cuda:
            x = self.pos_embed_add.pos_embed_add_cuda(x, self.pos_embedding)
        else:
            x += self.pos_embedding
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = self.to_cls_token(x[:, 0])
        
        # Use fused Linear+GELU for MLP head
        if x.is_cuda:
            x = x.unsqueeze(1)  # Add sequence dimension for kernel compatibility
            x = self.fused_linear_gelu.fused_linear_gelu_cuda(
                x, self.mlp_linear1.weight, self.mlp_linear1.bias
            )
            x = x.squeeze(1)  # Remove sequence dimension
        else:
            x = self.mlp_linear1(x)
            x = F.gelu(x)
        
        x = self.mlp_dropout(x)
        x = self.mlp_linear2(x)
        
        return x