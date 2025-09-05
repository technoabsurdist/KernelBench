import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for patch embedding
patch_embedding_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void patch_embedding_kernel(
    const float* patches,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int num_patches,
    int patch_dim,
    int embed_dim
) {
    int batch_idx = blockIdx.x;
    int patch_idx = blockIdx.y;
    int embed_idx = threadIdx.x + blockDim.x * threadIdx.y;
    
    if (batch_idx >= batch_size || patch_idx >= num_patches || embed_idx >= embed_dim) return;
    
    int patch_offset = batch_idx * num_patches * patch_dim + patch_idx * patch_dim;
    int out_offset = batch_idx * num_patches * embed_dim + patch_idx * embed_dim;
    
    float sum = 0.0f;
    for (int i = 0; i < patch_dim; ++i) {
        sum += patches[patch_offset + i] * weight[embed_idx * patch_dim + i];
    }
    output[out_offset + embed_idx] = sum + bias[embed_idx];
}

torch::Tensor patch_embedding_cuda(torch::Tensor patches, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = patches.size(0);
    int num_patches = patches.size(1);
    int patch_dim = patches.size(2);
    int embed_dim = weight.size(0);
    
    auto output = torch::zeros({batch_size, num_patches, embed_dim}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    dim3 block_dim(16, 16);
    dim3 grid_dim(batch_size, (num_patches + 15) / 16);
    
    patch_embedding_kernel<<<grid_dim, block_dim>>>(
        patches.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_patches,
        patch_dim,
        embed_dim
    );
    
    return output;
}
"""

patch_embedding_cpp_source = """
torch::Tensor patch_embedding_cuda(torch::Tensor patches, torch::Tensor weight, torch::Tensor bias);
"""

# Custom CUDA kernel for GELU activation
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    gelu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    
    return output;
}
"""

gelu_cpp_source = """
torch::Tensor gelu_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
patch_embedding = load_inline(
    name="patch_embedding",
    cpp_sources=patch_embedding_cpp_source,
    cuda_sources=patch_embedding_source,
    functions=["patch_embedding_cuda"],
    verbose=False,
)

gelu = load_inline(
    name="gelu",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        """
        Optimized Vision Transformer (ViT) model with custom CUDA kernels.
        """
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
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )
        
        # Custom CUDA modules
        self.patch_embedding_cuda = patch_embedding
        self.gelu_cuda = gelu

    def forward(self, img):
        """
        Forward pass of the optimized Vision Transformer.
        """
        p = self.patch_size
        
        # Extract patches using unfold
        x = img.unfold(2, p, p).unfold(3, p, p).reshape(img.shape[0], -1, p*p*img.shape[1])
        
        # Use custom CUDA kernel for patch embedding
        x = self.patch_embedding_cuda.patch_embedding_cuda(
            x,
            self.patch_to_embedding.weight.T.contiguous(),
            self.patch_to_embedding.bias
        )
        
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = self.to_cls_token(x[:, 0])
        
        # Standard linear layers
        x = self.mlp_head[0](x)
        x = self.mlp_head[1](x)
        
        # Use custom CUDA kernel for GELU activation
        x = self.gelu_cuda.gelu_cuda(x)
        
        # Final linear layer
        x = self.mlp_head[3](x)
        
        return x