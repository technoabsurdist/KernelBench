import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Conv2d + Flatten + Linear
conv_flatten_linear_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void im2col_kernel(const float* input, float* col, int batch_size,
                              int in_channels, int input_h, int input_w,
                              int kernel_size, int stride, int out_h, int out_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * in_channels * out_h * out_w * kernel_size * kernel_size;
    
    if (idx >= total_elements) return;
    
    int kernel_area = kernel_size * kernel_size;
    int patch_area = out_h * out_w;
    int channel_patch_area = patch_area * kernel_area;
    int batch_channel_area = channel_patch_area * in_channels;
    
    int b = idx / batch_channel_area;
    int c = (idx % batch_channel_area) / channel_patch_area;
    int p = (idx % channel_patch_area) / kernel_area;
    int k = idx % kernel_area;
    
    int out_y = p / out_w;
    int out_x = p % out_w;
    int k_y = k / kernel_size;
    int k_x = k % kernel_size;
    
    int in_y = out_y * stride + k_y;
    int in_x = out_x * stride + k_x;
    
    if (in_y < input_h && in_x < input_w) {
        col[idx] = input[((b * in_channels + c) * input_h + in_y) * input_w + in_x];
    } else {
        col[idx] = 0.0f;
    }
}

__global__ void matmul_add_kernel(const float* a, const float* b, const float* bias,
                                  float* out, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += a[row * K + k] * b[k * N + col];
        }
        out[row * N + col] = sum + bias[col];
    }
}

torch::Tensor conv_flatten_linear_fused_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                             int kernel_size, int stride) {
    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_h = input_sizes[2];
    int input_w = input_sizes[3];
    
    int out_h = (input_h - kernel_size) / stride + 1;
    int out_w = (input_w - kernel_size) / stride + 1;
    int num_patches = out_h * out_w;
    int embed_dim = weight.size(0);
    
    // Im2col
    auto col = torch::zeros({batch_size, in_channels * kernel_size * kernel_size, num_patches}, 
                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_elements = batch_size * in_channels * num_patches * kernel_size * kernel_size;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    im2col_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), col.data_ptr<float>(), batch_size, in_channels,
        input_h, input_w, kernel_size, stride, out_h, out_w
    );
    
    // MatMul + Bias
    auto output = torch::zeros({batch_size, embed_dim}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    dim3 block_dim(16, 16);
    dim3 grid_dim((embed_dim + block_dim.x - 1) / block_dim.x, 
                  (batch_size + block_dim.y - 1) / block_dim.y);
    
    matmul_add_kernel<<<grid_dim, block_dim>>>(
        col.transpose(1, 2).contiguous().data_ptr<float>(), // (B, num_patches, in_features)
        weight.transpose(0, 1).contiguous().data_ptr<float>(), // (in_features, embed_dim)
        bias.data_ptr<float>(),
        batch_size, embed_dim, col.size(1)
    );
    
    return output;
}
"""

conv_flatten_linear_fused_cpp_source = """
torch::Tensor conv_flatten_linear_fused_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                             int kernel_size, int stride);
"""

# Compile the inline CUDA code
conv_flatten_linear_fused = load_inline(
    name="conv_flatten_linear_fused",
    cpp_sources=conv_flatten_linear_fused_cpp_source,
    cuda_sources=conv_flatten_linear_fused_source,
    functions=["conv_flatten_linear_fused_cuda"],
    verbose=False,
)

# Custom CUDA kernel for fused attention (QKV computation + attention)
attention_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void qkv_compute_kernel(const float* input, const float* w_q, const float* w_k, const float* w_v,
                                   const float* b_q, const float* b_k, const float* b_v,
                                   float* q_out, float* k_out, float* v_out,
                                   int batch_size, int seq_len, int embed_dim, int head_dim, int num_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * embed_dim;
    
    if (idx >= total_elements) return;
    
    int b = idx / (seq_len * embed_dim);
    int s = (idx % (seq_len * embed_dim)) / embed_dim;
    int e = idx % embed_dim;
    
    float q_val = b_q[e];
    float k_val = b_k[e];
    float v_val = b_v[e];
    
    for (int i = 0; i < embed_dim; ++i) {
        float input_val = input[(b * seq_len + s) * embed_dim + i];
        q_val += input_val * w_q[i * embed_dim + e];
        k_val += input_val * w_k[i * embed_dim + e];
        v_val += input_val * w_v[i * embed_dim + e];
    }
    
    q_out[idx] = q_val;
    k_out[idx] = k_val;
    v_out[idx] = v_val;
}

__global__ void attention_score_kernel(const float* q, const float* k, float* scores,
                                       int batch_size, int num_heads, int seq_len, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_heads * seq_len * seq_len;
    
    if (idx >= total_elements) return;
    
    int b = idx / (num_heads * seq_len * seq_len);
    int h = (idx % (num_heads * seq_len * seq_len)) / (seq_len * seq_len);
    int i = (idx % (seq_len * seq_len)) / seq_len;
    int j = idx % seq_len;
    
    float sum = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        int q_idx = ((b * seq_len + i) * num_heads + h) * head_dim + d;
        int k_idx = ((b * seq_len + j) * num_heads + h) * head_dim + d;
        sum += q[q_idx] * k[k_idx];
    }
    
    scores[idx] = sum / sqrtf((float)head_dim);
}

__global__ void softmax_kernel(float* scores, int batch_size, int num_heads, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_softmax = batch_size * num_heads * seq_len;
    
    if (idx >= total_softmax) return;
    
    int b = idx / (num_heads * seq_len);
    int h = (idx % (num_heads * seq_len)) / seq_len;
    int i = idx % seq_len;
    
    // Compute max for numerical stability
    float max_val = scores[(b * num_heads + h) * seq_len * seq_len + i * seq_len];
    for (int j = 1; j < seq_len; ++j) {
        float val = scores[(b * num_heads + h) * seq_len * seq_len + i * seq_len + j];
        if (val > max_val) max_val = val;
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        float& val = scores[(b * num_heads + h) * seq_len * seq_len + i * seq_len + j];
        val = expf(val - max_val);
        sum += val;
    }
    
    // Normalize
    for (int j = 0; j < seq_len; ++j) {
        float& val = scores[(b * num_heads + h) * seq_len * seq_len + i * seq_len + j];
        val /= sum;
    }
}

__global__ void attention_output_kernel(const float* scores, const float* v, float* output,
                                        int batch_size, int num_heads, int seq_len, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * num_heads * head_dim;
    
    if (idx >= total_elements) return;
    
    int b = idx / (seq_len * num_heads * head_dim);
    int s = (idx % (seq_len * num_heads * head_dim)) / (num_heads * head_dim);
    int h = (idx % (num_heads * head_dim)) / head_dim;
    int d = idx % head_dim;
    
    float sum = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        float score = scores[(b * num_heads + h) * seq_len * seq_len + s * seq_len + j];
        float v_val = v[((b * seq_len + j) * num_heads + h) * head_dim + d];
        sum += score * v_val;
    }
    
    output[idx] = sum;
}

torch::Tensor attention_fused_cuda(torch::Tensor input, 
                                   torch::Tensor w_q, torch::Tensor w_k, torch::Tensor w_v,
                                   torch::Tensor b_q, torch::Tensor b_k, torch::Tensor b_v,
                                   int num_heads) {
    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int seq_len = input_sizes[1];
    int embed_dim = input_sizes[2];
    int head_dim = embed_dim / num_heads;
    
    // QKV computation
    auto q = torch::zeros({batch_size, seq_len, embed_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto k = torch::zeros({batch_size, seq_len, embed_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto v = torch::zeros({batch_size, seq_len, embed_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_elements = batch_size * seq_len * embed_dim;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    qkv_compute_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        w_q.data_ptr<float>(), w_k.data_ptr<float>(), w_v.data_ptr<float>(),
        b_q.data_ptr<float>(), b_k.data_ptr<float>(), b_v.data_ptr<float>(),
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        batch_size, seq_len, embed_dim, head_dim, num_heads
    );
    
    // Reshape for multi-head attention
    q = q.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2).contiguous(); // (B, H, S, D)
    k = k.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2).contiguous(); // (B, H, S, D)
    v = v.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2).contiguous(); // (B, H, S, D)
    
    // Attention scores
    auto scores = torch::zeros({batch_size, num_heads, seq_len, seq_len}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    total_elements = batch_size * num_heads * seq_len * seq_len;
    num_blocks = (total_elements + block_size - 1) / block_size;
    
    attention_score_kernel<<<num_blocks, block_size>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), scores.data_ptr<float>(),
        batch_size, num_heads, seq_len, head_dim
    );
    
    // Softmax
    total_elements = batch_size * num_heads * seq_len;
    num_blocks = (total_elements + block_size - 1) / block_size;
    
    softmax_kernel<<<num_blocks, block_size>>>(
        scores.data_ptr<float>(), batch_size, num_heads, seq_len
    );
    
    // Output computation
    auto output = torch::zeros({batch_size, seq_len, num_heads, head_dim}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    total_elements = batch_size * seq_len * num_heads * head_dim;
    num_blocks = (total_elements + block_size - 1) / block_size;
    
    attention_output_kernel<<<num_blocks, block_size>>>(
        scores.data_ptr<float>(), v.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, num_heads, seq_len, head_dim
    );
    
    // Reshape output
    output = output.transpose(1, 2).contiguous().view({batch_size, seq_len, embed_dim});
    
    return output;
}
"""

attention_fused_cpp_source = """
torch::Tensor attention_fused_cuda(torch::Tensor input, 
                                   torch::Tensor w_q, torch::Tensor w_k, torch::Tensor w_v,
                                   torch::Tensor b_q, torch::Tensor b_k, torch::Tensor b_v,
                                   int num_heads);
"""

# Compile the attention fused kernel
attention_fused = load_inline(
    name="attention_fused",
    cpp_sources=attention_fused_cpp_source,
    cuda_sources=attention_fused_source,
    functions=["attention_fused_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6, 
                 mlp_ratio=4.0, patch_size=4, in_channels=3, image_size=32):
        """
        Optimized Convolutional Vision Transformer (CViT) with custom CUDA kernels.
        """
        super(ModelNew, self).__init__()

        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Replace Conv2d + Flatten + Linear with fused kernel
        self.conv_weight = nn.Parameter(torch.randn(embed_dim, in_channels, patch_size, patch_size))
        self.conv_bias = nn.Parameter(torch.randn(embed_dim))
        
        num_patches = (image_size // patch_size) ** 2
        self.linear_proj_weight = nn.Parameter(torch.randn(embed_dim, embed_dim * num_patches))
        self.linear_proj_bias = nn.Parameter(torch.randn(embed_dim))

        # Transformer layers with custom attention
        self.transformer_layers = nn.ModuleList([])
        for _ in range(num_layers):
            # Self-attention components
            self_attn = nn.Module()
            self_attn.w_q = nn.Parameter(torch.randn(embed_dim, embed_dim))
            self_attn.w_k = nn.Parameter(torch.randn(embed_dim, embed_dim))
            self_attn.w_v = nn.Parameter(torch.randn(embed_dim, embed_dim))
            self_attn.b_q = nn.Parameter(torch.randn(embed_dim))
            self_attn.b_k = nn.Parameter(torch.randn(embed_dim))
            self_attn.b_v = nn.Parameter(torch.randn(embed_dim))
            self_attn.out_proj = nn.Linear(embed_dim, embed_dim)
            
            # MLP components
            mlp_dim = int(embed_dim * mlp_ratio)
            mlp = nn.Module()
            mlp.fc1 = nn.Linear(embed_dim, mlp_dim)
            mlp.fc2 = nn.Linear(mlp_dim, embed_dim)
            mlp.activation = nn.GELU()
            
            layer = nn.Module()
            layer.self_attn = self_attn
            layer.mlp = mlp
            layer.norm1 = nn.LayerNorm(embed_dim)
            layer.norm2 = nn.LayerNorm(embed_dim)
            
            self.transformer_layers.append(layer)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc_out = nn.Linear(embed_dim, num_classes)
        
        # Register custom CUDA functions
        self.conv_flatten_linear_fused = conv_flatten_linear_fused
        self.attention_fused = attention_fused

    def forward(self, x):
        """
        Forward pass of the optimized CViT model.
        """
        B = x.size(0)
        
        # Fused Conv2d + Flatten + Linear
        x = self.conv_flatten_linear_fused.conv_flatten_linear_fused_cuda(
            x, self.linear_proj_weight, self.linear_proj_bias,
            self.patch_size, self.patch_size
        )  # (B, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)  # (B, 2, embed_dim)

        for layer in self.transformer_layers:
            # Self-attention with custom CUDA kernel
            residual = x
            x = layer.norm1(x)
            
            # Use custom fused attention
            attn_out = self.attention_fused.attention_fused_cuda(
                x,
                layer.self_attn.w_q, layer.self_attn.w_k, layer.self_attn.w_v,
                layer.self_attn.b_q, layer.self_attn.b_k, layer.self_attn.b_v,
                self.num_heads
            )
            attn_out = layer.self_attn.out_proj(attn_out)
            x = residual + attn_out
            
            # MLP
            residual = x
            x = layer.norm2(x)
            x = layer.mlp.fc1(x)
            x = layer.mlp.activation(x)
            x = layer.mlp.fc2(x)
            x = residual + x

        return self.fc_out(x[:, 0])  # Use [CLS] token for classification