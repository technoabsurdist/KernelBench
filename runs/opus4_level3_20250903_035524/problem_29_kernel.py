import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Linear + GELU
fused_linear_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__device__ float gelu_func(float x) {
    const float c = 0.7978845608f; // sqrt(2/pi)
    const float a = 0.044715f;
    float x3 = x * x * x;
    float inner = c * (x + a * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void apply_gelu_inplace(float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = gelu_func(output[idx]);
    }
}

torch::Tensor fused_linear_gelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto output = torch::matmul(input, weight.t());
    if (bias.defined()) {
        output = output + bias;
    }
    
    const int size = output.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    apply_gelu_inplace<<<num_blocks, block_size>>>(output.data_ptr<float>(), size);
    
    return output;
}
"""

fused_linear_gelu_cpp_source = "torch::Tensor fused_linear_gelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Custom CUDA kernel for LayerNorm
layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

template<int BLOCK_SIZE>
__global__ void layernorm_kernel(const float* __restrict__ input,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ output,
                                 int batch_size, int hidden_size, float eps) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* input_ptr = input + batch_idx * hidden_size;
    float* output_ptr = output + batch_idx * hidden_size;
    
    __shared__ float shared_mean;
    __shared__ float shared_var;
    
    // Compute mean
    float thread_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        thread_sum += input_ptr[i];
    }
    
    __shared__ float reduction_buffer[BLOCK_SIZE];
    reduction_buffer[tid] = thread_sum;
    __syncthreads();
    
    if (tid == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < BLOCK_SIZE && i < hidden_size; i++) {
            total_sum += reduction_buffer[i];
        }
        shared_mean = total_sum / hidden_size;
    }
    __syncthreads();
    
    // Compute variance
    float thread_var = 0.0f;
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float diff = input_ptr[i] - shared_mean;
        thread_var += diff * diff;
    }
    
    reduction_buffer[tid] = thread_var;
    __syncthreads();
    
    if (tid == 0) {
        float total_var = 0.0f;
        for (int i = 0; i < BLOCK_SIZE && i < hidden_size; i++) {
            total_var += reduction_buffer[i];
        }
        shared_var = total_var / hidden_size;
    }
    __syncthreads();
    
    // Normalize and apply affine transformation
    float inv_std = rsqrtf(shared_var + eps);
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float normalized = (input_ptr[i] - shared_mean) * inv_std;
        output_ptr[i] = normalized * gamma[i] + beta[i];
    }
}

torch::Tensor layernorm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps) {
    auto batch_size = input.size(0) * input.size(1);
    auto hidden_size = input.size(2);
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    dim3 grid(batch_size);
    dim3 block(block_size);
    
    layernorm_kernel<block_size><<<grid, block>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, hidden_size, eps
    );
    
    return output;
}
"""

layernorm_cpp_source = "torch::Tensor layernorm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps);"

# Custom CUDA kernel for window partition
window_partition_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void window_partition_kernel(const float* input, float* output,
                                       int B, int H, int W, int C, int window_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * H * W * C;
    
    if (idx >= total_elements) return;
    
    int c = idx % C;
    int w = (idx / C) % W;
    int h = (idx / (C * W)) % H;
    int b = idx / (C * W * H);
    
    int h_win = h / window_size;
    int w_win = w / window_size;
    int h_in_win = h % window_size;
    int w_in_win = w % window_size;
    
    int num_win_h = H / window_size;
    int num_win_w = W / window_size;
    
    int out_idx = b * num_win_h * num_win_w * window_size * window_size * C +
                  h_win * num_win_w * window_size * window_size * C +
                  w_win * window_size * window_size * C +
                  h_in_win * window_size * C +
                  w_in_win * C +
                  c;
    
    output[out_idx] = input[idx];
}

torch::Tensor window_partition_cuda(torch::Tensor x, int window_size) {
    int B = x.size(0);
    int H = x.size(1);
    int W = x.size(2);
    int C = x.size(3);
    
    auto output = torch::empty({B * (H/window_size) * (W/window_size), window_size, window_size, C}, x.options());
    
    int total_elements = B * H * W * C;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    window_partition_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, H, W, C, window_size
    );
    
    return output;
}
"""

window_partition_cpp_source = "torch::Tensor window_partition_cuda(torch::Tensor x, int window_size);"

# Compile CUDA kernels
fused_linear_gelu = load_inline(
    name="fused_linear_gelu",
    cpp_sources=fused_linear_gelu_cpp_source,
    cuda_sources=fused_linear_gelu_source,
    functions=["fused_linear_gelu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

layernorm = load_inline(
    name="layernorm",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_source,
    functions=["layernorm_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

window_ops = load_inline(
    name="window_ops",
    cpp_sources=window_partition_cpp_source,
    cuda_sources=window_partition_source,
    functions=["window_partition_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class OptimizedLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        
    def forward(self, x):
        return layernorm.layernorm_cuda(x, self.weight, self.bias, self.eps)

class MlpNew(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = fused_linear_gelu.fused_linear_gelu_cuda(x, self.fc1.weight, self.fc1.bias)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    if x.is_cuda:
        return window_ops.window_partition_cuda(x, window_size)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinMLPBlockNew(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.padding = [self.window_size - self.shift_size, self.shift_size,
                        self.window_size - self.shift_size, self.shift_size]

        if norm_layer is None:
            norm_layer = OptimizedLayerNorm
        self.norm1 = norm_layer(dim)
        self.spatial_mlp = nn.Conv1d(self.num_heads * self.window_size ** 2,
                                     self.num_heads * self.window_size ** 2,
                                     kernel_size=1,
                                     groups=self.num_heads)

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpNew(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            shifted_x = F.pad(x, [0, 0, P_l, P_r, P_t, P_b], "constant", 0)
        else:
            shifted_x = x
        _, _H, _W, _ = shifted_x.shape

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        x_windows_heads = x_windows.view(-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads)
        x_windows_heads = x_windows_heads.transpose(1, 2)
        x_windows_heads = x_windows_heads.reshape(-1, self.num_heads * self.window_size * self.window_size,
                                                  C // self.num_heads)
        spatial_mlp_windows = self.spatial_mlp(x_windows_heads)
        spatial_mlp_windows = spatial_mlp_windows.view(-1, self.num_heads, self.window_size * self.window_size,
                                                       C // self.num_heads).transpose(1, 2)
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size * self.window_size, C)

        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(spatial_mlp_windows, self.window_size, _H, _W)

        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            x = shifted_x[:, P_t:-P_b, P_l:-P_r, :].contiguous()
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        if norm_layer is None:
            norm_layer = OptimizedLayerNorm
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=None, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        if norm_layer is None:
            norm_layer = OptimizedLayerNorm

        self.blocks = nn.ModuleList([
            SwinMLPBlockNew(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         shift_size=0 if (i % 2 == 0) else window_size // 2,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                         norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

class ModelNew(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
                 norm_layer=None, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        if norm_layer is None:
            norm_layer = OptimizedLayerNorm

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x