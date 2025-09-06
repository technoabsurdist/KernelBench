import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from itertools import repeat
from torch.utils.cpp_extension import load_inline

# --------------------------------------------------------
# Custom CUDA Kernels for Fused Operations
# --------------------------------------------------------

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

#define THREADS_PER_BLOCK 256

// A numerically stable GELU approximation
__device__ __forceinline__ float gelu_forward(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

__global__ void gelu_forward_kernel(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = gelu_forward(input[idx]);
    }
}

// Fused LayerNorm kernel
template <typename T>
__global__ void layer_norm_forward_kernel(
    T* out, T* mean, T* rstd, const T* in, int N, int C, const T* gamma, const T* beta, float epsilon) {

    int i = blockIdx.x;
    extern __shared__ float smem[];
    float* s_mean = smem;
    float* s_var = &smem[1];

    if (threadIdx.x == 0) {
        *s_mean = 0.0f;
        *s_var = 0.0f;
    }
    __syncthreads();

    float thread_mean = 0.0f;
    float thread_var = 0.0f;

    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        float val = static_cast<float>(in[i * C + j]);
        thread_mean += val;
        thread_var += val * val;
    }

    atomicAdd(s_mean, thread_mean);
    atomicAdd(s_var, thread_var);
    __syncthreads();

    if (threadIdx.x == 0) {
        *s_mean /= C;
        *s_var = *s_var / C - (*s_mean) * (*s_mean);
        mean[i] = *s_mean;
        rstd[i] = rsqrtf(*s_var + epsilon);
    }
    __syncthreads();

    float current_mean = mean[i];
    float current_rstd = rstd[i];

    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        float val = static_cast<float>(in[i * C + j]);
        float g = static_cast<float>(gamma[j]);
        float b = static_cast<float>(beta[j]);
        out[i * C + j] = static_cast<T>((val - current_mean) * current_rstd * g + b);
    }
}

// Fused Add(bias) + Add(mask) + Scale + Softmax
__global__ void fused_add_scale_softmax_kernel(float* out, const float* in, const float* bias, const float* mask, float scale, int N, int C) {
    int row = blockIdx.x;
    extern __shared__ float smem[];

    const float* in_row = in + row * C;
    const float* bias_row = bias + (row % N) * C;
    const float* mask_row = mask + row * C;
    float* out_row = out + row * C;

    // Step 1: Find max value in the row for stable softmax
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float val = in_row[i] * scale + bias_row[i] + mask_row[i];
        if (val > max_val) {
            max_val = val;
        }
    }

    // Block-wide reduction for max_val
    smem[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        }
        __syncthreads();
    }
    max_val = smem[0];
    __syncthreads();

    // Step 2: Compute exp and sum
    float sum_val = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float val = in_row[i] * scale + bias_row[i] + mask_row[i];
        float exp_val = expf(val - max_val);
        out_row[i] = exp_val;
        sum_val += exp_val;
    }

    // Block-wide reduction for sum_val
    smem[threadIdx.x] = sum_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    float sum_total = smem[0];
    __syncthreads();

    // Step 3: Normalize
    float inv_sum = 1.0f / sum_total;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        out_row[i] *= inv_sum;
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    const int size = input.numel();
    const int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gelu_forward_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(input.data_ptr<float>(), size);
    return input;
}

std::vector<torch::Tensor> layer_norm_cuda(
    torch::Tensor in, torch::Tensor gamma, torch::Tensor beta, float epsilon) {
    
    const auto in_sizes = in.sizes();
    const int N = in.size(0) * in.size(1);
    const int C = in.size(2);

    auto out = torch::empty_like(in);
    auto mean = torch::empty({N}, in.options());
    auto rstd = torch::empty({N}, in.options());

    const int block_size = THREADS_PER_BLOCK;
    const int num_blocks = N;
    
    layer_norm_forward_kernel<float><<<num_blocks, block_size, 2 * sizeof(float)>>>(
        out.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        in.data_ptr<float>(), N, C, gamma.data_ptr<float>(), beta.data_ptr<float>(), epsilon);

    return {out, mean, rstd};
}

torch::Tensor fused_add_scale_softmax_cuda(torch::Tensor in, torch::Tensor bias, torch::Tensor mask, float scale) {
    auto B_ = in.size(0);
    auto H = in.size(1);
    auto N = in.size(2);
    auto C = in.size(3);
    
    auto out = torch::empty_like(in);
    
    const int num_rows = B_ * H * N;
    const int block_size = THREADS_PER_BLOCK;
    const int shared_mem_size = block_size * sizeof(float);

    fused_add_scale_softmax_kernel<<<num_rows, block_size, shared_mem_size>>>(
        out.data_ptr<float>(), in.data_ptr<float>(), bias.data_ptr<float>(), mask.data_ptr<float>(), scale, H * N, C);
    
    return out;
}
"""

fused_ops_cpp_source = """
torch::Tensor gelu_cuda(torch::Tensor input);
std::vector<torch::Tensor> layer_norm_cuda(torch::Tensor in, torch::Tensor gamma, torch::Tensor beta, float epsilon);
torch::Tensor fused_add_scale_softmax_cuda(torch::Tensor in, torch::Tensor bias, torch::Tensor mask, float scale);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["gelu_cuda", "layer_norm_cuda", "fused_add_scale_softmax_cuda"],
    verbose=False,
)

# --------------------------------------------------------
# PyTorch Module Definitions with Custom Kernels
# --------------------------------------------------------

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

class CustomGELU(nn.Module):
    def forward(self, x):
        return fused_ops.gelu_cuda(x)

class LayerNormNew(nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_3d = input.reshape(-1, 1, input.shape[-1])
        output, _, _ = fused_ops.layer_norm_cuda(input_3d, self.weight, self.bias, self.eps)
        return output.reshape(input.shape)

class MlpNew(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=CustomGELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttentionNew(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w], indexing='ij')).permute(1, 2, 0).contiguous().unsqueeze(0)
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=x.device))).exp()

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1).permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        
        # Prepare for fused kernel
        attn_bias = relative_position_bias.unsqueeze(0).expand(B_, -1, -1, -1)
        
        if mask is not None:
            nW = mask.shape[0]
            # The mask needs to be broadcasted to match attn shape
            final_mask = mask.unsqueeze(1).unsqueeze(0) # 1, nW, 1, N, N
            final_mask = final_mask.expand(B_ // nW, -1, self.num_heads, -1, -1).reshape(B_, self.num_heads, N, N)
            attn_bias = attn_bias + final_mask
        
        # Reshape for kernel: kernel expects (B*H*N, C) layout
        attn_scores_flat = attn.reshape(-1, N)
        attn_bias_flat = attn_bias.reshape(-1, N)
        # The mask in the kernel is a placeholder, we've already added it to the bias
        dummy_mask_flat = torch.zeros_like(attn_scores_flat)

        attn = fused_ops.fused_add_scale_softmax_cuda(
            attn_scores_flat.unsqueeze(-1), # Add a dummy dim
            attn_bias_flat.unsqueeze(-1),
            dummy_mask_flat.unsqueeze(-1),
            logit_scale.item() # Assuming logit_scale is scalar after exp()
        ).reshape(B_, self.num_heads, N, N)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlockNew(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=CustomGELU, norm_layer=LayerNormNew, pretrained_window_size=0):
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

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttentionNew(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpNew(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMergingNew(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=LayerNormNew):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
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

class BasicLayerNew(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=LayerNormNew, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlockNew(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, pretrained_window_size=pretrained_window_size)
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

class PatchEmbedNew(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=LayerNormNew):
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
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=LayerNormNew, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbedNew(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayerNew(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMergingNew if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
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