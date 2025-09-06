import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from itertools import repeat
import collections.abc

# ---------------------------------------------------------------------------
# Custom CUDA Kernels for Swin MLP
# ---------------------------------------------------------------------------

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// ------------------
// Fused Bias + GELU
// ------------------

// CUDA device function for GELU approximation
__device__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_bias_gelu_kernel(
    const float* input,
    const float* bias,
    float* output,
    int num_elements,
    int features) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int feature_idx = idx % features;
    float val = input[idx] + bias[feature_idx];
    output[idx] = gelu_approx(val);
}

// ------------------
// Window Partition
// ------------------

__global__ void window_partition_kernel(
    const float* input,
    float* output,
    int B, int H, int W, int C,
    int window_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * H * W * C;
    if (idx >= total_elements) return;

    // Deconstruct input index from flat index
    int c = idx % C;
    int w = (idx / C) % W;
    int h = (idx / (C * W)) % H;
    int b = idx / (C * W * H);

    // Calculate output coordinates
    int win_row = h / window_size;
    int win_col = w / window_size;
    int num_windows_w = W / window_size;
    int num_windows_h = H / window_size;

    int window_idx = b * (num_windows_h * num_windows_w) + win_row * num_windows_w + win_col;

    int out_r = h % window_size;
    int out_c = w % window_size;

    // Construct flat output index
    long long output_idx = (long long)window_idx * window_size * window_size * C +
                           (long long)out_r * window_size * C +
                           (long long)out_c * C +
                           c;

    output[output_idx] = input[idx];
}

// ------------------
// Window Reverse
// ------------------

__global__ void window_reverse_kernel(
    const float* input, // windows
    float* output,      // x
    int B, int H, int W, int C,
    int window_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * H * W * C;
    if (idx >= total_elements) return;

    // Deconstruct output index from flat index
    int c = idx % C;
    int w = (idx / C) % W;
    int h = (idx / (C * W)) % H;
    int b = idx / (C * W * H);

    // Calculate input coordinates from output coordinates
    int win_row = h / window_size;
    int win_col = w / window_size;
    int num_windows_w = W / window_size;
    int num_windows_h = H / window_size;

    int window_idx = b * (num_windows_h * num_windows_w) + win_row * num_windows_w + win_col;

    int in_r = h % window_size;
    int in_c = w % window_size;

    // Construct flat input index
    long long input_idx = (long long)window_idx * window_size * window_size * C +
                          (long long)in_r * window_size * C +
                          (long long)in_c * C +
                          c;

    output[idx] = input[input_idx];
}


// ---------------------------------------------------------------------------
// C++ Wrapper Functions
// ---------------------------------------------------------------------------

torch::Tensor fused_bias_gelu_cuda(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    TORCH_CHECK(input.size(-1) == bias.numel(), "Input feature dim must match bias size");

    auto out = torch::empty_like(input);
    const int num_elements = input.numel();
    const int features = input.size(-1);

    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    fused_bias_gelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        num_elements,
        features
    );
    return out;
}

torch::Tensor window_partition_cuda(torch::Tensor x, int window_size) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor (B, H, W, C)");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    const auto B = x.size(0);
    const auto H = x.size(1);
    const auto W = x.size(2);
    const auto C = x.size(3);
    TORCH_CHECK(H % window_size == 0, "Height must be divisible by window_size");
    TORCH_CHECK(W % window_size == 0, "Width must be divisible by window_size");

    const int nW_H = H / window_size;
    const int nW_W = W / window_size;
    const int num_windows = nW_H * nW_W;

    auto out = torch::empty({B * num_windows, window_size, window_size, C}, x.options());

    const int total_elements = B * H * W * C;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    window_partition_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        B, H, W, C,
        window_size
    );
    return out;
}

torch::Tensor window_reverse_cuda(torch::Tensor windows, int window_size, int H, int W) {
    TORCH_CHECK(windows.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(windows.dim() == 4, "Input must be a 4D tensor (nW*B, ws, ws, C)");
    TORCH_CHECK(windows.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(windows.size(1) == window_size && windows.size(2) == window_size, "Window dimensions must match window_size");

    const auto nWB = windows.size(0);
    const auto C = windows.size(3);
    const int num_windows_per_img = (H / window_size) * (W / window_size);
    TORCH_CHECK(nWB % num_windows_per_img == 0, "Total number of windows not divisible by windows per image");
    const int B = nWB / num_windows_per_img;

    auto out = torch::empty({B, H, W, C}, windows.options());

    const int total_elements = B * H * W * C;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    window_reverse_kernel<<<num_blocks, block_size>>>(
        windows.data_ptr<float>(),
        out.data_ptr<float>(),
        B, H, W, C,
        window_size
    );
    return out;
}
"""

cpp_source = """
torch::Tensor fused_bias_gelu_cuda(torch::Tensor input, torch::Tensor bias);
torch::Tensor window_partition_cuda(torch::Tensor x, int window_size);
torch::Tensor window_reverse_cuda(torch::Tensor windows, int window_size, int H, int W);
"""

# Compile the inline CUDA code
custom_ops = load_inline(
    name="swin_mlp_custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_bias_gelu_cuda", "window_partition_cuda", "window_reverse_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------------
# Optimized Swin MLP Architecture (ModelNew)
# ---------------------------------------------------------------------------

class MlpNew(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.act is replaced by the fused kernel
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # Fused fc1 + act
        x_matmul = F.linear(x, self.fc1.weight)
        x = custom_ops.fused_bias_gelu_cuda(x_matmul, self.fc1.bias)

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinMLPBlockNew(nn.Module):
    r""" Swin MLP Block with Custom CUDA Kernels. """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
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

        # partition windows using custom CUDA kernel
        x_windows = custom_ops.window_partition_cuda(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        x_windows_heads = x_windows.view(-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads)
        x_windows_heads = x_windows_heads.transpose(1, 2)
        x_windows_heads = x_windows_heads.reshape(-1, self.num_heads * self.window_size * self.window_size, C // self.num_heads)
        spatial_mlp_windows = self.spatial_mlp(x_windows_heads)
        spatial_mlp_windows = spatial_mlp_windows.view(-1, self.num_heads, self.window_size * self.window_size, C // self.num_heads).transpose(1, 2)
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size * self.window_size, C)

        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size, self.window_size, C)
        
        # merge windows using custom CUDA kernel
        shifted_x = custom_ops.window_reverse_cuda(spatial_mlp_windows, self.window_size, _H, _W)

        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            x = shifted_x[:, P_t:-P_b, P_l:-P_r, :].contiguous()
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayerNew(nn.Module):
    """ A basic Swin MLP layer for one stage, using optimized blocks. """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

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
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# ---------------------------------------------------------------------------
# Unchanged Helper Modules/Functions
# ---------------------------------------------------------------------------

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
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


# ---------------------------------------------------------------------------
# Final Assembled Model
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

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
            layer = BasicLayerNew(dim=int(embed_dim * 2 ** i_layer),
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