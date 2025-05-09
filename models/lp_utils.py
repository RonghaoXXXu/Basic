import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def bchw_to_blc(x: torch.Tensor) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, C, H, W) to (B, L, C)."""
    return x.flatten(2).transpose(1, 2)


def blc_to_bchw(x, x_size) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, L, C) to (B, C, H, W)."""
    B, L, C = x.shape
    return x.transpose(1, 2).view(B, C, *x_size)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat, reduction=4):
        super(RCAB, self).__init__()
        
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat, 3, 1, 1),
                                  nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                  nn.Conv2d(n_feat, n_feat, 3, 1, 1),
                                  CALayer(n_feat, reduction))

    def forward(self, x):
        res = self.body(x) # x: [4, 48, 8, 8]; res: [4, 48, 8, 8]
        res += x
        return res


# class LocalContextExtractor(nn.Module):

#     def __init__(self, dim, reduction=8):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(dim, dim // reduction, kernel_size=1, padding=0, bias=True),
#             nn.Conv2d(dim // reduction, dim // reduction, kernel_size=3, padding=1, bias=True),
#             nn.Conv2d(dim // reduction, dim, kernel_size=1, padding=0, bias=True),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         )
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(dim, dim // reduction, bias=False),
#             nn.ReLU(),
#             nn.Linear(dim // reduction, dim, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.conv(x) # Eq.(3)中的conv block [16 60 128 128]
#         B, C, _, _ = x.size()
#         y = self.avg_pool(x).view(B, C) # [16 60]
#         y = self.fc(y).view(B, C, 1, 1) # [16 60 1 1]
#         return x * y.expand_as(x) # [16 60 128 128]


class SpatialTemporalTransformer(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        # self.lce = LocalContextExtractor(self.dim)
        # Residual Channel Attention Block (RCAB)
        # self.rcab = RCAB(self.dim, reduction=dim//4)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape # [16, 16384, 60]
        # assert L == H * W, "input feature has wrong size"

        shortcut = x # Fig.2(a)
        x = self.norm1(x)
        x = x.view(B, H, W, C) # [16, 128, 128, 60]
        # local context features # [16, 60, 128, 128]
        # rcab = x.permute(0, 3, 1, 2).contiguous() # note:contiguous()原来没有加 

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C) # MSA后的结果

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # local context
        # lc = self.lce(lcf)
        # rcab = self.rcab(rcab)
        # rcab = rcab.view(B, C, H * W).permute(0, 2, 1)
        # x = rcab + x # context fusion Fig.2(a)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SpatialTemporalTransformer(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size) # B L C
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class SpatialTemporalTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.dilated_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=2, bias=True, dilation=2)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.dilated_conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, kernel_size=3, padding=2, dilation=2), 
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, kernel_size=3, padding=2, dilation=2)
                )

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        res = self.residual_group(x, x_size) # B L C
        res = self.patch_unembed(res, x_size) # B c H W
        res = self.dilated_conv(res) # dilated conv layer
        res = self.patch_embed(res) + x # Fig.3(b), 下面的那个残差连接
        return res

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()
        return flops


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

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # [B C H W] --> [B C H*W] --> [B, H*W, C]=[16, 16384, 60]
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):

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

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


# class SpatialAttentionModule(nn.Module):

#     def __init__(self, dim):
#         super(SpatialAttentionModule, self).__init__()
#         self.att1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, bias=True)
#         self.att2 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True)
#         self.relu = nn.LeakyReLU()

#     def forward(self, x1, x2):
#         f_cat = torch.cat((x1, x2), dim=1) # concatenate each non-reference feature with the reference feature
#         att_map = torch.sigmoid(self.att2(self.relu(self.att1(f_cat))))
#         return att_map


# class TAFusion(nn.Module):
#     """Temporal Spatial Attention (TSA) fusion module.

#     Temporal: Calculate the correlation between center frame and
#         neighboring frames;
#     Spatial: It has 3 pyramid levels, the attention is similar to SFT.
#         (SFT: Recovering realistic texture in image super-resolution by deep
#             spatial feature transform.)

#     Args:
#         num_feat (int): Channel number of middle features. Default: 64.
#         num_frame (int): Number of frames. Default: 5.
#         center_frame_idx (int): The index of center frame. Default: 2.
#     """

#     def __init__(self, num_feat=64, num_frame=3):
#         super(TAFusion, self).__init__()
#         # self.center_frame_idx = center_frame_idx
#         # temporal attention (before fusion conv)
#         self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 3, 1, 1)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

#     def forward(self, f1, f2, f3):
#         """
#         Args:
#             input_fea (Tensor): Aligned features with shape (b, t, c, h, w).

#         Returns:
#             Tensor: Features after TSA with the shape (b, c, h, w).
#         """
#         input_no_ref_fea = torch.stack([f1, f3], dim=1)
#         b, t, c, h, w = input_no_ref_fea.size() # [1, 3, 64, 256, 256]
#         # temporal attention
#         embedding_ref = self.temporal_attn1(f2.clone()) # [1, 64, 256, 256]
#         embedding = self.temporal_attn2(input_no_ref_fea.view(-1, c, h, w)) # [1, 5, 64, 256, 256] -> [5, 64, 256, 256] -> [5, 64, 256, 256]
#         embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w) # [1, 5, 64, 256, 256]

#         corr_l = []  # correlation list
#         for i in range(t):
#             emb_neighbor = embedding[:, i, :, :, :] # [1, 64, 256, 256]

#             # equal to: corr = troch.sum(emb_neighbor * embedding_ref, 1, keepdim=True)
#             corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)  # [1, 256, 256]
#             corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)

#         corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)  # [1, 5, 256, 256]
#         corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w) # [1, 5, 64, 256, 256]
#         corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
#         input_no_ref_fea = input_no_ref_fea.view(b, -1, h, w) * corr_prob # [1, 320, 256, 256]

#         # fusion
#         input_fea = self.lrelu(self.feat_fusion(torch.cat([input_no_ref_fea, f2], dim=1))) # [1, 64, 256, 256]

#         return input_fea

class TAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.

    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=64, num_frame=3, center_frame_idx=1):
        super(TAFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, input_fea):
        """
        Args:
            input_fea (Tensor): Aligned features with shape (b, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = input_fea.size() # [1, 3, 64, 256, 256]
        # temporal attention
        embedding_ref = self.temporal_attn1(
            input_fea[:, self.center_frame_idx, :, :, :].clone()) # [1, 64, 256, 256]
        embedding = self.temporal_attn2(input_fea.view(-1, c, h, w)) # [1, 5, 64, 256, 256] -> [5, 64, 256, 256] -> [5, 64, 256, 256]
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w) # [1, 5, 64, 256, 256]

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :] # [1, 64, 256, 256]

            # equal to: corr = troch.sum(emb_neighbor * embedding_ref, 1, keepdim=True)
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)  # [1, 256, 256]
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)

        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)  # [1, 5, 256, 256]
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w) # [1, 5, 64, 256, 256]
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        input_fea = input_fea.view(b, -1, h, w) * corr_prob # [1, 320, 256, 256]

        # fusion
        # input_fea = self.lrelu(self.feat_fusion(input_fea)) # [1, 64, 256, 256]

        return input_fea

class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1):
        super(conv_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, 
                      stride=stride, padding=padding)
            # nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


class ReduceChannel(nn.Module):
    def __init__(self, n_feat):
        super(ReduceChannel, self).__init__()
        self.conv = nn.Conv2d(n_feat, n_feat // 2, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(n_feat, n_feat // 2, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x