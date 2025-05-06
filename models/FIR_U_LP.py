#-*- coding:utf-8 -*-
import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys 
sys.path.append(os.getcwd())
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY

from thop import profile
from typing import Tuple
# from utils.ops import *
import pywt
from torch.autograd import Function
from torch.autograd import Variable, gradcheck


class Lap_Pyramid_Bicubic(nn.Module):
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Bicubic, self).__init__()

        self.interpolate_mode = 'bicubic'
        self.num_high = num_high

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for i in range(self.num_high):
            down = nn.functional.interpolate(current, size=(current.shape[2] // 2, current.shape[3] // 2), mode=self.interpolate_mode, align_corners=True)
            up = nn.functional.interpolate(down, size=(current.shape[2], current.shape[3]), mode=self.interpolate_mode, align_corners=True)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            image = F.interpolate(image, size=(level.shape[2], level.shape[3]), mode=self.interpolate_mode, align_corners=True) + level
        return image

class Lap_Pyramid_Conv(nn.Module):
    r"""
    Args:
        num_high (int): Number of high-frequency components
    """
    def __init__(self, num_high=3, gauss_chans=24):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.channels = gauss_chans
        self.kernel = self.gauss_kernel(channels=self.channels)
        

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256. # normalize
        kernel = kernel.repeat(channels, 1, 1, 1) # size -> [channels, 1, 5, 5]
        kernel = kernel.to(device)
        return kernel

    # downsamples the image by rejecting even rows and colums
    # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=pyrdown#void%20cvPyrDown(const%20CvArr*%20src,%20CvArr*%20dst,%20int%20filter)
    def downsample(self, x):
        return x[:, :, ::2, ::2] # downsamples the image by rejecting even rows and columns.

    def upsample(self, x):
        r"""it upsamples the source image by injecting even zero rows and columns and 
        then convolves the result with the same kernel as in downsample() multiplied by 4.
        https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=pyrdown#pyrup
        """
        # 以下面if __name__ == "__main__"的输入为例
        # -----------------------------------
        # inject even zero colums
        # x.shape=[1, 3, 132, 92] -> [1, 3, 132, 184]
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        # [1, 3, 132, 184] -> [1, 3, 264, 92]
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        # [1, 3, 92, 264]
        cc = cc.permute(0, 1, 3, 2)
        # ----------------------------------

        # ----------------------------------
        # inject even zero rows
        # cat([1, 3, 92, 264], [1, 3, 92, 264], dim=3) -> [1, 3, 92, 528]
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        # [1, 3, 184, 264]
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        # [1, 3, 264, 184]
        x_up = cc.permute(0, 1, 3, 2)
        # ----------------------------------

        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = F.pad(img, (2, 2, 2, 2), mode='reflect')
        # out = F.conv2d(img, kernel, groups=img.shape[1])
        # return out

        # https://www.pudn.com/news/6228cd129ddf223e1ad105c7.html
        return F.conv2d(img, kernel, groups=img.shape[1])

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel) # Blurs an image with a gaussian kernel
            down = self.downsample(filtered) # downsample the blurred image, i.e., Gaussian Pyramid
            up = self.upsample(down) # Upsamples the downsampled image and then blurs it
            # --------------------
            # if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            # --------------------
            if up.shape != current.shape:
                # -------------------
                # up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
                # -------------------
                up = F.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up # Laplacial Pyramid
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1] # pyr=[h_0^hat, h_1^hat, h_2^hat, I_3^hat]
        # print("***********************")
        # print(image.size())
        # print(len(pyr[:-1]))
        for level in reversed(pyr[:-1]):
            # https://www.jianshu.com/p/e7de4cd92f68
            up = self.upsample(image)
            # if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
            if up.shape != level.shape:
                # up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
                up = F.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = F.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_lh = F.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hl = F.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hh = F.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H//2, W//2)

            dx = dx.transpose(1,2).reshape(B, -1, H//2, W//2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = F.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None

class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None

class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)
        
        w_ll = rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float32)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)

class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1]) 
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


def bchw_to_blc(x: torch.Tensor) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, C, H, W) to (B, L, C)."""
    return x.flatten(2).transpose(1, 2)


def blc_to_bchw(x: torch.Tensor, x_size: Tuple) -> torch.Tensor:
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

# @ARCH_REGISTRY.register()
class FIR_LP(nn.Module):
    # original: embed_dim=12, depths=[2, 2, 2, 6, 2], num_heads=[3, 3, 6, 12, 12]
    def __init__(self, img_size=128, patch_size=1, in_chans=6, out_chans=3, num_high=3,
                 embed_dim=24, depths=[2, 2, 2, 6, 2], num_heads=[2, 4, 6, 8, 8],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, resi_connection='1conv',
                 **kwargs):
        super(FIR_LP, self).__init__()

        ################################### 1. Feature Extraction Network ###################################
        # coarse feature
        self.conv_f1 = nn.Conv2d(in_chans, embed_dim // 3, 3, 1, 1)
        self.conv_f2 = nn.Conv2d(in_chans, embed_dim // 3, 3, 1, 1)
        self.conv_f3 = nn.Conv2d(in_chans, embed_dim // 3, 3, 1, 1)
        self.conv_f2_end = nn.Conv2d(embed_dim // 3, embed_dim, 3, 1, 1)
        # # spatial attention module
        # self.att_module_l = SpatialAttentionModule(embed_dim)
        # self.att_module_h = SpatialAttentionModule(embed_dim)
        # self.conv_first = nn.Conv2d(embed_dim * 3, embed_dim, 3, 1, 1)

        ################################## 2. Coarse Feature Temporal Alignment #############################
        # self.tem_att = TAFusion(num_feat=embed_dim // 3)

        ################################### 3. HDR Reconstruction Network ##################################  
        self.num_layers = len(depths) # depths: Depth of each Swin Transformer layer
        self.embed_dim = embed_dim
        # self.ape = ape
        self.patch_norm = patch_norm
        # self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        # DWT
        # self.dwt = DWT_2D(wave='haar')
        # self.idwt = IDWT_2D(wave='haar')

        # LP
        self.num_high = num_high
        self.lap_pyramid = Lap_Pyramid_Conv(num_high=self.num_high, gauss_chans=self.embed_dim)

        # conv & channel attention
        self.channel_attention = nn.ModuleList()
        # for i_chanel_att in range(self.num_high+1):
        #     high_fre = RCAB(n_feat=2*embed_dim if i_chanel_att<self.num_high else embed_dim, reduction=2)
        #     self.channel_attention.append(high_fre)
        for i_chanel_att in range(self.num_high+1):
            high_fre = RCAB(n_feat=2*embed_dim if i_chanel_att<self.num_high else embed_dim, reduction=2)
            self.channel_attention.append(high_fre)

        self.up_sample = nn.ModuleList()
        for _ in range(self.num_high-1):
            res_up = Upsample(2*embed_dim)
            self.up_sample.append(res_up)

        self.reduce_channel = nn.ModuleList()
        for _ in range(self.num_high):
            res_down = ReduceChannel(2*embed_dim)
            self.reduce_channel.append(res_down)
        

        # self.reduce_chans = nn.ModuleList()
        # for _ in range(self.num_high):
        #     reduce_chans = nn.Conv2d(2*embed_dim, embed_dim, 1, 1)
        #     self.reduce_chans.append(reduce_chans)

        # self.conv_fuse = nn.Conv2d(2*embed_dim, embed_dim, 1, 1)
        # self.conv_x = 

        # split image into non-overlapping patches
        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        # patches_resolution = self.patch_embed.patches_resolution
        # self.patches_resolution = patches_resolution

        # # # merge non-overlapping patches into image
        # self.patch_unembed = PatchUnEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        # if self.ape:
        #     self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        #     trunc_normal_(self.absolute_pos_embed, std=.02)

        # Norm layer
        # self.norm_layer = [norm_layer(embed_dim * 2 ** i_layer) for i_layer in range(self.num_layers)]

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Context-aware Transformer Blocks (CTBs)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SpatialTemporalTransformerBlock(dim=int(2*embed_dim) if i_layer<self.num_layers-1 else embed_dim,
                         input_resolution=(img_size // (2 ** i_layer),
                                           img_size // (2 ** i_layer)),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers.append(layer)
        # self.norm = norm_layer(self.num_features)
        # self.wave_norm = norm_layer(4*self.num_features)

        # build the last conv layer
        if resi_connection == '1conv':
            # self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_after_body = nn.Conv2d(embed_dim, out_chans, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.normal_(0.0, 0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        # Gaussian and Laplacian Pyramid
        pyr = self.lap_pyramid.pyramid_decom(x) # 

        # level 3
        pyr_low = pyr[-1]
        pyr_low_size = (pyr_low.shape[2:])
        pyr_low = bchw_to_blc(self.channel_attention[-1](pyr_low))
        pyr_low = self.pos_drop(pyr_low)
        pyr_low = self.layers[-1](pyr_low, pyr_low_size)
        pyr_low = blc_to_bchw(pyr_low, pyr_low_size) # [4, 48, 8, 8]

        # print(pyr_low.shape)
        ## ----
        # pyr_low_up = F.interpolate(pyr_low, size=(pyr[-2].shape[2], pyr[-2].shape[3]))
        pyr_low_up = F.interpolate(pyr_low, scale_factor=2, mode='nearest')
        # -----
        # pyr_low_up = self.up_sample[-1](pyr_low)
        # print(pyr_low_up.shape)

        pyr_result = []
        for i in range(self.num_high):
            # print(pyr[self.num_high-(i+1)].shape)
            # print(pyr_low_up.shape)
            pyr_high_with_low = torch.cat([pyr[self.num_high-(i+1)], pyr_low_up], dim=1)
            # print(pyr[self.num_high-(i+1)].shape, pyr_low_up.shape)

            pyr_high_with_low_size = (pyr_high_with_low.shape[2:])
            pyr_high_with_low = bchw_to_blc(self.channel_attention[i](pyr_high_with_low))
            pyr_high_with_low = self.pos_drop(pyr_high_with_low)
            # print(self.num_high-(i+1))
            pyr_high_with_low = self.layers[self.num_high-(i+1)](pyr_high_with_low, pyr_high_with_low_size)
            result_highfreq = blc_to_bchw(pyr_high_with_low, pyr_high_with_low_size)

            if i < self.num_high-1:
                # pyr_low_up = F.interpolate(result_highfreq, size=pyr[self.num_high-(i+2)].shape[2:])
                # pyr_low_up = F.interpolate(result_highfreq, scale_factor=2, mode='nearest')
                pyr_low_up = self.up_sample[i](result_highfreq)
            # pre_high, pyr_low_up = self.up_sample[i](pyr_high_with_low)
            # pyr_result.append(pre_high)
            result_highfreq = self.reduce_channel[i](result_highfreq)
            # print(result_highfreq.shape)
            setattr(self, f'result_highfreq_{str(i)}', result_highfreq)
        
        for i in reversed(range(self.num_high)):
            result_highfreq = getattr(self, f'result_highfreq_{str(i)}')
            pyr_result.append(result_highfreq)

        pyr_result.append(pyr_low)

        pry_rec = self.lap_pyramid.pyramid_recons(pyr_result)

        return pry_rec

    def check_image_size(self, x1, x2, x3, H, W):

        # lcm = self.window_size * (2 ** self.layers) // math.gcd(self.window_size, 2 ** self.layers)
        # lcm = self.window_size * (np.power(2, self.levels)) // math.gcd(self.window_size, np.power(2, self.levels))
        lcm = np.power(2, 3) * self.window_size
        pad_input = (H % lcm != 0) or (W % lcm != 0)
        if pad_input:
            # to pad the last 3 dimensions
            # (W_left, W_right, H_top, H_bottom, C_front, C_back)
            mod_pad_h = (lcm - H % lcm) % lcm
            mod_pad_w = (lcm - W % lcm) % lcm
            x1 = F.pad(x1, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            x2 = F.pad(x2, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            x3 = F.pad(x3, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        
        return x1, x2, x3

    def forward(self, x1, x2, x3):

        H, W = x1.shape[2:]
        x1, x2, x3 = self.check_image_size(x1, x2, x3, H, W) # check image size 
        f1 = self.conv_f1(x1) # x1=[4, 6, 128, 128]
        f2 = self.conv_f2(x2)
        f3 = self.conv_f3(x3)
        # x = self.tem_att(torch.stack([f1, f2, f3], dim=1))
        x = torch.cat([f1, f2, f3], dim=1)

        # x = self.forward_features(x) + x
        x = self.forward_features(x) + self.conv_f2_end(f2)
        x = self.conv_after_body(x)

        x = torch.sigmoid(x)
        return x[:, :, :H, :W]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        return flops


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = 'cuda'
    model = FIR_LP(depths=[6, 6, 6, 6], num_heads=[4, 4, 4, 4], embed_dim=48).to(device)
    height, width = 128, 128
    x1 = torch.randn((1, 6, height, width)).to(device)
    x2 = torch.randn((1, 6, height, width)).to(device)
    x3 = torch.randn((1, 6, height, width)).to(device)
    result = model(x1, x2, x3)
    print(result.shape)
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # for _ in range(5):
    #     with torch.no_grad():
    #         x_sr = model(x1, x2, x3)
    # start.record()
    # with torch.no_grad():
    #     x = model(x1, x2, x3)
    # end.record()
    # torch.cuda.synchronize()
    # print('time cost: {:.2f}s'.format(start.elapsed_time(end)/(1000.)))
    # # print(x.shape)

    # msg = 'Params:{:.2f}M'.format(sum(map(lambda x: x.numel(), model.parameters()))/(1024)**2)
    # print(msg)