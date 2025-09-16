"""
RVRT Network Implementation
Recurrent Video Restoration Transformer for video enhancement fallback.
"""

"""
MIT License

Copyright (c) 2024 Video Enhancement Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

@dataclass
class RVRTConfig:
    """Configuration for RVRT network."""
    embed_dim: int = 180
    depths: Tuple[int, ...] = (6, 6, 6)
    num_heads: Tuple[int, ...] = (6, 6, 6)
    window_size: Tuple[int, int] = (8, 8)
    mlp_ratio: float = 2.0
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.2
    norm_layer: nn.Module = nn.LayerNorm
    patch_norm: bool = True
    frozen_stages: int = -1
    use_checkpoint: bool = False
    resi_connection: str = '1conv'
    upscale: int = 4
    img_size: int = 64
    patch_size: int = 1
    in_chans: int = 3
    num_feat: int = 64
    num_frame: int = 7


class WindowAttention3D(nn.Module):
    """3D Window-based Multi-Head Self Attention."""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, 
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # Define relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # Get pair-wise relative position index for each token inside the window
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
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
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


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block for video processing."""
    
    def __init__(self, dim, num_heads, window_size=(8, 8), shift_size=(0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = self.window_size, self.shift_size
        
        x = self.norm1(x)
        
        # Pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_d0 = pad_d1 = 0
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        
        # Cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
            
        # Partition windows
        x_windows = self.window_partition(shifted_x, window_size)
        x_windows = x_windows.view(-1, window_size[0] * window_size[1], C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, window_size[0], window_size[1], C)
        shifted_x = self.window_reverse(attn_windows, window_size, Dp, Hp, Wp)
        
        # Reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(2, 3))
        else:
            x = shifted_x
            
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
            
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x
    
    def window_partition(self, x, window_size):
        """Partition into non-overlapping windows."""
        B, D, H, W, C = x.shape
        x = x.view(B, D, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        windows = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(-1, window_size[0], window_size[1], C)
        return windows
    
    def window_reverse(self, windows, window_size, D, H, W):
        """Reverse window partitioning."""
        B = int(windows.shape[0] / (D * H * W / window_size[0] / window_size[1]))
        x = windows.view(B, D, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(B, D, H, W, -1)
        return x


class RVRT_BasicLayer(nn.Module):
    """Basic layer for RVRT consisting of multiple Swin Transformer blocks."""
    
    def __init__(self, dim, depth, num_heads, window_size=(8, 8),
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])
            
    def forward(self, x):
        # Calculate attention mask for SW-MSA
        Dp = int(np.ceil(x.size(1) / self.window_size[0])) * self.window_size[0]
        Hp = int(np.ceil(x.size(2) / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(x.size(3) / self.window_size[1])) * self.window_size[1]
        attn_mask = self.compute_mask(Dp, Hp, Wp, x.device)
        
        for blk in self.blocks:
            x = blk(x, attn_mask)
        return x
    
    def compute_mask(self, D, H, W, device):
        """Compute attention mask for SW-MSA."""
        img_mask = torch.zeros((1, D, H, W, 1), device=device)
        cnt = 0
        
        for d in slice(-self.window_size[0]), slice(-self.shift_size[0], -self.window_size[0]), slice(-self.shift_size[0], None):
            for h in slice(-self.window_size[0]), slice(-self.shift_size[0], -self.window_size[0]), slice(-self.shift_size[0], None):
                for w in slice(-self.window_size[1]), slice(-self.shift_size[1], -self.window_size[1]), slice(-self.shift_size[1], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
                    
        mask_windows = self.window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        return attn_mask
        
    def window_partition(self, x, window_size):
        """Partition into non-overlapping windows."""
        B, D, H, W, C = x.shape
        x = x.view(B, D, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        windows = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(-1, window_size[0], window_size[1], C)
        return windows


class RVRTNetwork(nn.Module):
    """RVRT Network for Video Restoration."""
    
    def __init__(self, config: RVRTConfig):
        super().__init__()
        self.config = config
        self.num_layers = len(config.depths)
        self.embed_dim = config.embed_dim
        self.patch_norm = config.patch_norm
        self.num_features = config.embed_dim
        self.mlp_ratio = config.mlp_ratio
        
        # Split image into non-overlapping patches
        self.patch_embed = nn.Conv3d(config.in_chans, config.embed_dim, 
                                   kernel_size=(1, config.patch_size, config.patch_size),
                                   stride=(1, config.patch_size, config.patch_size))
        if self.patch_norm:
            self.norm = config.norm_layer(config.embed_dim)
            
        self.pos_drop = nn.Dropout(p=config.drop_rate)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RVRT_BasicLayer(
                dim=int(config.embed_dim),
                depth=config.depths[i_layer],
                num_heads=config.num_heads[i_layer],
                window_size=config.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=config.qkv_bias,
                qk_scale=config.qk_scale,
                drop=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=dpr[sum(config.depths[:i_layer]):sum(config.depths[:i_layer + 1])],
                norm_layer=config.norm_layer,
                use_checkpoint=config.use_checkpoint)
            self.layers.append(layer)
            
        self.norm = config.norm_layer(self.num_features)
        
        # Reconstruction layers
        self.conv_before_upsample = nn.Sequential(
            nn.Conv3d(config.embed_dim, config.num_feat, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.LeakyReLU(inplace=True))
            
        self.upsample = nn.Sequential(
            nn.Conv3d(config.num_feat, config.num_feat * 4, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(config.num_feat, config.num_feat * 4, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(inplace=True))
            
        self.conv_last = nn.Conv3d(config.num_feat, config.in_chans, (1, 3, 3), (1, 1, 1), (0, 1, 1))
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        """Forward pass for RVRT network.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
            
        Returns:
            Enhanced tensor of same shape
        """
        B, C, T, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, T, H, W)
        
        if self.patch_norm:
            x = x.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
            x = self.norm(x)
            x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
            
        x = self.pos_drop(x)
        
        # Transform to layer input format (B, T, H, W, C)
        x = x.permute(0, 2, 3, 4, 1)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        
        # Transform back to conv format (B, C, T, H, W)
        x = x.permute(0, 4, 1, 2, 3)
        
        # Reconstruction
        x = self.conv_before_upsample(x)
        x = self.upsample(x)
        x = self.conv_last(x)
        
        return x


import numpy as np