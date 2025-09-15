"""
Video Transformer for video restoration tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MultiheadSelfAttention(nn.Module):
    """Multi-head self attention with optional positional bias."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=True, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (B, N, C)
            mask: Optional attention mask
        """
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation."""
    
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and MLP."""
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0, layer_scale_init=None):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadSelfAttention(dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)
        
        # Layer scale for better training stability
        self.layer_scale1 = None
        self.layer_scale2 = None
        if layer_scale_init is not None:
            self.layer_scale1 = nn.Parameter(layer_scale_init * torch.ones(dim))
            self.layer_scale2 = nn.Parameter(layer_scale_init * torch.ones(dim))
    
    def forward(self, x, mask=None):
        # Self-attention
        if self.layer_scale1 is not None:
            x = x + self.layer_scale1 * self.attn(self.norm1(x), mask)
        else:
            x = x + self.attn(self.norm1(x), mask)
            
        # MLP
        if self.layer_scale2 is not None:
            x = x + self.layer_scale2 * self.mlp(self.norm2(x))
        else:
            x = x + self.mlp(self.norm2(x))
            
        return x

class VideoTransformer(nn.Module):
    """
    Video Transformer for processing video sequences.
    
    Supports both spatial-temporal joint processing and separate processing modes.
    """
    
    def __init__(self, 
                 dim=768,
                 depth=12,
                 heads=12,
                 mlp_dim=None,
                 dropout=0.1,
                 layer_scale_init=1e-6):
        super().__init__()
        
        self.dim = dim
        self.depth = depth
        self.heads = heads
        mlp_dim = mlp_dim or 4 * dim
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=heads,
                mlp_ratio=mlp_dim / dim,
                dropout=dropout,
                layer_scale_init=layer_scale_init
            ) for _ in range(depth)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, mask=None):
        """
        Forward pass of Video Transformer.
        
        Args:
            x: Input tensor of shape (B, N, C) where N = T*H*W for video patches
            mask: Optional attention mask of shape (B, N, N)
            
        Returns:
            Output tensor of shape (B, N, C)
        """
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm
        x = self.norm(x)
        
        return x
    
    def forward_spatial_temporal(self, x, num_frames, spatial_size):
        """
        Process video with explicit spatial-temporal separation.
        
        Args:
            x: Input tensor of shape (B, T*H*W, C)
            num_frames: Number of frames T
            spatial_size: Tuple of (H, W) for spatial dimensions
            
        Returns:
            Output tensor of shape (B, T*H*W, C)
        """
        B, N, C = x.shape
        H, W = spatial_size
        
        # Reshape to separate spatial and temporal dimensions
        x = x.view(B, num_frames, H * W, C)
        
        # Process each frame spatially
        spatial_outputs = []
        for t in range(num_frames):
            frame_tokens = x[:, t, :, :]  # (B, H*W, C)
            
            # Apply some transformer blocks for spatial processing
            for i, block in enumerate(self.blocks[:self.depth//2]):
                frame_tokens = block(frame_tokens)
            
            spatial_outputs.append(frame_tokens)
        
        # Stack frames back
        x_spatial = torch.stack(spatial_outputs, dim=1)  # (B, T, H*W, C)
        
        # Process temporal relationships
        # Reshape to (B*H*W, T, C) for temporal processing
        x_temporal = x_spatial.permute(0, 2, 1, 3).reshape(B * H * W, num_frames, C)
        
        # Apply remaining blocks for temporal processing
        for block in self.blocks[self.depth//2:]:
            x_temporal = block(x_temporal)
        
        # Reshape back to original format
        x_temporal = x_temporal.reshape(B, H * W, num_frames, C).permute(0, 2, 1, 3)
        x_output = x_temporal.reshape(B, num_frames * H * W, C)
        
        return self.norm(x_output)

class WindowAttention(nn.Module):
    """Window-based multi-head self attention for efficient video processing."""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wt, Wh, Ww)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # Define relative position bias table
        coords_t = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_t, coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self, x):
        """
        Args:
            x: Input features with shape (num_windows*B, Wt*Wh*Ww, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x