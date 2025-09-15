"""
Diffusion Video UNet for video restoration tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Tuple

class TimestepEmbedding(nn.Module):
    """Timestep embedding for diffusion process."""
    
    def __init__(self, channels, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(time_embed_dim, time_embed_dim)
    
    def forward(self, t):
        t_emb = self.linear1(t)
        t_emb = self.act(t_emb)
        t_emb = self.linear2(t_emb)
        return t_emb

class ResBlock3D(nn.Module):
    """3D ResNet block for video processing."""
    
    def __init__(self, in_channels, out_channels, time_embed_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_emb_proj = nn.Linear(time_embed_dim, out_channels)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.skip_connection = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x, time_emb):
        """Forward pass with timestep conditioning."""
        h = x
        
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add timestep embedding
        time_emb = self.time_emb_proj(time_emb)
        # Reshape to match spatial dimensions
        while len(time_emb.shape) < len(h.shape):
            time_emb = time_emb[..., None]
        h = h + time_emb
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip_connection(x)

class AttentionBlock3D(nn.Module):
    """3D attention block for video processing."""
    
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1)
        self.out_proj = nn.Conv3d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        """Apply 3D attention across spatial and temporal dimensions."""
        B, C, T, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        
        # Reshape for attention computation
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, T * H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, T*H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        h = torch.matmul(attn_weights, v)
        h = h.permute(1, 2, 0, 3).reshape(B, C, T, H, W)
        
        h = self.out_proj(h)
        return x + h

class DiffusionVideoUNet(nn.Module):
    """
    Video UNet for diffusion-based restoration.
    """
    
    def __init__(self, 
                 in_channels=6,  # Input + noise
                 out_channels=3,
                 model_channels=128,
                 num_res_blocks=2,
                 attention_resolutions=[16, 8],
                 dropout=0.0,
                 channel_mult=(1, 2, 4, 8),
                 num_heads=8,
                 num_frames=8):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.num_frames = num_frames
        
        # Timestep embedding
        time_embed_dim = model_channels * 4
        self.time_embed = TimestepEmbedding(model_channels, time_embed_dim)
        
        # Input projection
        self.input_blocks = nn.ModuleList([
            nn.Conv3d(in_channels, model_channels, kernel_size=3, padding=1)
        ])
        
        # Downsampling path
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock3D(ch, mult * model_channels, time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(AttentionBlock3D(ch, num_heads))
                
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    nn.Conv3d(ch, ch, kernel_size=3, stride=2, padding=1)
                )
                input_block_chans.append(ch)
                ds *= 2
        
        # Middle block
        self.middle_block = nn.Sequential(
            ResBlock3D(ch, ch, time_embed_dim, dropout),
            AttentionBlock3D(ch, num_heads),
            ResBlock3D(ch, ch, time_embed_dim, dropout)
        )
        
        # Upsampling path
        self.output_blocks = nn.ModuleList([])
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock3D(ch + ich, mult * model_channels, time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(AttentionBlock3D(ch, num_heads))
                
                if level and i == num_res_blocks:
                    layers.append(
                        nn.ConvTranspose3d(ch, ch, kernel_size=4, stride=2, padding=1)
                    )
                    ds //= 2
                
                self.output_blocks.append(nn.Sequential(*layers))
        
        # Output projection
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv3d(ch, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, timestep, context=None):
        """
        Forward pass of the Video UNet.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
            timestep: Timestep tensor of shape (B,)
            context: Optional context conditioning
            
        Returns:
            Output tensor of shape (B, out_channels, T, H, W)
        """
        # Timestep embedding
        t_emb = self.time_embed(self._get_timestep_embedding(timestep, self.model_channels))
        
        # Downsampling
        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.Sequential) and len(module) > 0:
                if isinstance(module[0], ResBlock3D):
                    h = module[0](h, t_emb)
                    if len(module) > 1:  # Has attention
                        h = module[1](h)
                else:
                    h = module(h)
            else:
                h = module(h)
            hs.append(h)
        
        # Middle processing
        for module in self.middle_block:
            if isinstance(module, ResBlock3D):
                h = module(h, t_emb)
            else:
                h = module(h)
        
        # Upsampling
        for module in self.output_blocks:
            if isinstance(module, nn.Sequential) and len(module) > 0:
                h = torch.cat([h, hs.pop()], dim=1)
                if isinstance(module[0], ResBlock3D):
                    h = module[0](h, t_emb)
                    # Apply remaining modules (attention, upsample)
                    for submodule in module[1:]:
                        h = submodule(h)
                else:
                    h = module(h)
            else:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h)
        
        # Output
        return self.out(h)
    
    def _get_timestep_embedding(self, timesteps, embedding_dim):
        """Generate sinusoidal timestep embeddings."""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, [0, 1])
        
        return emb