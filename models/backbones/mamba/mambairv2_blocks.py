"""
MambaIRv2 - Attentive State Space Restoration blocks.
Efficient Mamba-based architecture for image and video restoration.
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

class AttentiveSSM(nn.Module):
    """Attentive State Space Model with improved restoration capabilities."""
    
    def __init__(self, dim, d_state=64, d_conv=3, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        d_inner = dim * expand
        
        # Input/output projections
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=False)
        self.out_proj = nn.Linear(d_inner, dim, bias=False)
        
        # Convolution for local context
        self.conv = nn.Conv2d(d_inner, d_inner, 
                             kernel_size=d_conv, 
                             padding=d_conv//2, 
                             groups=d_inner)
        
        # State space parameters
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
        self.D = nn.Parameter(torch.randn(d_inner))
        
        # Attention for adaptive selection with dimension compatibility
        # Ensure embed_dim is divisible by num_heads
        self.embed_dim = max(64, ((d_inner + 7) // 8) * 8)  # Round up to nearest multiple of 8
        self.num_heads = min(8, self.embed_dim // 8)  # Ensure at least 8 dims per head
        
        # Dimension projection if needed
        if d_inner != self.embed_dim:
            self.attn_proj = nn.Linear(d_inner, self.embed_dim)
            self.attn_unproj = nn.Linear(self.embed_dim, d_inner)
        else:
            self.attn_proj = nn.Identity()
            self.attn_unproj = nn.Identity()
        
        self.attention = nn.MultiheadAttention(self.embed_dim, num_heads=self.num_heads, batch_first=True)
        self.norm_attn = nn.LayerNorm(d_inner)
    
    def forward(self, x):
        """Forward with attentive state space processing."""
        B, H, W, C = x.shape
        
        # Project and split
        xz = self.in_proj(x)  # (B, H, W, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)
        
        # Apply convolution
        x_ssm = x_ssm.permute(0, 3, 1, 2)  # (B, d_inner, H, W)
        x_ssm = self.conv(x_ssm)
        x_ssm = x_ssm.permute(0, 2, 3, 1)  # (B, H, W, d_inner)
        
        # Flatten for attention
        x_flat = x_ssm.view(B, H*W, -1)
        
        # Self-attention for adaptive processing with dimension projection
        # Project to compatible dimension for attention
        x_proj = self.attn_proj(x_flat)
        x_attended, _ = self.attention(x_proj, x_proj, x_proj)
        # Project back and apply layer norm
        x_attended_back = self.attn_unproj(x_attended)
        x_attended_norm = self.norm_attn(x_attended_back)
        x_ssm = x_flat + x_attended_norm  # Residual
        
        # Reshape back
        x_ssm = x_ssm.view(B, H, W, -1)
        
        # Apply gating
        x_ssm = F.silu(x_ssm) * F.silu(z)
        
        # Output projection
        output = self.out_proj(x_ssm)
        return output

class MambaIRv2Block(nn.Module):
    """MambaIRv2 block for image/video restoration."""
    
    def __init__(self, dim, d_state=64, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ssm = AttentiveSSM(dim, d_state)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # SSM block
        residual = x
        x = self.norm1(x)
        x = self.ssm(x)
        x = residual + x
        
        # MLP block
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x