"""
MambaIRv2 - Attentive State Space Restoration blocks.
Efficient Mamba-based architecture for image and video restoration.
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
        
        # Attention for adaptive selection
        self.attention = nn.MultiheadAttention(d_inner, num_heads=8, batch_first=True)
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
        
        # Self-attention for adaptive processing
        x_attended, _ = self.attention(x_flat, x_flat, x_flat)
        x_attended = self.norm_attn(x_attended)
        x_ssm = x_flat + x_attended  # Residual
        
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