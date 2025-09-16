"""
3D Patch Embedding for video transformers.
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
from typing import Tuple

class PatchEmbedding3D(nn.Module):
    """
    3D Patch Embedding for video data.
    
    Converts video patches into token embeddings for transformer processing.
    """
    
    def __init__(self, 
                 patch_size=(2, 4, 4),  # (T, H, W)
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=None,
                 flatten=True):
        super().__init__()
        
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten = flatten
        
        # 3D convolution for patch embedding
        self.proj = nn.Conv3d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        self.norm = norm_layer(embed_dim) if norm_layer else None
        
    def forward(self, x):
        """
        Forward pass of 3D patch embedding.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
            
        Returns:
            Embedded patches of shape (B, N, embed_dim) if flatten=True,
            else (B, embed_dim, Tp, Hp, Wp)
        """
        B, C, T, H, W = x.shape
        
        # Check if input dimensions are divisible by patch size
        assert T % self.patch_size[0] == 0, f"Video depth {T} not divisible by patch size {self.patch_size[0]}"
        assert H % self.patch_size[1] == 0, f"Video height {H} not divisible by patch size {self.patch_size[1]}"
        assert W % self.patch_size[2] == 0, f"Video width {W} not divisible by patch size {self.patch_size[2]}"
        
        # Apply 3D convolution to create patches
        x = self.proj(x)  # (B, embed_dim, Tp, Hp, Wp)
        
        if self.flatten:
            # Flatten spatial and temporal dimensions
            x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim) where N = Tp*Hp*Wp
        
        if self.norm is not None:
            x = self.norm(x)
            
        return x
    
    def get_num_patches(self, input_shape):
        """
        Get number of patches for given input shape.
        
        Args:
            input_shape: Tuple of (T, H, W)
            
        Returns:
            Number of patches
        """
        T, H, W = input_shape
        Tp = T // self.patch_size[0]
        Hp = H // self.patch_size[1]
        Wp = W // self.patch_size[2]
        return Tp * Hp * Wp

class AdaptivePatchEmbedding3D(nn.Module):
    """
    Adaptive 3D Patch Embedding that can handle variable patch sizes.
    """
    
    def __init__(self,
                 min_patch_size=(1, 2, 2),
                 max_patch_size=(4, 8, 8),
                 in_chans=3,
                 embed_dim=768,
                 num_scales=3):
        super().__init__()
        
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_scales = num_scales
        
        # Create multiple patch embedding layers for different scales
        self.embeddings = nn.ModuleList()
        
        for i in range(num_scales):
            # Interpolate patch sizes
            t_size = int(min_patch_size[0] + i * (max_patch_size[0] - min_patch_size[0]) / (num_scales - 1))
            h_size = int(min_patch_size[1] + i * (max_patch_size[1] - min_patch_size[1]) / (num_scales - 1))
            w_size = int(min_patch_size[2] + i * (max_patch_size[2] - min_patch_size[2]) / (num_scales - 1))
            
            patch_size = (t_size, h_size, w_size)
            
            embedding = PatchEmbedding3D(
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                flatten=True
            )
            
            self.embeddings.append(embedding)
        
        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
    def forward(self, x, scale_idx=None):
        """
        Forward pass with adaptive patch sizes.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
            scale_idx: Optional scale index. If None, uses weighted combination
            
        Returns:
            Embedded patches
        """
        if scale_idx is not None:
            # Use specific scale
            return self.embeddings[scale_idx](x)
        
        # Use weighted combination of all scales
        outputs = []
        valid_scales = []
        
        B, C, T, H, W = x.shape
        
        for i, embedding in enumerate(self.embeddings):
            patch_size = embedding.patch_size
            
            # Check if this scale is valid for current input
            if (T % patch_size[0] == 0 and 
                H % patch_size[1] == 0 and 
                W % patch_size[2] == 0):
                
                output = embedding(x)
                outputs.append(output)
                valid_scales.append(i)
        
        if not outputs:
            # Fallback to first embedding with padding if needed
            return self.embeddings[0](self._pad_to_patch_size(x, self.embeddings[0].patch_size))
        
        # Weight the outputs
        weighted_output = torch.zeros_like(outputs[0])
        total_weight = 0
        
        for i, (output, scale_idx) in enumerate(zip(outputs, valid_scales)):
            weight = F.softmax(self.scale_weights, dim=0)[scale_idx]
            weighted_output += weight * output
            total_weight += weight
        
        return weighted_output / total_weight if total_weight > 0 else outputs[0]
    
    def _pad_to_patch_size(self, x, patch_size):
        """Pad input to be divisible by patch size."""
        B, C, T, H, W = x.shape
        
        # Calculate padding needed
        pad_t = (patch_size[0] - T % patch_size[0]) % patch_size[0]
        pad_h = (patch_size[1] - H % patch_size[1]) % patch_size[1]
        pad_w = (patch_size[2] - W % patch_size[2]) % patch_size[2]
        
        # Apply padding
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode='replicate')
        
        return x

class SinusoidalPositionEmbedding3D(nn.Module):
    """
    Sinusoidal 3D positional embedding for video transformers.
    """
    
    def __init__(self, embed_dim, max_len=(100, 64, 64)):  # (T, H, W)
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        # Create 3D positional encoding
        pe = torch.zeros(max_len[0] * max_len[1] * max_len[2], embed_dim)
        
        # Temporal positions
        t_pos = torch.arange(0, max_len[0]).float().unsqueeze(1).repeat(1, max_len[1] * max_len[2])
        t_pos = t_pos.flatten().unsqueeze(1)
        
        # Spatial positions
        h_pos = torch.arange(0, max_len[1]).float().unsqueeze(1).repeat(max_len[0], max_len[2])
        h_pos = h_pos.flatten().unsqueeze(1)
        
        w_pos = torch.arange(0, max_len[2]).float().repeat(max_len[0] * max_len[1])
        w_pos = w_pos.unsqueeze(1)
        
        # Create sinusoidal encodings
        div_term = torch.exp(torch.arange(0, embed_dim // 3, 2).float() *
                           -(torch.log(torch.tensor(10000.0)) / (embed_dim // 3)))
        
        # Temporal encoding
        pe[:, 0:embed_dim//3:2] = torch.sin(t_pos * div_term)
        pe[:, 1:embed_dim//3:2] = torch.cos(t_pos * div_term)
        
        # Height encoding
        pe[:, embed_dim//3:2*embed_dim//3:2] = torch.sin(h_pos * div_term)
        pe[:, embed_dim//3+1:2*embed_dim//3:2] = torch.cos(h_pos * div_term)
        
        # Width encoding
        pe[:, 2*embed_dim//3::2] = torch.sin(w_pos * div_term)
        pe[:, 2*embed_dim//3+1::2] = torch.cos(w_pos * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x, shape_3d):
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (B, N, embed_dim)
            shape_3d: Tuple of (T, H, W) for original video dimensions
            
        Returns:
            Tensor with positional encoding added
        """
        T, H, W = shape_3d
        seq_len = T * H * W
        
        if seq_len <= self.pe.size(1):
            x = x + self.pe[:, :seq_len, :]
        else:
            # Handle sequences longer than pre-computed embeddings
            # Simple approach: repeat the pattern
            repeat_factor = (seq_len // self.pe.size(1)) + 1
            extended_pe = self.pe.repeat(1, repeat_factor, 1)
            x = x + extended_pe[:, :seq_len, :]
        
        return x