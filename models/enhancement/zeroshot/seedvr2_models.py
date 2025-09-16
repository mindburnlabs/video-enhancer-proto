"""
SeedVR2 Model Wrappers
Convenient wrapper classes for SeedVR2-3B and SeedVR2-7B models.
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


from typing import Optional, Dict, Tuple
from .ditvr_handler import DiTVRHandler

class SeedVR2_3B(DiTVRHandler):
    """SeedVR2-3B: Latest 2025 model with balanced performance and efficiency."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = "cuda",
                 num_frames: int = 16,
                 patch_size: Tuple[int, int, int] = (2, 4, 4),
                 tile_size: int = 224,
                 tile_overlap: int = 32,
                 auto_download: bool = True):
        """Initialize SeedVR2-3B model.
        
        Args:
            model_path: Optional path to model weights
            device: Device to run on ('cuda' or 'cpu')
            num_frames: Number of frames to process together
            patch_size: 3D patch size (T, H, W)
            tile_size: Size of tiles for large frame processing
            tile_overlap: Overlap between tiles
            auto_download: Automatically download weights from HuggingFace
        """
        super().__init__(
            model_path=model_path,
            device=device,
            model_size="3B",
            num_frames=num_frames,
            patch_size=patch_size,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            auto_download=auto_download
        )

class SeedVR2_7B(DiTVRHandler):
    """SeedVR2-7B: Latest 2025 model with maximum quality and capability."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = "cuda",
                 num_frames: int = 16,
                 patch_size: Tuple[int, int, int] = (2, 4, 4),
                 tile_size: int = 224,
                 tile_overlap: int = 32,
                 auto_download: bool = True):
        """Initialize SeedVR2-7B model.
        
        Args:
            model_path: Optional path to model weights
            device: Device to run on ('cuda' or 'cpu')
            num_frames: Number of frames to process together
            patch_size: 3D patch size (T, H, W)
            tile_size: Size of tiles for large frame processing
            tile_overlap: Overlap between tiles
            auto_download: Automatically download weights from HuggingFace
        """
        super().__init__(
            model_path=model_path,
            device=device,
            model_size="7B",
            num_frames=num_frames,
            patch_size=patch_size,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            auto_download=auto_download
        )

# Convenience functions for quick instantiation
def create_seedvr2_3b(device="cuda", auto_download=True, **kwargs):
    """Create SeedVR2-3B model with default settings."""
    return SeedVR2_3B(device=device, auto_download=auto_download, **kwargs)

def create_seedvr2_7b(device="cuda", auto_download=True, **kwargs):
    """Create SeedVR2-7B model with default settings."""
    return SeedVR2_7B(device=device, auto_download=auto_download, **kwargs)