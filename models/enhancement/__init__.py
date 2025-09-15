"""
Video Enhancement Models

This module provides access to all video enhancement handlers including:
- Legacy handlers (for compatibility)
- Modern handlers (enhanced versions)
- 2025 SOTA handlers (latest state-of-the-art)
"""

# Legacy handlers (being phased out)
try:
    from .rife_handler import RIFEHandler
except ImportError:
    RIFEHandler = None
    
try:
    from .esrgan_handler import ESRGANHandler
except ImportError:
    ESRGANHandler = None

# Modern handlers
try:
    from .ftvsr_handler import FTVSRHandler
except ImportError:
    FTVSRHandler = None
    
try:
    from .enhanced_rife_handler import EnhancedRIFEHandler
except ImportError:
    EnhancedRIFEHandler = None

# 2025 SOTA handlers - Video Super-Resolution
try:
    from .vsr.vsrm_handler import VSRMHandler
except ImportError:
    VSRMHandler = None
    
try:
    from .vsr.fast_mamba_vsr_handler import FastMambaVSRHandler
except ImportError:
    FastMambaVSRHandler = None

# 2025 SOTA handlers - Diffusion-based
try:
    from .diffusion.seedvr2_handler import SeedVR2Handler
except ImportError:
    SeedVR2Handler = None

# 2025 SOTA handlers - Zero-shot
try:
    from .zeroshot.ditvr_handler import DiTVRHandler
except ImportError:
    DiTVRHandler = None

# Available handlers (only non-None)
__all__ = [name for name in [
    # Legacy
    'RIFEHandler',
    'ESRGANHandler', 
    # Modern
    'FTVSRHandler',
    'EnhancedRIFEHandler',
    # 2025 SOTA
    'VSRMHandler',
    'FastMambaVSRHandler',
    'SeedVR2Handler',
    'DiTVRHandler'
] if globals().get(name) is not None]

# Handler categories for easy organization
SOTA_2025_HANDLERS = {
    'vsr': ['VSRMHandler', 'FastMambaVSRHandler'],
    'diffusion': ['SeedVR2Handler'],
    'zeroshot': ['DiTVRHandler']
}

MODERN_HANDLERS = ['FTVSRHandler', 'EnhancedRIFEHandler']
LEGACY_HANDLERS = ['RIFEHandler', 'ESRGANHandler']

def get_available_handlers():
    """Get list of all available enhancement handlers."""
    return [name for name in __all__ if globals().get(name) is not None]

def get_sota_handlers():
    """Get list of 2025 SOTA enhancement handlers."""
    sota_list = []
    for category in SOTA_2025_HANDLERS.values():
        sota_list.extend([name for name in category if globals().get(name) is not None])
    return sota_list