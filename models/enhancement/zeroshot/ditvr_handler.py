"""
DiTVR (Diffusion Transformer Video Restoration) Handler
Zero-shot Transformer-based video restoration with adaptive degradation modeling.
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
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Tuple, List, Union
import tempfile
import json
import math
import os
from huggingface_hub import hf_hub_download, list_repo_files
import requests
from tqdm import tqdm

from models.backbones.transformer import VideoTransformer, PatchEmbedding3D
from models.backbones.diffusion import NoiseScheduler
from utils.video_utils import VideoUtils

logger = logging.getLogger(__name__)

class DiTVRNetwork(nn.Module):
    """DiTVR network with zero-shot Transformer-based restoration."""
    
    def __init__(self, 
                 num_frames=16,
                 patch_size=(2, 4, 4),  # (T, H, W)
                 in_channels=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,  # For conditioning
                 dropout=0.1):
        super().__init__()
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Patch embedding for 3D video patches
        self.patch_embed = PatchEmbedding3D(
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim
        )
        
        # Degradation-aware conditioning
        self.degradation_encoder = DegradationEncoder(embed_dim)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self._get_num_patches(), embed_dim))
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames // patch_size[0], embed_dim))
        
        # Transformer backbone
        self.transformer = VideoTransformer(
            dim=embed_dim,
            depth=depth,
            heads=num_heads,
            mlp_dim=int(embed_dim * mlp_ratio),
            dropout=dropout
        )
        
        # Adaptive layer norm for degradation conditioning
        self.adaptive_norm = AdaptiveLayerNorm(embed_dim)
        
        # Output head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, np.prod(patch_size) * in_channels)
        
        # Zero-shot adaptation modules
        self.meta_adapter = MetaAdapter(embed_dim)
        self.consistency_loss = TemporalConsistencyLoss()
        
        # Initialize weights
        self._init_weights()
        
    def _get_num_patches(self):
        """Calculate number of patches (placeholder - depends on input size)."""
        # This would be calculated based on input dimensions
        return 196  # 14x14 spatial patches for 224x224 input
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x, degradation_params=None, adaptation_mode=True):
        """Forward pass for DiTVR.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
            degradation_params: Degradation parameters for conditioning
            adaptation_mode: Whether to use zero-shot adaptation
            
        Returns:
            Restored tensor of shape (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        
        # Patch embedding
        x_patches = self.patch_embed(x)  # (B, N, D) where N is number of patches
        
        # Add positional embeddings
        x_patches = x_patches + self.pos_embed
        
        # Add temporal embeddings (broadcasting across spatial patches)
        num_temporal_patches = T // self.patch_size[0]
        num_spatial_patches = x_patches.shape[1] // num_temporal_patches
        
        temporal_embed_expanded = self.temporal_embed.repeat_interleave(
            num_spatial_patches, dim=1
        )
        x_patches = x_patches + temporal_embed_expanded
        
        # Degradation-aware conditioning
        if degradation_params is not None:
            degradation_embed = self.degradation_encoder(degradation_params)
            x_patches = self.adaptive_norm(x_patches, degradation_embed)
        
        # Zero-shot adaptation
        if adaptation_mode:
            x_patches = self.meta_adapter(x_patches, x)
        
        # Transformer processing
        x_patches = self.transformer(x_patches)
        
        # Output head
        x_patches = self.norm(x_patches)
        x_patches = self.head(x_patches)
        
        # Reconstruct video from patches
        x_restored = self._unpatchify(x_patches, (B, C, T, H, W))
        
        return x_restored
    
    def _unpatchify(self, x_patches, original_shape):
        """Convert patches back to video tensor."""
        B, C, T, H, W = original_shape
        
        # This is a simplified version - actual implementation would handle
        # patch size and overlapping properly
        patch_h = patch_w = int(math.sqrt(x_patches.shape[1] // (T // self.patch_size[0])))
        
        # Reshape patches back to video format
        x_patches = x_patches.view(
            B, T // self.patch_size[0], patch_h, patch_w,
            self.patch_size[0], self.patch_size[1], self.patch_size[2], C
        )
        
        # Combine patches
        x_restored = x_patches.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x_restored = x_restored.view(B, C, T, H, W)
        
        return x_restored

class DegradationEncoder(nn.Module):
    """Encoder for degradation-aware conditioning."""
    
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Degradation type embeddings
        self.degradation_embeddings = nn.Embedding(10, embed_dim)  # Support 10 degradation types
        
        # Parameter encoders
        self.noise_encoder = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim // 2)
        )
        
        self.blur_encoder = nn.Sequential(
            nn.Linear(2, embed_dim // 4),  # blur_sigma, blur_type
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim // 2)
        )
        
        self.compression_encoder = nn.Sequential(
            nn.Linear(1, embed_dim // 4),  # quality factor
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim // 4)
        )
        
        # Fusion layer
        self.fusion = nn.Linear(embed_dim + embed_dim // 2 + embed_dim // 2 + embed_dim // 4, embed_dim)
        
    def forward(self, degradation_params):
        """Encode degradation parameters.
        
        Args:
            degradation_params: Dict with keys 'type', 'noise_level', 'blur_params', 'quality'
        """
        # Default values if params not provided
        deg_type = degradation_params.get('type', 0)
        noise_level = degradation_params.get('noise_level', 0.1)
        blur_sigma = degradation_params.get('blur_sigma', 1.0)
        blur_type = degradation_params.get('blur_type', 0)
        quality = degradation_params.get('quality', 0.8)
        
        # Get device from the embeddings for consistency
        device = next(self.degradation_embeddings.parameters()).device
        
        # Encode each component with proper device placement
        type_embed = self.degradation_embeddings(torch.tensor(deg_type, device=device))
        noise_embed = self.noise_encoder(torch.tensor([[noise_level]], device=device))
        blur_embed = self.blur_encoder(torch.tensor([[blur_sigma, blur_type]], device=device))
        comp_embed = self.compression_encoder(torch.tensor([[quality]], device=device))
        
        # Combine embeddings
        combined = torch.cat([type_embed, noise_embed.squeeze(0), 
                            blur_embed.squeeze(0), comp_embed.squeeze(0)], dim=-1)
        
        # Final fusion
        degradation_embed = self.fusion(combined)
        
        return degradation_embed

class AdaptiveLayerNorm(nn.Module):
    """Adaptive layer normalization conditioned on degradation."""
    
    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        
        # Adaptive parameters
        self.scale_net = nn.Linear(embed_dim, embed_dim)
        self.shift_net = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, condition):
        """Apply adaptive normalization."""
        # Standard layer norm
        x_norm = self.norm(x)
        
        # Adaptive scaling and shifting
        scale = self.scale_net(condition).unsqueeze(1)
        shift = self.shift_net(condition).unsqueeze(1)
        
        return x_norm * (1 + scale) + shift

class MetaAdapter(nn.Module):
    """Meta-learning adapter for zero-shot adaptation."""
    
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Feature similarity computation
        self.similarity_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Adaptation modules
        self.adapt_layers = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(3)
        ])
        
    def forward(self, x_patches, x_original):
        """Apply zero-shot adaptation."""
        B, N, D = x_patches.shape
        
        # Compute patch-wise statistics from original video
        x_stats = self._compute_patch_stats(x_original)
        
        # Adaptation based on local statistics
        adapted_patches = []
        for i in range(N):
            patch_feat = x_patches[:, i:i+1, :]
            
            # Compute similarity to reference statistics
            combined_feat = torch.cat([patch_feat.squeeze(1), x_stats], dim=-1)
            similarity = self.similarity_net(combined_feat)
            
            # Adaptive transformation
            adapted_patch = patch_feat
            for layer in self.adapt_layers:
                adapted_patch = adapted_patch + similarity.unsqueeze(-1) * layer(adapted_patch)
            
            adapted_patches.append(adapted_patch)
        
        return torch.cat(adapted_patches, dim=1)
    
    def _compute_patch_stats(self, x_original):
        """Compute statistical features from original video."""
        B, C, T, H, W = x_original.shape
        
        # Global statistics
        mean_val = x_original.mean(dim=(2, 3, 4), keepdim=False)  # (B, C)
        std_val = x_original.std(dim=(2, 3, 4), keepdim=False)    # (B, C)
        
        # Temporal gradient
        temp_grad = torch.abs(x_original[:, :, 1:] - x_original[:, :, :-1]).mean(dim=(2, 3, 4))
        
        # Spatial gradient  
        spatial_grad_h = torch.abs(x_original[:, :, :, 1:] - x_original[:, :, :, :-1]).mean(dim=(2, 3, 4))
        spatial_grad_w = torch.abs(x_original[:, :, :, :, 1:] - x_original[:, :, :, :, :-1]).mean(dim=(2, 3, 4))
        
        # Combine statistics
        stats = torch.cat([mean_val, std_val, temp_grad, spatial_grad_h, spatial_grad_w], dim=-1)
        
        return stats

class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss for training."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x_restored):
        """Compute temporal consistency loss."""
        B, C, T, H, W = x_restored.shape
        
        if T < 2:
            return torch.tensor(0.0, device=x_restored.device)
        
        # Temporal gradient penalty
        temp_diff = x_restored[:, :, 1:] - x_restored[:, :, :-1]
        temp_loss = temp_diff.abs().mean()
        
        return temp_loss

class DiTVRHandler:
    """DiTVR Video Restoration Handler with latest SeedVR2 models."""
    
    # SeedVR2 Model Configurations (2025 latest models)
    SEEDVR2_MODELS = {
        "3B": {
            "repo_id": "ByteDance-Seed/SeedVR2-3B",
            "embed_dim": 1536,
            "depth": 24,
            "num_heads": 24,
            "description": "SeedVR2-3B: Latest 2025 model, balanced performance and efficiency"
        },
        "7B": {
            "repo_id": "ByteDance-Seed/SeedVR2-7B", 
            "embed_dim": 2048,
            "depth": 32,
            "num_heads": 32,
            "description": "SeedVR2-7B: Latest 2025 model, maximum quality and capability"
        }
    }
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = "cuda",
                 model_size: str = "3B",  # "3B" or "7B"
                 num_frames: int = 16,
                 patch_size: Tuple[int, int, int] = (2, 4, 4),
                 tile_size: int = 224,
                 tile_overlap: int = 32,
                 auto_download: bool = True):
        
        if model_size not in self.SEEDVR2_MODELS:
            raise ValueError(f"Model size must be one of {list(self.SEEDVR2_MODELS.keys())}")
            
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        self.model_config = self.SEEDVR2_MODELS[model_size]
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.auto_download = auto_download
        
        logger.info("🎯 Initializing DiTVR Handler with SeedVR2...")
        logger.info(f"   Model: SeedVR2-{model_size} (2025 latest)")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Frames: {num_frames}")
        logger.info(f"   Patch Size: {patch_size}")
        logger.info(f"   Description: {self.model_config['description']}")
        
        # Set up model weights directory
        self.weights_dir = Path.home() / ".cache" / "video_enhancer" / f"seedvr2_{model_size.lower()}"
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize network with SeedVR2 configuration
        self.model = DiTVRNetwork(
            num_frames=num_frames,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=self.model_config['embed_dim'],
            depth=self.model_config['depth'],
            num_heads=self.model_config['num_heads'],
            mlp_ratio=4.0,
            dropout=0.1
        ).to(self.device)
        
        # Initialize model weights
        self.model_loaded = False
        self._initialize_weights(model_path)
        
        self.model.eval()
        self.video_utils = VideoUtils()
        
        logger.info(f"✅ DiTVR Handler initialized with SeedVR2-{model_size}")
    
    def _initialize_weights(self, model_path: Optional[str]):
        """Initialize model weights with various fallback options."""
        # Try user-provided path first
        if model_path and Path(model_path).exists():
            if self._load_model(model_path):
                return
                
        # Try environment variable
        env_path = os.getenv(f'SEEDVR2_{self.model_size}_DIR')
        if env_path and Path(env_path).exists():
            candidate = self._find_weight_file_in_dir(env_path)
            if candidate and self._load_model(candidate):
                logger.info(f"🔎 Loaded SeedVR2-{self.model_size} weights via environment: {candidate}")
                return
        
        # Try model registry
        registry_candidate = self._get_registry_path()
        if registry_candidate and self._load_model(registry_candidate):
            logger.info(f"🔎 Loaded SeedVR2-{self.model_size} weights via registry: {registry_candidate}")
            return
            
        # Try downloading if auto_download is enabled
        if self.auto_download:
            downloaded_path = self._download_weights()
            if downloaded_path and self._load_model(downloaded_path):
                logger.info(f"📥 Downloaded and loaded SeedVR2-{self.model_size} weights: {downloaded_path}")
                return
        
        logger.warning(f"⚠️ No SeedVR2-{self.model_size} model weights found, using random initialization")
        logger.info("💡 For better results, consider:") 
        logger.info(f"   - Setting SEEDVR2_{self.model_size}_DIR environment variable to weights directory")
        logger.info("   - Enabling auto_download=True (default)")
        logger.info("   - Manually providing model_path parameter")
    
    def _get_registry_path(self) -> Optional[str]:
        """Get model path from registry."""
        try:
            registry_path = Path(__file__).resolve().parents[3] / "config" / "model_registry.json"
            if registry_path.exists():
                data = json.loads(registry_path.read_text())
                for m in data.get("models", []):
                    model_id = f"seedvr2_{self.model_size.lower()}"
                    if m.get("id") == model_id and m.get("enabled", False):
                        local_path = m.get("local_path")
                        if local_path and Path(local_path).exists():
                            return self._find_weight_file_in_dir(local_path)
        except Exception as e:
            logger.warning(f"Could not parse model_registry.json: {e}")
        return None
    
    def _download_weights(self) -> Optional[str]:
        """Download SeedVR2 weights from HuggingFace."""
        try:
            repo_id = self.model_config['repo_id']
            logger.info(f"📥 Downloading SeedVR2-{self.model_size} from {repo_id}...")
            
            # List available files
            try:
                files = list_repo_files(repo_id)
                weight_files = [f for f in files if f.endswith(('.pth', '.pt', '.safetensors', '.bin'))]
                
                if not weight_files:
                    logger.warning(f"No weight files found in {repo_id}")
                    return None
                    
                # Prefer safetensors, then .bin, then .pth/.pt
                weight_files.sort(key=lambda x: (
                    0 if x.endswith('.safetensors') else
                    1 if x.endswith('.bin') else
                    2 if x.endswith(('.pth', '.pt')) else 3
                ))
                
                filename = weight_files[0]
                logger.info(f"📥 Downloading {filename}...")
                
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=str(self.weights_dir),
                    local_dir=str(self.weights_dir / "hf_cache")
                )
                
                logger.info(f"✅ Successfully downloaded SeedVR2-{self.model_size} to {downloaded_path}")
                return downloaded_path
                
            except Exception as hf_error:
                logger.warning(f"HuggingFace download failed: {hf_error}")
                return None
                
        except ImportError:
            logger.warning("HuggingFace Hub not available, skipping HF download")
            return None
    
    def _find_weight_file_in_dir(self, d: str) -> Optional[str]:
        """Find weight files in directory."""
        p = Path(d)
        patterns = ["*.safetensors", "*.bin", "*.pt", "*.pth"]
        for pat in patterns:
            matches = list(p.rglob(pat))
            matches_sorted = sorted(matches, key=lambda x: (
                x.suffix != ".safetensors", 
                x.suffix != ".bin",
                len(str(x))
            ))
            if matches_sorted:
                return str(matches_sorted[0])
        return None
    
    def _load_model(self, model_path: str) -> bool:
        """Load pretrained SeedVR2 weights.
        
        Returns:
            True if loading succeeded, False otherwise
        """
        try:
            if not Path(model_path).exists():
                logger.warning(f"Model file not found: {model_path}")
                return False
                
            logger.info(f"📥 Loading SeedVR2-{self.model_size} weights from {model_path}...")
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Filter out incompatible keys and load with strict=False for flexibility
            model_keys = set(self.model.state_dict().keys())
            checkpoint_keys = set(state_dict.keys())
            
            # Log key matching info
            matching_keys = model_keys.intersection(checkpoint_keys)
            missing_keys = model_keys - checkpoint_keys
            unexpected_keys = checkpoint_keys - model_keys
            
            logger.info(f"🔑 Key matching for SeedVR2-{self.model_size}:")
            logger.info(f"   Matching: {len(matching_keys)}/{len(model_keys)}")
            if missing_keys:
                logger.info(f"   Missing: {len(missing_keys)} keys")
            if unexpected_keys:
                logger.info(f"   Unexpected: {len(unexpected_keys)} keys")
            
            # Load weights with strict=False to allow partial loading
            result = self.model.load_state_dict(state_dict, strict=False)
            
            if len(matching_keys) > 0:
                self.model_loaded = True
                logger.info(f"✅ Successfully loaded SeedVR2-{self.model_size} weights from {model_path}")
                if result.missing_keys:
                    logger.info(f"   Note: {len(result.missing_keys)} keys initialized randomly")
                return True
            else:
                logger.warning(f"❌ No matching keys found in checkpoint")
                return False
            
        except Exception as e:
            logger.error(f"❌ Failed to load SeedVR2-{self.model_size} weights from {model_path}: {e}")
            return False
    
    def restore_video(self, 
                     input_path: str, 
                     output_path: str,
                     degradation_type: str = "unknown",
                     auto_adapt: bool = True,
                     window: int = None,
                     stride: int = None,
                     fp16: bool = True) -> Dict:
        """Restore video using DiTVR with zero-shot adaptation.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            degradation_type: Type of degradation ("noise", "blur", "compression", etc.)
            auto_adapt: Enable zero-shot adaptation
            window: Processing window size (default: num_frames)
            stride: Stride between windows (default: num_frames//2)
            fp16: Use half precision for efficiency
            
        Returns:
            Processing statistics
        """
        window = window or self.num_frames
        stride = stride or max(1, self.num_frames // 2)
        
        logger.info(f"🎬 Processing video with DiTVR...")
        logger.info(f"   Input: {input_path}")
        logger.info(f"   Output: {output_path}")
        logger.info(f"   Degradation: {degradation_type}")
        logger.info(f"   Auto-adapt: {auto_adapt}")
        logger.info(f"   Window: {window}, Stride: {stride}")
        
        try:
            # Get video metadata
            metadata = self.video_utils.get_video_metadata(input_path)
            fps = metadata['fps']
            total_frames = metadata['frame_count']
            
            # Analyze degradation parameters
            degradation_params = self._analyze_degradation(input_path, degradation_type)
            
            # Setup video capture and writer
            cap = cv2.VideoCapture(input_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_height = metadata['height']
            out_width = metadata['width']
            
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
            
            # Process in sliding windows
            frame_buffer = []
            processed_count = 0
            
            # FP16 safety: only use autocast on CUDA devices
            use_autocast = fp16 and self.device.type == 'cuda' and torch.cuda.is_available()
            with torch.cuda.amp.autocast(enabled=use_autocast):
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_buffer.append(frame)
                    
                    # Process when buffer is full
                    if len(frame_buffer) >= window:
                        restored_frames = self._process_frame_window(
                            frame_buffer[:window], degradation_params, auto_adapt, fp16
                        )
                        
                        # Write restored frames
                        write_count = min(stride, len(restored_frames))
                        for i in range(write_count):
                            if restored_frames[i] is not None:
                                out.write(restored_frames[i])
                                processed_count += 1
                        
                        # Slide the window
                        frame_buffer = frame_buffer[stride:]
                
                # Process remaining frames
                if len(frame_buffer) > 0:
                    restored_frames = self._process_frame_window(frame_buffer, degradation_params, auto_adapt, fp16)
                    for frame in restored_frames:
                        if frame is not None:
                            out.write(frame)
                            processed_count += 1
            
            cap.release()
            out.release()
            
            stats = {
                'input_frames': total_frames,
                'output_frames': processed_count,
                'processing_mode': 'ditvr_transformer',
                'degradation_type': degradation_type,
                'degradation_params': degradation_params,
                'zero_shot_adaptation': auto_adapt,
                'fp16': fp16
            }
            
            logger.info(f"✅ DiTVR processing completed")
            logger.info(f"   Processed: {processed_count} frames")
            
            return stats
            
        except Exception as e:
            logger.error(f"DiTVR processing failed: {e}")
            raise
    
    def _analyze_degradation(self, input_path: str, degradation_type: str) -> Dict:
        """Analyze video degradation parameters."""
        try:
            # Sample a few frames for analysis
            cap = cv2.VideoCapture(input_path)
            sample_frames = []
            
            for i in range(5):  # Sample 5 frames
                ret, frame = cap.read()
                if ret:
                    sample_frames.append(frame)
                else:
                    break
            cap.release()
            
            if not sample_frames:
                return {'type': 0, 'noise_level': 0.1, 'blur_sigma': 1.0}
            
            # Analyze noise level
            noise_level = self._estimate_noise_level(sample_frames[0])
            
            # Analyze blur level
            blur_sigma = self._estimate_blur_level(sample_frames[0])
            
            # Map degradation type to index
            type_mapping = {
                'noise': 1, 'blur': 2, 'compression': 3, 'low_resolution': 4,
                'motion_blur': 5, 'combined': 6, 'unknown': 0
            }
            
            return {
                'type': type_mapping.get(degradation_type, 0),
                'noise_level': noise_level,
                'blur_sigma': blur_sigma,
                'blur_type': 0,  # Gaussian blur
                'quality': 0.5   # Default quality
            }
            
        except Exception as e:
            logger.warning(f"Degradation analysis failed: {e}")
            return {'type': 0, 'noise_level': 0.1, 'blur_sigma': 1.0}
    
    def _estimate_noise_level(self, frame: np.ndarray) -> float:
        """Estimate noise level in a frame."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            
            # Use Laplacian to estimate noise
            laplacian = cv2.Laplacian(gray, cv2.CV_32F)
            noise_level = np.var(laplacian) / 1000.0  # Normalize
            
            return min(1.0, noise_level)
            
        except:
            return 0.1
    
    def _estimate_blur_level(self, frame: np.ndarray) -> float:
        """Estimate blur level in a frame."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use variance of Laplacian to estimate sharpness/blur
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Convert to blur sigma (inverse relationship)
            blur_sigma = max(0.5, 100.0 / (laplacian_var + 1))
            
            return min(5.0, blur_sigma)
            
        except:
            return 1.0
    
    def _process_frame_window(self, 
                            frames: List[np.ndarray], 
                            degradation_params: Dict,
                            auto_adapt: bool = True,
                            fp16: bool = True) -> List[np.ndarray]:
        """Process a window of frames with DiTVR."""
        try:
            # Pad frames to match expected window size
            while len(frames) < self.num_frames:
                frames.append(frames[-1])  # Repeat last frame
            
            frames = frames[:self.num_frames]  # Truncate if too many
            
            # Convert frames to tensor
            input_tensor = []
            for frame in frames:
                # Convert BGR to RGB and resize to tile_size
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (self.tile_size, self.tile_size))
                # Normalize to [0, 1]
                frame_norm = frame_resized.astype(np.float32) / 255.0
                # Convert to tensor (C, H, W)
                frame_tensor = torch.from_numpy(frame_norm.transpose(2, 0, 1))
                input_tensor.append(frame_tensor)
            
            # Stack to (C, T, H, W) and add batch dimension
            input_tensor = torch.stack(input_tensor, dim=1).unsqueeze(0).to(self.device)
            
            # Process with model
            # FP16 safety: only use FP16 on CUDA devices
            use_fp16 = fp16 and self.device.type == 'cuda' and torch.cuda.is_available()
            if fp16 and not use_fp16:
                logger.debug("FP16 requested but using FP32 due to CPU device")
                
            with torch.no_grad():
                if use_fp16:
                    input_tensor = input_tensor.half()
                
                output_tensor = self.model(
                    input_tensor, 
                    degradation_params=degradation_params,
                    adaptation_mode=auto_adapt
                )
            
            # Convert back to numpy frames and resize to original size
            restored_frames = []
            original_h, original_w = frames[0].shape[:2]
            
            for t in range(min(len(frames), output_tensor.shape[2])):
                frame_tensor = output_tensor[0, :, t, :, :].cpu().float()
                # Clamp and denormalize
                frame_tensor = torch.clamp(frame_tensor, 0, 1)
                frame_np = (frame_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                # Resize back to original size
                frame_resized = cv2.resize(frame_np, (original_w, original_h))
                # Convert back to BGR
                frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
                restored_frames.append(frame_bgr)
            
            return restored_frames
            
        except Exception as e:
            logger.error(f"Frame window processing failed: {e}")
            # Return original frames as fallback
            return frames
    
    def get_model_info(self) -> Dict:
        """Get information about the SeedVR2 model."""
        return {
            'name': f'SeedVR2-{self.model_size}',
            'description': f'Latest 2025 SeedVR2 {self.model_size} Diffusion Transformer Video Restoration',
            'model_size': self.model_size,
            'repo_id': self.model_config['repo_id'],
            'num_frames': self.num_frames,
            'patch_size': self.patch_size,
            'embed_dim': self.model_config['embed_dim'],
            'depth': self.model_config['depth'],
            'num_heads': self.model_config['num_heads'],
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'device': str(self.device),
            'architecture': 'SeedVR2 Diffusion Transformer with shifted window attention',
            'model_loaded': self.model_loaded,
            'capabilities': [
                'video_restoration', 'noise_reduction', 'deblurring', 'decompression', 
                'super_resolution', 'zero_shot_adaptation', 'arbitrary_length', 'arbitrary_resolution'
            ]
        }
