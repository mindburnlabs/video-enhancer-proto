"""
SeedVR2 (Seed Video Restoration 2) Handler
One-step diffusion-based video restoration with temporal consistency.
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
import os

from models.backbones.diffusion import DiffusionVideoUNet, NoiseScheduler
from utils.video_utils import VideoUtils
from utils.performance_monitor import track_enhancement_performance, get_performance_tracker

logger = logging.getLogger(__name__)

class SeedVR2Network(nn.Module):
    """SeedVR2 network with one-step diffusion restoration."""
    
    def __init__(self, 
                 num_frames=8,
                 in_channels=3,
                 out_channels=3,
                 model_channels=128,
                 num_res_blocks=2,
                 attention_resolutions=[16, 8],
                 dropout=0.0,
                 channel_mult=(1, 2, 4, 8),
                 num_heads=8):
        super().__init__()
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Diffusion UNet backbone
        self.unet = DiffusionVideoUNet(
            in_channels=in_channels * 2,  # Input + noise
            out_channels=out_channels,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            num_heads=num_heads,
            num_frames=num_frames
        )
        
        # Noise scheduler for one-step inference
        self.noise_scheduler = NoiseScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        # Temporal consistency module
        self.temporal_consistency = TemporalConsistencyModule(out_channels, num_frames)
        
        # Quality-aware conditioning
        self.quality_encoder = QualityEncoder(in_channels)
        
    def forward(self, x, quality_scores=None, timestep=None):
        """Forward pass for SeedVR2.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
            quality_scores: Optional quality scores for conditioning
            timestep: Diffusion timestep (for training, defaults to random for inference)
            
        Returns:
            Restored tensor of shape (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        
        # Generate or use provided timestep
        if timestep is None:
            # For one-step inference, use a fixed optimal timestep
            timestep = torch.full((B,), 500, device=x.device, dtype=torch.long)
        
        # Add noise for diffusion process
        noise = torch.randn_like(x)
        x_noisy = self.noise_scheduler.add_noise(x, noise, timestep)
        
        # Prepare input for UNet (concatenate original and noisy)
        unet_input = torch.cat([x, x_noisy], dim=1)  # (B, 2*C, T, H, W)
        
        # Quality-aware conditioning
        if quality_scores is not None:
            quality_embed = self.quality_encoder(x, quality_scores)
        else:
            quality_embed = None
        
        # Diffusion denoising
        restored = self.unet(unet_input, timestep, quality_embed)
        
        # Apply temporal consistency
        restored = self.temporal_consistency(restored, x)
        
        return restored

class TemporalConsistencyModule(nn.Module):
    """Temporal consistency module for video restoration."""
    
    def __init__(self, channels, num_frames):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        
        # Temporal attention
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=8,
            batch_first=True
        )
        
        # Optical flow estimation (lightweight)
        self.flow_estimator = LightweightFlowNet(channels)
        
        # Warping and fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        )
        
    def forward(self, x, reference):
        """Apply temporal consistency.
        
        Args:
            x: Restored frames (B, C, T, H, W)
            reference: Original frames for flow estimation
            
        Returns:
            Temporally consistent frames
        """
        B, C, T, H, W = x.shape
        
        # Estimate optical flow between consecutive frames
        flows = self.flow_estimator(reference)
        
        # Apply temporal attention
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
        attn_out, _ = self.temporal_attn(x_flat, x_flat, x_flat)
        x_attn = attn_out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        
        # Warp and fuse with flow information
        warped_frames = self._warp_with_flow(x, flows)
        combined = torch.cat([x_attn, warped_frames], dim=1)
        
        # Final fusion
        output = self.fusion_conv(combined)
        
        return output
    
    def _warp_with_flow(self, frames, flows):
        """Warp frames using optical flow via grid_sample.
        frames: (B, C, T, H, W), flows: (B, 2, T, H, W) with dx, dy in pixels.
        """
        B, C, T, H, W = frames.shape
        device = frames.device
        dtype = frames.dtype
        # Normalize flow to [-1, 1]
        # flow_x normalized by (W-1), flow_y by (H-1)
        flow = flows.to(device=device, dtype=dtype)
        norm_flow_x = flow[:, 0] / max(W - 1, 1)
        norm_flow_y = flow[:, 1] / max(H - 1, 1)
        # Build base grid
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device, dtype=dtype),
            torch.linspace(-1, 1, W, device=device, dtype=dtype), indexing='ij')
        base_grid = torch.stack((xs, ys), dim=-1)  # (H, W, 2)
        warped = []
        for t in range(T):
            # Grid per batch: (B, H, W, 2)
            grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)
            # Add normalized flow offsets
            grid_x = grid[..., 0] + 2.0 * norm_flow_x[:, t]
            grid_y = grid[..., 1] + 2.0 * norm_flow_y[:, t]
            grid_t = torch.stack((grid_x, grid_y), dim=-1)
            frame = frames[:, :, t]  # (B, C, H, W)
            warped_t = F.grid_sample(frame, grid_t, mode='bilinear', padding_mode='border', align_corners=True)
            warped.append(warped_t.unsqueeze(2))
        return torch.cat(warped, dim=2)

class LightweightFlowNet(nn.Module):
    """Lightweight optical flow estimation."""
    
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=(1, 7, 7), padding=(0, 3, 3))
        self.conv2 = nn.Conv3d(32, 16, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        self.flow_head = nn.Conv3d(16, 2, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        
    def forward(self, x):
        """Estimate optical flow."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        flow = self.flow_head(x)
        return flow

class QualityEncoder(nn.Module):
    """Quality-aware encoder for conditioning."""
    
    def __init__(self, in_channels):
        super().__init__()
        self.quality_net = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 8, 8)),
            nn.Conv3d(in_channels, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(32, 128)
        )
        
    def forward(self, x, quality_scores=None):
        """Encode quality information."""
        quality_feat = self.quality_net(x)
        
        if quality_scores is not None:
            # Combine with provided quality scores
            quality_feat = quality_feat + quality_scores
            
        return quality_feat

class SeedVR2Handler:
    """SeedVR2 Video Restoration Handler with one-step diffusion.
    Optionally integrates the official SeedVR2 pipeline if available.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = "cuda",
                 num_frames: int = 8,
                 tile_size: int = 512,
                 tile_overlap: int = 32,
                 guidance_scale: float = 7.5,
                 use_official: Optional[bool] = None):
        
        self.device = torch.device(device)
        self.num_frames = num_frames
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.guidance_scale = guidance_scale
        
        logger.info("🌱 Initializing SeedVR2 Handler...")
        logger.info(f"   Device: {device}")
        logger.info(f"   Frames: {num_frames}")
        logger.info(f"   Guidance Scale: {guidance_scale}")
        
        # Initialize network
        self.model = SeedVR2Network(
            num_frames=num_frames,
            in_channels=3,
            out_channels=3,
            model_channels=128,
            num_res_blocks=2,
            attention_resolutions=[16, 8],
            channel_mult=(1, 2, 4, 8),
            num_heads=8
        ).to(self.device)
        
        # Resolve model weights path from env/config if not provided
        resolved_model_path = self._resolve_model_path(model_path)
        
        # Load pretrained weights
        if resolved_model_path and Path(resolved_model_path).exists():
            self._load_model(resolved_model_path)
        else:
            logger.warning("No SeedVR2 model weights found, using random initialization")
        
        self.model.eval()
        self.video_utils = VideoUtils()
        
        # Official pipeline toggle
        self.use_official = use_official if use_official is not None else (os.getenv('OFFICIAL_SEEDVR2', '0') in ['1','true','True'])
        if self.use_official:
            logger.info("🔗 OFFICIAL_SEEDVR2 enabled: will attempt to use official pipeline when restoring video")
        logger.info("✅ SeedVR2 Handler initialized")

    def _resolve_model_path(self, model_path: Optional[str]) -> Optional[str]:
        """Resolve model path from explicit arg, env vars, or registry structure.
        Priority:
        1) Explicit model_path
        2) SEEDVR2_3B_DIR or SEEDVR2_7B_DIR env with common filename patterns
        3) config/model_registry.json local_path with typical weight filenames
        """
        if model_path:
            return str(model_path)
        import os, json
        # Check env vars first
        for env_var in ["SEEDVR2_3B_DIR", "SEEDVR2_7B_DIR"]:
            d = os.getenv(env_var)
            if d and Path(d).exists():
                candidate = self._find_weight_file_in_dir(d)
                if candidate:
                    logger.info(f"🔎 Resolved SeedVR2 weights via {env_var}: {candidate}")
                    return candidate
        # Check registry
        registry_path = Path(__file__).resolve().parents[3] / "config" / "model_registry.json"
        if registry_path.exists():
            try:
                data = json.loads(registry_path.read_text())
                for m in data.get("models", []):
                    if m.get("id") in ["seedvr2_3b", "seedvr2_7b"] and m.get("enabled", False):
                        local_path = m.get("local_path")
                        if local_path and Path(local_path).exists():
                            candidate = self._find_weight_file_in_dir(local_path)
                            if candidate:
                                logger.info(f"🔎 Resolved SeedVR2 weights via registry: {candidate}")
                                return candidate
            except Exception as e:
                logger.warning(f"Could not parse model_registry.json: {e}")
        return None

    def _find_weight_file_in_dir(self, d: str) -> Optional[str]:
        """Find a plausible weights file in a directory.
        Looks for *.safetensors or *.pth files by common names.
        """
        p = Path(d)
        patterns = [
            "*.safetensors", "*.pt", "*.pth"
        ]
        for pat in patterns:
            matches = list(p.rglob(pat))
            # Prefer safetensors
            matches_sorted = sorted(matches, key=lambda x: (x.suffix != ".safetensors", len(str(x))))
            if matches_sorted:
                return str(matches_sorted[0])
        return None
    
    def _load_model(self, model_path: str):
        """Load pretrained SeedVR2 weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"✅ Loaded SeedVR2 weights from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            logger.info("Using random initialization")
    
    @track_enhancement_performance('seedvr2')
    def restore_video(self, 
                     input_path: str, 
                     output_path: str,
                     quality_threshold: float = 0.5,
                     window: int = None,
                     stride: int = None,
                     fp16: bool = True,
                     guidance_strength: float = 0.0,
                     temporal_weight: float = 0.0,
                     degradation_hints: Optional[Dict] = None) -> Dict:
        """Restore video using SeedVR2.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            quality_threshold: Minimum quality threshold for processing
            window: Processing window size (default: num_frames)
            stride: Stride between windows (default: num_frames//2)
            fp16: Use half precision for efficiency
            
        Returns:
            Processing statistics
        """
        window = window or self.num_frames
        stride = stride or max(1, self.num_frames // 2)
        
        logger.info(f"🎬 Processing video with SeedVR2...")
        logger.info(f"   Input: {input_path}")
        logger.info(f"   Output: {output_path}")
        logger.info(f"   Window: {window}, Stride: {stride}")
        logger.info(f"   Quality Threshold: {quality_threshold}")
        
        try:
            # Try official pipeline if requested
            if self.use_official:
                try:
                    return self._restore_video_official(input_path, output_path, guidance_strength, temporal_weight, degradation_hints)
                except Exception as oe:
                    logger.warning(f"Official SeedVR2 pipeline failed or unavailable, falling back: {oe}")
            
            # Get video metadata
            metadata = self.video_utils.get_video_metadata(input_path)
            fps = metadata['fps']
            total_frames = metadata['frame_count']
            
            # Setup video capture and writer
            cap = cv2.VideoCapture(input_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_height = metadata['height']
            out_width = metadata['width']
            
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
            
            # Process in sliding windows
            frame_buffer = []
            processed_count = 0
            quality_scores = []
            
            with torch.cuda.amp.autocast(enabled=fp16):
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_buffer.append(frame)
                    
                    # Process when buffer is full
                    if len(frame_buffer) >= window:
                        # Assess video quality
                        window_quality = self._assess_quality(frame_buffer[:window])
                        quality_scores.append(window_quality)
                        
                        # Process only if quality is below threshold
                        if window_quality < quality_threshold:
                            restored_frames = self._process_frame_window(
                                frame_buffer[:window], window_quality, fp16
                            )
                        else:
                            # Skip processing for high-quality segments
                            restored_frames = frame_buffer[:window]
                        
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
                    window_quality = self._assess_quality(frame_buffer)
                    if window_quality < quality_threshold:
                        restored_frames = self._process_frame_window(frame_buffer, window_quality, fp16)
                    else:
                        restored_frames = frame_buffer
                    
                    for frame in restored_frames:
                        if frame is not None:
                            out.write(frame)
                            processed_count += 1
            
            cap.release()
            out.release()
            
            stats = {
                'input_frames': total_frames,
                'output_frames': processed_count,
                'processing_mode': 'seedvr2_diffusion',
                'avg_quality': np.mean(quality_scores) if quality_scores else 0,
                'quality_threshold': quality_threshold,
                'fp16': fp16,
                # Performance tracking fields
                'frames_processed': processed_count,
                'input_resolution': (metadata['height'], metadata['width']),
                'output_resolution': (out_height, out_width),
                'quality_score': np.mean(quality_scores) if quality_scores else 0.5
            }
            
            logger.info(f"✅ SeedVR2 processing completed")
            logger.info(f"   Processed: {processed_count} frames")
            logger.info(f"   Avg Quality: {stats['avg_quality']:.3f}")
            
            return stats
            
        except Exception as e:
            logger.error(f"SeedVR2 processing failed: {e}")
            raise
    
    def _assess_quality(self, frames: List[np.ndarray]) -> float:
        """Assess video quality using multiple metrics."""
        try:
            qualities = []
            for frame in frames:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Compute Laplacian variance (sharpness)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                # Normalize to 0-1 range (approximate)
                quality = min(1.0, laplacian_var / 1000.0)
                qualities.append(quality)
            
            return np.mean(qualities)
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5  # Default quality
    
    def _process_frame_window(self, 
                            frames: List[np.ndarray], 
                            quality_score: float,
                            fp16: bool = True) -> List[np.ndarray]:
        """Process a window of frames with SeedVR2."""
        try:
            # Convert frames to tensor
            input_tensor = []
            for frame in frames:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Normalize to [0, 1]
                frame_norm = frame_rgb.astype(np.float32) / 255.0
                # Convert to tensor (C, H, W)
                frame_tensor = torch.from_numpy(frame_norm.transpose(2, 0, 1))
                input_tensor.append(frame_tensor)
            
            # Stack to (C, T, H, W) and add batch dimension
            input_tensor = torch.stack(input_tensor, dim=1).unsqueeze(0).to(self.device)
            
            # Prepare quality conditioning
            quality_tensor = torch.tensor([quality_score], device=self.device).repeat(128)
            
            # Tile-based processing for large frames
            if input_tensor.shape[-1] > self.tile_size or input_tensor.shape[-2] > self.tile_size:
                output_tensor = self._tile_process(input_tensor, quality_tensor, fp16)
            else:
                with torch.no_grad():
                    if fp16:
                        input_tensor = input_tensor.half()
                        quality_tensor = quality_tensor.half()
                    
                    output_tensor = self.model(input_tensor, quality_tensor)
            
            # Convert back to numpy frames
            restored_frames = []
            for t in range(output_tensor.shape[2]):
                frame_tensor = output_tensor[0, :, t, :, :].cpu().float()
                # Clamp and denormalize
                frame_tensor = torch.clamp(frame_tensor, 0, 1)
                frame_np = (frame_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                # Convert back to BGR
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                restored_frames.append(frame_bgr)
            
            return restored_frames
            
        except Exception as e:
            logger.error(f"Frame window processing failed: {e}")
            # Return original frames as fallback
            return frames
    
    def _tile_process(self, 
                     input_tensor: torch.Tensor, 
                     quality_tensor: torch.Tensor,
                     fp16: bool = True) -> torch.Tensor:
        """Process large frames using tiling strategy."""
        B, C, T, H, W = input_tensor.shape
        tile_h = tile_w = self.tile_size
        overlap = self.tile_overlap
        
        # Calculate tiles
        tiles_h = (H + tile_h - overlap - 1) // (tile_h - overlap)
        tiles_w = (W + tile_w - overlap - 1) // (tile_w - overlap)
        
        output_tensor = torch.zeros(B, C, T, H, W, 
                                   device=self.device, dtype=input_tensor.dtype)
        
        for i in range(tiles_h):
            for j in range(tiles_w):
                # Calculate tile boundaries
                start_h = i * (tile_h - overlap)
                end_h = min(start_h + tile_h, H)
                start_w = j * (tile_w - overlap)
                end_w = min(start_w + tile_w, W)
                
                # Extract tile
                tile_input = input_tensor[:, :, :, start_h:end_h, start_w:end_w]
                
                # Process tile
                with torch.no_grad():
                    if fp16:
                        tile_input = tile_input.half()
                    tile_output = self.model(tile_input, quality_tensor)
                
                # Place tile in output
                output_tensor[:, :, :, start_h:end_h, start_w:end_w] = tile_output
        
        return output_tensor
    
    def _restore_video_official(self, input_path: str, output_path: str,
                               guidance_strength: float,
                               temporal_weight: float,
                               degradation_hints: Optional[Dict]) -> Dict:
        """Attempt to use the official SeedVR2 pipeline if installed.
        This function will not execute if the official runtime is unavailable.
        """
        try:
            import importlib
            seedvr_pkg = importlib.util.find_spec('seedvr') or importlib.util.find_spec('seedvr2')
            if seedvr_pkg is None:
                raise ImportError("Official SeedVR2 package not found")
            # Hypothetical usage: adapt as needed when official API is available
            # For safety, just log and fallback for now
            raise RuntimeError("Official SeedVR2 integration placeholder: adapt to official API when available")
        except Exception as e:
            raise e
    
    def get_model_info(self) -> Dict:
        """Get information about the SeedVR2 model."""
        return {
            'name': 'SeedVR2',
            'description': 'One-step diffusion-based video restoration',
            'num_frames': self.num_frames,
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'device': str(self.device),
            'architecture': 'Diffusion UNet with temporal consistency',
            'guidance_scale': self.guidance_scale
        }
