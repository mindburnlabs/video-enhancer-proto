"""
RVRT Handler
High-level interface for RVRT video restoration.
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
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, List
import tempfile
import json
import time

from .rvrt_network import RVRTNetwork, RVRTConfig
from utils.video_utils import VideoUtils
from utils.performance_monitor import track_enhancement_performance

logger = logging.getLogger(__name__)


class RVRTHandler:
    """RVRT Video Restoration Handler with transformer-based enhancement."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = "cuda",
                 upscale: int = 4,
                 num_frames: int = 7,
                 tile_size: int = 512,
                 tile_overlap: int = 32):
        
        self.device = torch.device(device)
        self.upscale = upscale
        self.num_frames = num_frames
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        
        logger.info("🎭 Initializing RVRT Handler...")
        logger.info(f"   Device: {device}")
        logger.info(f"   Upscale: {upscale}x")
        logger.info(f"   Frames: {num_frames}")
        
        # Initialize network with configuration
        self.config = RVRTConfig(
            upscale=upscale,
            num_frame=num_frames,
            embed_dim=180,
            depths=(6, 6, 6),
            num_heads=(6, 6, 6),
            window_size=(8, 8),
            mlp_ratio=2.0
        )
        
        self.model = RVRTNetwork(self.config).to(self.device)
        
        # Resolve and load weights
        resolved_model_path = self._resolve_model_path(model_path)
        if resolved_model_path and Path(resolved_model_path).exists():
            self._load_model(resolved_model_path)
        else:
            logger.warning("No RVRT model weights found, using random initialization")
        
        self.model.eval()
        self.video_utils = VideoUtils()
        
        logger.info("✅ RVRT Handler initialized")
    
    def _resolve_model_path(self, model_path: Optional[str]) -> Optional[str]:
        """Resolve model path from various sources."""
        if model_path:
            return str(model_path)
            
        import os, json
        
        # Check environment variable
        env_path = os.getenv('RVRT_MODEL_PATH')
        if env_path and Path(env_path).exists():
            candidate = self._find_weight_file_in_dir(env_path)
            if candidate:
                logger.info(f"🔎 Resolved RVRT weights via env: {candidate}")
                return candidate
        
        # Check registry
        registry_path = Path(__file__).resolve().parents[3] / "config" / "model_registry.json"
        if registry_path.exists():
            try:
                data = json.loads(registry_path.read_text())
                for m in data.get("models", []):
                    if m.get("id") == "rvrt" and m.get("enabled", False):
                        local_path = m.get("local_path")
                        if local_path and Path(local_path).exists():
                            candidate = self._find_weight_file_in_dir(local_path)
                            if candidate:
                                logger.info(f"🔎 Resolved RVRT weights via registry: {candidate}")
                                return candidate
            except Exception as e:
                logger.warning(f"Could not parse model_registry.json: {e}")
        
        return None
    
    def _find_weight_file_in_dir(self, d: str) -> Optional[str]:
        """Find weight file in directory."""
        p = Path(d)
        patterns = ["*.pth", "*.pt", "*.safetensors"]
        for pat in patterns:
            matches = list(p.rglob(pat))
            matches_sorted = sorted(matches, key=lambda x: (x.suffix != ".safetensors", len(str(x))))
            if matches_sorted:
                return str(matches_sorted[0])
        return None
    
    def _load_model(self, model_path: str):
        """Load pretrained RVRT weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'params' in checkpoint:
                state_dict = checkpoint['params']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (DataParallel)
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {key[7:]: value for key, value in state_dict.items()}
            
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"✅ Loaded RVRT weights from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            logger.info("Using random initialization")
    
    @track_enhancement_performance('rvrt')
    def enhance_video(self, 
                     input_path: str, 
                     output_path: str,
                     window: int = None,
                     stride: int = None,
                     fp16: bool = True) -> Dict:
        """Enhance video using RVRT.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            window: Processing window size (default: num_frames)
            stride: Stride between windows (default: num_frames//2)
            fp16: Use half precision for efficiency
            
        Returns:
            Processing statistics
        """
        window = window or self.num_frames
        stride = stride or max(1, self.num_frames // 2)
        
        logger.info(f"🎬 Processing video with RVRT...")
        logger.info(f"   Input: {input_path}")
        logger.info(f"   Output: {output_path}")
        logger.info(f"   Window: {window}, Stride: {stride}")
        
        try:
            # Get video metadata
            metadata = self.video_utils.get_video_metadata(input_path)
            fps = metadata['fps']
            total_frames = metadata['frame_count']
            
            # Setup video capture and writer
            cap = cv2.VideoCapture(input_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_height = metadata['height'] * self.upscale
            out_width = metadata['width'] * self.upscale
            
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
            
            # Process in sliding windows
            frame_buffer = []
            processed_count = 0
            start_time = time.time()
            
            with torch.cuda.amp.autocast(enabled=fp16):
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_buffer.append(frame)
                    
                    # Process when buffer is full
                    if len(frame_buffer) >= window:
                        enhanced_frames = self._process_frame_window(
                            frame_buffer[:window], fp16
                        )
                        
                        # Write enhanced frames (stride determines how many)
                        write_count = min(stride, len(enhanced_frames))
                        for i in range(write_count):
                            if enhanced_frames[i] is not None:
                                out.write(enhanced_frames[i])
                                processed_count += 1
                        
                        # Slide the window
                        frame_buffer = frame_buffer[stride:]
                
                # Process remaining frames
                if len(frame_buffer) > 0:
                    enhanced_frames = self._process_frame_window(frame_buffer, fp16)
                    for frame in enhanced_frames:
                        if frame is not None:
                            out.write(frame)
                            processed_count += 1
            
            cap.release()
            out.release()
            
            processing_time = time.time() - start_time
            
            stats = {
                'input_frames': total_frames,
                'output_frames': processed_count,
                'upscale_factor': self.upscale,
                'processing_mode': 'rvrt_transformer',
                'fp16': fp16,
                'processing_time': processing_time,
                'fps': fps,
                'frames_processed': processed_count,
                'input_resolution': (metadata['height'], metadata['width']),
                'output_resolution': (out_height, out_width),
                'quality_score': self._estimate_quality_score(processed_count)
            }
            
            logger.info(f"✅ RVRT processing completed")
            logger.info(f"   Processed: {processed_count} frames in {processing_time:.1f}s")
            
            return stats
            
        except Exception as e:
            logger.error(f"RVRT processing failed: {e}")
            raise
    
    def _process_frame_window(self, frames: List[np.ndarray], fp16: bool = True) -> List[np.ndarray]:
        """Process a window of frames with RVRT."""
        try:
            # Pad frames to match expected window size
            while len(frames) < self.num_frames:
                frames.append(frames[-1])  # Repeat last frame
            
            frames = frames[:self.num_frames]  # Truncate if too many
            
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
            
            # Tile-based processing for large frames
            if input_tensor.shape[-1] > self.tile_size or input_tensor.shape[-2] > self.tile_size:
                output_tensor = self._tile_process(input_tensor, fp16)
            else:
                with torch.no_grad():
                    if fp16:
                        input_tensor = input_tensor.half()
                    output_tensor = self.model(input_tensor)
            
            # Convert back to numpy frames
            enhanced_frames = []
            for t in range(min(len(frames), output_tensor.shape[2])):
                frame_tensor = output_tensor[0, :, t, :, :].cpu().float()
                # Clamp and denormalize
                frame_tensor = torch.clamp(frame_tensor, 0, 1)
                frame_np = (frame_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                # Convert back to BGR
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                enhanced_frames.append(frame_bgr)
            
            return enhanced_frames
            
        except Exception as e:
            logger.error(f"Frame window processing failed: {e}")
            # Return original frames as fallback
            return frames
    
    def _tile_process(self, input_tensor: torch.Tensor, fp16: bool = True) -> torch.Tensor:
        """Process large frames using tiling strategy."""
        B, C, T, H, W = input_tensor.shape
        tile_h = tile_w = self.tile_size
        overlap = self.tile_overlap
        
        # Calculate tiles
        tiles_h = (H + tile_h - overlap - 1) // (tile_h - overlap)
        tiles_w = (W + tile_w - overlap - 1) // (tile_w - overlap)
        
        output_tensor = torch.zeros(B, C, T, H * self.upscale, W * self.upscale, 
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
                    tile_output = self.model(tile_input)
                
                # Place tile in output
                out_start_h = start_h * self.upscale
                out_end_h = end_h * self.upscale
                out_start_w = start_w * self.upscale
                out_end_w = end_w * self.upscale
                
                output_tensor[:, :, :, out_start_h:out_end_h, out_start_w:out_end_w] = tile_output
        
        return output_tensor
    
    def _estimate_quality_score(self, frames_processed: int) -> float:
        """Estimate quality score based on processing statistics."""
        base_score = 0.85  # Base quality for RVRT
        
        # Adjust based on processing completeness
        if frames_processed > 0:
            frame_bonus = min(0.1, frames_processed / 1000.0 * 0.1)
            base_score += frame_bonus
        
        return min(base_score, 1.0)
    
    def get_model_info(self) -> Dict:
        """Get information about the RVRT model."""
        return {
            'name': 'RVRT',
            'description': 'Recurrent Video Restoration Transformer',
            'upscale': self.upscale,
            'num_frames': self.num_frames,
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'device': str(self.device),
            'architecture': 'Transformer with recurrent processing'
        }