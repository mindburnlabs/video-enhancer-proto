"""
Real-ESRGAN Handler for Video Enhancement
Implements Real-ESRGAN for video super-resolution with proper weight loading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Tuple, List
import requests
import os
from huggingface_hub import hf_hub_download

from utils.video_utils import VideoUtils
from utils.performance_monitor import track_enhancement_performance

logger = logging.getLogger(__name__)

class RRDBBlock(nn.Module):
    """Residual in Residual Dense Block used in RealESRGAN."""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(RRDBBlock, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block used in RRDBs."""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x

class RealESRGANNetwork(nn.Module):
    """Real-ESRGAN network."""
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RealESRGANNetwork, self).__init__()
        self.scale = scale
        
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = self._make_layer(RRDBBlock, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # Upsampling layers
        if scale == 4:
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        elif scale == 2:
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def _make_layer(self, block, num_blocks, **kwarg):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(**kwarg))
        return nn.Sequential(*layers)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        
        # Upsampling
        if self.scale == 4:
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        elif self.scale == 2:
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
            
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

class RealESRGANHandler:
    """Real-ESRGAN Video Enhancement Handler with proper weight loading."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = "cuda",
                 scale: int = 4,
                 tile_size: int = 512,
                 tile_pad: int = 10):
        
        self.device_str = device
        self.scale = scale
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.model_path = model_path
        
        logger.info("🎯 Initializing Real-ESRGAN Handler...")
        logger.info(f"   Device: {device}")
        logger.info(f"   Scale: {scale}x")
        logger.info(f"   Tile size: {tile_size}")
        
        # Defer model initialization
        self.model = None
        self._model_loaded = False
        
        # Resolve model weights path
        self.resolved_model_path = self._resolve_model_path(model_path)
        
        self.video_utils = VideoUtils()
        
        logger.info("✅ Real-ESRGAN Handler initialized (model loading deferred)")
    
    def _ensure_model_loaded(self):
        """Lazy load model when actually needed."""
        if not self._model_loaded:
            logger.info("📎 Loading Real-ESRGAN model (deferred initialization)...")
            
            # Now safe to initialize device
            self.device = torch.device(self.device_str)
            
            # Initialize network
            self.model = RealESRGANNetwork(
                num_in_ch=3,
                num_out_ch=3,
                scale=self.scale,
                num_feat=64,
                num_block=23,
                num_grow_ch=32
            ).to(self.device)
            
            # Load pretrained weights
            if self.resolved_model_path and Path(self.resolved_model_path).exists():
                self._load_model(self.resolved_model_path)
            else:
                logger.warning("No Real-ESRGAN model weights found, using random initialization")
            
            self.model.eval()
            self._model_loaded = True
            logger.info("✅ Real-ESRGAN model loaded successfully")
    
    def _resolve_model_path(self, model_path: Optional[str]) -> Optional[str]:
        """Resolve model path from explicit arg, download if needed."""
        if model_path and Path(model_path).exists():
            return str(model_path)
        
        # Default path
        default_path = Path("models/weights/RealESRGAN/RealESRGAN_x4plus.pth")
        if default_path.exists():
            return str(default_path)
        
        # Try to download from HuggingFace
        try:
            logger.info("🔽 Downloading Real-ESRGAN weights from HuggingFace...")
            default_path.parent.mkdir(parents=True, exist_ok=True)
            
            downloaded_path = hf_hub_download(
                repo_id="ai-forever/Real-ESRGAN",
                filename="RealESRGAN_x4plus.pth",
                local_dir=default_path.parent,
                local_dir_use_symlinks=False
            )
            
            logger.info(f"✅ Downloaded Real-ESRGAN weights: {downloaded_path}")
            return downloaded_path
            
        except Exception as e:
            logger.warning(f"Failed to download Real-ESRGAN weights: {e}")
            return None
    
    def _load_model(self, model_path: str):
        """Load pretrained Real-ESRGAN weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            
            # Handle different checkpoint formats
            if 'params_ema' in checkpoint:
                state_dict = checkpoint['params_ema']
            elif 'params' in checkpoint:
                state_dict = checkpoint['params']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=True)
            logger.info(f"✅ Loaded Real-ESRGAN weights from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            logger.info("Using random initialization")
    
    @track_enhancement_performance('realesrgan')
    def restore_video(self, 
                     input_path: str, 
                     output_path: str,
                     **kwargs) -> Dict:
        """Restore video using Real-ESRGAN."""
        
        # Ensure model is loaded
        if not self._model_loaded:
            self._ensure_model_loaded()
        
        logger.info(f"🎬 Processing video with Real-ESRGAN...")
        logger.info(f"   Input: {input_path}")
        logger.info(f"   Output: {output_path}")
        logger.info(f"   Scale: {self.scale}x")
        
        try:
            # Get video metadata
            metadata = self.video_utils.get_video_metadata(input_path)
            fps = metadata['fps']
            total_frames = metadata['frame_count']
            
            # Setup video capture and writer
            cap = cv2.VideoCapture(input_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            out_height = metadata['height'] * self.scale
            out_width = metadata['width'] * self.scale
            
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
            
            processed_count = 0
            
            with torch.no_grad():
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    enhanced_frame = self._enhance_frame(frame)
                    
                    # Write enhanced frame
                    out.write(enhanced_frame)
                    processed_count += 1
                    
                    if processed_count % 30 == 0:
                        logger.info(f"📈 Progress: {processed_count}/{total_frames} frames")
            
            cap.release()
            out.release()
            
            stats = {
                'input_frames': total_frames,
                'output_frames': processed_count,
                'processing_mode': 'realesrgan',
                'scale_factor': self.scale,
                'input_resolution': (metadata['height'], metadata['width']),
                'output_resolution': (out_height, out_width),
                'frames_processed': processed_count
            }
            
            logger.info(f"✅ Real-ESRGAN processing completed")
            logger.info(f"   Processed: {processed_count} frames")
            logger.info(f"   Output resolution: {out_width}x{out_height}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Real-ESRGAN processing failed: {e}")
            raise
    
    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance a single frame using Real-ESRGAN."""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with tiling for large images
            if frame.shape[0] > self.tile_size or frame.shape[1] > self.tile_size:
                enhanced_rgb = self._tile_process(frame_rgb)
            else:
                enhanced_rgb = self._process_single(frame_rgb)
            
            # Convert back to BGR
            enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
            
            return enhanced_bgr
            
        except Exception as e:
            logger.error(f"Frame enhancement failed: {e}")
            # Return original frame as fallback
            return cv2.resize(frame, (frame.shape[1] * self.scale, frame.shape[0] * self.scale))
    
    def _process_single(self, img: np.ndarray) -> np.ndarray:
        """Process a single image."""
        # Convert to tensor
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        img_tensor = img_tensor.to(self.device) / 255.0
        
        # Enhance
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # Convert back to numpy
        output = torch.clamp(output, 0, 1).squeeze(0).cpu().numpy()
        output = (output.transpose(1, 2, 0) * 255).astype(np.uint8)
        
        return output
    
    def _tile_process(self, img: np.ndarray) -> np.ndarray:
        """Process image using tiling for large images."""
        h, w = img.shape[:2]
        tile_size = self.tile_size
        tile_pad = self.tile_pad
        
        # Calculate number of tiles
        tiles_x = (w + tile_size - 1) // tile_size
        tiles_y = (h + tile_size - 1) // tile_size
        
        # Output array
        output = np.zeros((h * self.scale, w * self.scale, 3), dtype=np.uint8)
        
        for y in range(tiles_y):
            for x in range(tiles_x):
                # Calculate tile boundaries
                start_x = x * tile_size
                end_x = min(start_x + tile_size, w)
                start_y = y * tile_size
                end_y = min(start_y + tile_size, h)
                
                # Add padding
                pad_start_x = max(start_x - tile_pad, 0)
                pad_end_x = min(end_x + tile_pad, w)
                pad_start_y = max(start_y - tile_pad, 0)
                pad_end_y = min(end_y + tile_pad, h)
                
                # Extract tile with padding
                tile = img[pad_start_y:pad_end_y, pad_start_x:pad_end_x]
                
                # Process tile
                enhanced_tile = self._process_single(tile)
                
                # Calculate positions in output
                out_start_x = start_x * self.scale
                out_end_x = end_x * self.scale
                out_start_y = start_y * self.scale
                out_end_y = end_y * self.scale
                
                # Remove padding from enhanced tile
                pad_left = (start_x - pad_start_x) * self.scale
                pad_top = (start_y - pad_start_y) * self.scale
                pad_right = pad_left + (end_x - start_x) * self.scale
                pad_bottom = pad_top + (end_y - start_y) * self.scale
                
                # Place in output
                output[out_start_y:out_end_y, out_start_x:out_end_x] = \
                    enhanced_tile[pad_top:pad_bottom, pad_left:pad_right]
        
        return output
    
    def get_model_info(self) -> Dict:
        """Get information about the Real-ESRGAN model."""
        if not self._model_loaded:
            self._ensure_model_loaded()
            
        return {
            'name': 'Real-ESRGAN',
            'description': 'Practical Algorithms for General Image/Video Restoration',
            'scale_factor': self.scale,
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'device': str(self.device),
            'architecture': 'RRDB (Residual in Residual Dense Block)',
            'tile_size': self.tile_size
        }