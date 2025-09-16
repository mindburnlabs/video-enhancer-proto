
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
import cv2
import numpy as np
from PIL import Image
import subprocess
import tempfile
import os
import sys
import logging
from typing import List, Tuple, Optional
import gc

class EnhancedRIFEHandler:
    def __init__(self, device="cuda", model_variant="RIFE_HDv3"):
        """
        Enhanced RIFE handler with improved interpolation capabilities
        
        Args:
            device: Device to run on ("cuda", "cpu")
            model_variant: RIFE model variant ("RIFE_HDv3", "RIFE_HDv2")
        """
        self.device = device
        self.model_variant = model_variant
        self.logger = logging.getLogger(__name__)
        
        # Download RIFE model if not exists
        self.model_path = "/data/models/rife"
        if not os.path.exists(self.model_path):
            self._download_rife_model(self.model_path)
        
        # Load model
        self._load_model()
        
        self.logger.info(f"Enhanced RIFE handler initialized with {model_variant} on {device}")
    
    def _download_rife_model(self, model_path):
        """Download RIFE model files"""
        try:
            os.makedirs(model_path, exist_ok=True)
            
            # Clone RIFE repository
            self.logger.info("Downloading RIFE model...")
            subprocess.run([
                "git", "clone", 
                "https://github.com/megvii-research/ECCV2022-RIFE.git",
                model_path
            ], check=True)
            
            self.logger.info("✅ RIFE model downloaded successfully")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to download RIFE model: {e}")
            raise
    
    def _load_model(self):
        """Load RIFE model"""
        try:
            # Add RIFE path to sys.path
            if self.model_path not in sys.path:
                sys.path.append(self.model_path)
            
            # Import RIFE model
            if self.model_variant == "RIFE_HDv3":
                from model.RIFE_HDv3 import Model
            else:
                from model.RIFE_HD import Model
            
            self.model = Model()
            
            # Load model weights
            model_weights_path = os.path.join(self.model_path, "train_log")
            if os.path.exists(model_weights_path):
                self.model.load_model(model_weights_path, -1)
            else:
                self.logger.warning("Model weights not found, using default initialization")
            
            self.model.eval()
            
            if self.device == "cuda" and torch.cuda.is_available():
                self.model.device()
            
            self.logger.info("✅ RIFE model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load RIFE model: {e}")
            raise
    
    def interpolate_between_frames(self, frame1: np.ndarray, frame2: np.ndarray, 
                                 num_intermediate: int = 1, scale: float = 1.0,
                                 quality_mode: str = "balanced") -> List[np.ndarray]:
        """
        Generate intermediate frames between two frames with enhanced quality
        
        Args:
            frame1: First frame (numpy array, BGR)
            frame2: Second frame (numpy array, BGR)
            num_intermediate: Number of intermediate frames to generate
            scale: Processing scale (1.0 = original resolution)
            quality_mode: Processing quality ("fast", "balanced", "high")
            
        Returns:
            List of intermediate frames
        """
        
        # Prepare input frames
        h, w = frame1.shape[:2]
        
        # Apply padding to ensure divisibility by 32 (RIFE requirement)
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        
        if pad_h > 0 or pad_w > 0:
            frame1 = cv2.copyMakeBorder(frame1, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            frame2 = cv2.copyMakeBorder(frame2, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        
        # Resize if scale != 1.0
        if scale != 1.0:
            new_h = int(frame1.shape[0] * scale)
            new_w = int(frame1.shape[1] * scale)
            # Ensure dimensions are divisible by 32
            new_h = (new_h // 32) * 32
            new_w = (new_w // 32) * 32
            
            frame1 = cv2.resize(frame1, (new_w, new_h))
            frame2 = cv2.resize(frame2, (new_w, new_h))
        
        # Convert to tensors
        I0 = torch.from_numpy(frame1.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
        I1 = torch.from_numpy(frame2.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
        
        intermediate_frames = []
        
        try:
            for i in range(1, num_intermediate + 1):
                timestep = i / (num_intermediate + 1)
                
                with torch.no_grad():
                    # Enhanced interpolation with quality adjustments
                    if quality_mode == "high":
                        # Multi-scale processing for higher quality
                        output = self._multi_scale_inference(I0, I1, timestep)
                    else:
                        # Standard inference
                        output = self.model.inference(I0, I1, timestep)
                
                # Convert back to numpy
                output_frame = (output[0] * 255.0).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
                
                # Resize back to original scale if needed
                if scale != 1.0:
                    output_frame = cv2.resize(output_frame, (w + pad_w, h + pad_h))
                
                # Remove padding
                if pad_h > 0 or pad_w > 0:
                    output_frame = output_frame[:h, :w]
                
                intermediate_frames.append(output_frame)
                
                # Memory cleanup
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
        except Exception as e:
            self.logger.error(f"Frame interpolation failed: {e}")
            # Fallback to simple blending
            for i in range(1, num_intermediate + 1):
                alpha = i / (num_intermediate + 1)
                blended = cv2.addWeighted(frame1[:h, :w], 1-alpha, frame2[:h, :w], alpha, 0)
                intermediate_frames.append(blended)
        
        return intermediate_frames
    
    def _multi_scale_inference(self, I0, I1, timestep):
        """Multi-scale inference for enhanced quality"""
        
        # Original scale inference
        output_1x = self.model.inference(I0, I1, timestep)
        
        try:
            # Half-scale inference for better motion estimation
            I0_half = torch.nn.functional.interpolate(I0, scale_factor=0.5, mode='bilinear', align_corners=False)
            I1_half = torch.nn.functional.interpolate(I1, scale_factor=0.5, mode='bilinear', align_corners=False)
            
            output_half = self.model.inference(I0_half, I1_half, timestep)
            output_half_up = torch.nn.functional.interpolate(output_half, size=I0.shape[-2:], mode='bilinear', align_corners=False)
            
            # Weighted combination
            output = 0.7 * output_1x + 0.3 * output_half_up
            
        except Exception:
            # Fallback to single scale if multi-scale fails
            output = output_1x
        
        return output
    
    def interpolate_video(self, input_path: str, output_path: str, 
                         target_fps: Optional[float] = None, 
                         interpolation_factor: int = 2,
                         quality_mode: str = "balanced",
                         progress_callback: Optional[callable] = None) -> dict:
        """
        Interpolate entire video to higher FPS with enhanced processing
        
        Args:
            input_path: Input video path
            output_path: Output video path
            target_fps: Target FPS (if None, use interpolation_factor)
            interpolation_factor: FPS multiplication factor
            quality_mode: Processing quality ("fast", "balanced", "high")
            progress_callback: Optional progress callback
            
        Returns:
            Processing statistics
        """
        
        cap = cv2.VideoCapture(input_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if target_fps is None:
            target_fps = original_fps * interpolation_factor
        
        self.logger.info(f"Interpolating video: {original_fps:.2f} → {target_fps:.2f} FPS")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
        
        prev_frame = None
        frame_count = 0
        output_frame_count = 0
        
        # Determine processing scale based on resolution and quality mode
        if quality_mode == "fast" and (width > 1920 or height > 1080):
            processing_scale = 0.5
        elif quality_mode == "high":
            processing_scale = 1.0
        else:
            processing_scale = 0.75 if (width > 1280 or height > 720) else 1.0
        
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            if prev_frame is not None:
                # Write previous frame
                out.write(prev_frame)
                output_frame_count += 1
                
                # Generate intermediate frames
                num_intermediate = int(target_fps / original_fps) - 1
                if num_intermediate > 0:
                    intermediate_frames = self.interpolate_between_frames(
                        prev_frame, curr_frame, num_intermediate, 
                        scale=processing_scale, quality_mode=quality_mode
                    )
                    
                    for inter_frame in intermediate_frames:
                        out.write(inter_frame)
                        output_frame_count += 1
                
                if progress_callback:
                    progress = (frame_count / total_frames) * 100
                    progress_callback(progress)
            
            prev_frame = curr_frame.copy()
            frame_count += 1
            
            if frame_count % 30 == 0:
                self.logger.info(f"Processed {frame_count}/{total_frames} frames")
                # Memory cleanup
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Write last frame
        if prev_frame is not None:
            out.write(prev_frame)
            output_frame_count += 1
        
        cap.release()
        out.release()
        
        stats = {
            "input_frames": total_frames,
            "output_frames": output_frame_count,
            "original_fps": original_fps,
            "target_fps": target_fps,
            "interpolation_factor": output_frame_count / total_frames if total_frames > 0 else 0,
            "processing_scale": processing_scale,
            "quality_mode": quality_mode,
            "model_variant": self.model_variant
        }
        
        self.logger.info(f"Enhanced RIFE interpolation completed: {stats}")
        return stats
    
    def batch_interpolate_frames(self, frames: List[np.ndarray], 
                                interpolation_factor: int = 2,
                                quality_mode: str = "balanced") -> List[np.ndarray]:
        """
        Batch interpolate a sequence of frames
        
        Args:
            frames: List of input frames
            interpolation_factor: FPS multiplication factor
            quality_mode: Processing quality mode
            
        Returns:
            List of interpolated frames
        """
        
        if len(frames) < 2:
            return frames
        
        interpolated_sequence = []
        
        for i in range(len(frames) - 1):
            # Add current frame
            interpolated_sequence.append(frames[i])
            
            # Generate intermediate frames
            num_intermediate = interpolation_factor - 1
            if num_intermediate > 0:
                intermediate_frames = self.interpolate_between_frames(
                    frames[i], frames[i + 1], num_intermediate, quality_mode=quality_mode
                )
                interpolated_sequence.extend(intermediate_frames)
        
        # Add last frame
        interpolated_sequence.append(frames[-1])
        
        return interpolated_sequence
    
    def get_optimal_settings(self, video_info: dict) -> dict:
        """
        Get optimal interpolation settings based on video characteristics
        
        Args:
            video_info: Video information dict with fps, width, height, etc.
            
        Returns:
            Optimal settings dict
        """
        
        width = video_info.get('width', 1920)
        height = video_info.get('height', 1080)
        fps = video_info.get('fps', 30)
        duration = video_info.get('duration', 60)
        
        settings = {
            'quality_mode': 'balanced',
            'processing_scale': 1.0,
            'interpolation_factor': 2
        }
        
        # Adjust based on resolution
        if width * height > 1920 * 1080:  # 4K+
            settings['quality_mode'] = 'fast'
            settings['processing_scale'] = 0.5
        elif width * height < 1280 * 720:  # HD-
            settings['quality_mode'] = 'high'
            settings['processing_scale'] = 1.0
        
        # Adjust based on FPS
        if fps < 15:
            settings['interpolation_factor'] = 4
        elif fps < 24:
            settings['interpolation_factor'] = 2
        else:
            settings['interpolation_factor'] = 2
        
        # Adjust based on duration
        if duration > 300:  # 5+ minutes
            settings['quality_mode'] = 'fast'
        
        return settings
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self.model, 'to'):
            self.model.to('cpu')
        
        del self.model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        self.logger.info("Enhanced RIFE handler cleaned up")