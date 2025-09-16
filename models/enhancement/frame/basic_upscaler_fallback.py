"""
Basic upscaling fallback for when Real-ESRGAN is unavailable.

This provides a simple OpenCV-based upscaling solution as a fallback
when Real-ESRGAN cannot be imported due to compatibility issues.
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


import os
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class BasicUpscalerFallback:
    """Basic upscaler using OpenCV when Real-ESRGAN is unavailable."""
    
    def __init__(self, device: str = "cpu", model_name: str = "LANCZOS", scale: int = 2):
        self.device = device  # Not used for OpenCV, but kept for compatibility
        self.model_name = model_name
        self.scale = scale
        
        # Map model names to OpenCV interpolation methods
        self.interpolation_methods = {
            'LANCZOS': cv2.INTER_LANCZOS4,
            'CUBIC': cv2.INTER_CUBIC,
            'LINEAR': cv2.INTER_LINEAR,
            'AREA': cv2.INTER_AREA,
            'NEAREST': cv2.INTER_NEAREST,
        }
        
        self.interpolation = self.interpolation_methods.get(
            model_name.upper(), 
            cv2.INTER_LANCZOS4  # Default to Lanczos
        )
        
        logger.info(f"🔧 BasicUpscaler initialized: {model_name} (OpenCV {cv2.__version__})")
        logger.info(f"   Scale: {scale}x, Method: {model_name}")

    def enhance_video(self, input_path: str, output_path: str) -> Dict:
        """
        Enhance video using basic OpenCV upscaling.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            
        Returns:
            Dict with processing statistics
        """
        logger.info(f"🎬 Processing video with BasicUpscaler...")
        logger.info(f"   Input: {input_path}")
        logger.info(f"   Output: {output_path}")
        logger.info(f"   Method: {self.model_name}, Scale: {self.scale}x")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate output dimensions
        out_w, out_h = width * self.scale, height * self.scale
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
        
        processed = 0
        failed = 0
        
        logger.info(f"📊 Input: {width}x{height} -> Output: {out_w}x{out_h}")
        logger.info(f"🎯 Processing {total_frames} frames...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # Apply additional preprocessing for better results
                    if self.model_name.upper() == 'LANCZOS':
                        # For Lanczos, apply slight denoising first
                        frame = cv2.bilateralFilter(frame, 5, 50, 50)
                    
                    # Upscale the frame
                    upscaled = cv2.resize(frame, (out_w, out_h), interpolation=self.interpolation)
                    
                    # Apply post-processing for better quality
                    if self.scale >= 2:
                        # Apply mild sharpening for upscaled images
                        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
                        kernel = kernel * 0.1  # Mild sharpening
                        upscaled = cv2.filter2D(upscaled, -1, kernel)
                        upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)
                    
                    out.write(upscaled)
                    processed += 1
                    
                    # Progress logging
                    if processed % 30 == 0:  # Every ~1 second at 30fps
                        progress = (processed / total_frames) * 100 if total_frames > 0 else 0
                        logger.info(f"🚀 Progress: {progress:.1f}% ({processed}/{total_frames})")
                    
                except Exception as frame_error:
                    logger.warning(f"⚠️ Frame {processed} processing failed: {frame_error}")
                    # Fallback: use simple resize without post-processing
                    try:
                        simple_upscaled = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
                        out.write(simple_upscaled)
                        processed += 1
                        failed += 1
                    except Exception as fallback_error:
                        logger.error(f"❌ Frame {processed} fallback failed: {fallback_error}")
                        failed += 1
                        
        finally:
            cap.release()
            out.release()
        
        stats = {
            'frames_processed': processed,
            'frames_failed': failed,
            'scale': self.scale,
            'model': f"BasicUpscaler-{self.model_name}",
            'input_resolution': f"{width}x{height}",
            'output_resolution': f"{out_w}x{out_h}",
            'interpolation_method': self.model_name,
        }
        
        logger.info(f"✅ BasicUpscaler processing completed")
        logger.info(f"   Processed: {processed} frames")
        logger.info(f"   Failed: {failed} frames")
        logger.info(f"   Success rate: {(processed-failed)/processed*100:.1f}%" if processed > 0 else "N/A")
        
        return stats
        
    def get_model_info(self) -> Dict:
        """Get information about the upscaler."""
        return {
            'name': 'BasicUpscaler',
            'description': f'OpenCV-based upscaling using {self.model_name}',
            'scale': self.scale,
            'device': 'CPU (OpenCV)',
            'interpolation_method': self.model_name,
            'capabilities': ['upscaling', 'basic_enhancement'],
            'opencv_version': cv2.__version__
        }