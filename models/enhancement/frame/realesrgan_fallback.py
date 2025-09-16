"""
Real-ESRGAN frame-wise fallback upscaler.
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

# Apply torchvision compatibility fix before importing realesrgan
from utils.torchvision_compatibility import patch_realesrgan_import

logger = logging.getLogger(__name__)

class RealESRGANFallback:
    def __init__(self, device: str = "cuda", model_name: str = "RealESRGAN_x4plus", scale: int = 2):
        self.device = device
        self.model_name = model_name
        self.scale = scale
        
        # Apply compatibility patch and lazy import to reduce app startup overhead
        try:
            # Ensure compatibility fix is applied
            patch_success = patch_realesrgan_import()
            if not patch_success:
                logger.warning("⚠️ TorchVision compatibility fix failed, attempting direct import...")
            
            from realesrgan import RealESRGANer
            
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=None,  # auto-download
                model=self.model_name,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=(device == 'cuda')
            )
            logger.info(f"✅ Real-ESRGAN initialized: {self.model_name}")
            
        except ImportError as e:
            if "functional_tensor" in str(e):
                logger.error(f"❌ Real-ESRGAN functional_tensor compatibility issue: {e}")
                logger.info("💡 This is a known torchvision compatibility issue. The main video processing still works.")
            else:
                logger.error(f"❌ Real-ESRGAN import failed: {e}")
            raise
            
        except Exception as e:
            logger.error(f"❌ Real-ESRGAN initialization failed: {e}")
            raise

    def enhance_video(self, input_path: str, output_path: str) -> Dict:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_w, out_h = width * self.scale, height * self.scale
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
        processed = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                output, _ = self.upsampler.enhance(rgb, outscale=self.scale)
                bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                out.write(bgr)
                processed += 1
            except Exception as e:
                logger.warning(f"Real-ESRGAN frame failed: {e}")
                # Fallback: resize
                up = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
                out.write(up)
                processed += 1
        cap.release(); out.release()
        return { 'frames_processed': processed, 'scale': self.scale, 'model': self.model_name }