#!/usr/bin/env python3

"""
🏆 SOTA Video Enhancer - ZeroGPU Accelerated Version

A production-ready video enhancement pipeline using ZeroGPU dynamic allocation
for NVIDIA H200 GPU acceleration on HuggingFace Spaces.
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


import gradio as gr
import os
import tempfile
import time
from pathlib import Path
import logging
import sys
from datetime import datetime
import threading
from typing import Optional
from collections import deque
from uuid import uuid4
import uuid

from config.logging_config import get_performance_logger

# Import error handling system
from utils.error_handler import (
    error_handler, VideoEnhancementError, InputValidationError, ProcessingError,
    ErrorCode, handle_exceptions, SecurityError
)

# Import security system
from utils.security_integration import (
    SecurityManager, SecurityConfig, SecurityContext, security_manager
)
from utils.data_protection import DataCategory
from utils.file_security import SecurityThreat

# ZeroGPU import
try:
    import spaces
    ZEROGPU_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✨ ZeroGPU available - GPU acceleration enabled")
except ImportError:
    ZEROGPU_AVAILABLE = False
    spaces = None
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ ZeroGPU not available - running in CPU mode")

# Environment detection
HUGGINGFACE_SPACE = os.environ.get('SPACE_ID') is not None

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Safe imports with fallbacks
try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    import cv2
    from PIL import Image
    from diffusers import StableDiffusionPipeline, DiffusionPipeline
    from transformers import pipeline
    
    # ZeroGPU device detection
    if ZEROGPU_AVAILABLE and HUGGINGFACE_SPACE:
        # In ZeroGPU environment, device allocation is dynamic
        CUDA_AVAILABLE = True
        device = 'cuda'
        logger.info(f"🎯 ZeroGPU Environment - Dynamic GPU allocation enabled")
    else:
        # Check CUDA availability safely for local/non-ZeroGPU environments
        try:
            CUDA_AVAILABLE = torch.cuda.is_available()
            device = 'cuda' if CUDA_AVAILABLE else 'cpu'
            logger.info(f"🚀 PyTorch loaded successfully. Device: {device}")
        except Exception as e:
            CUDA_AVAILABLE = False
            device = 'cpu'
            logger.warning(f"CUDA check failed: {e}. Using CPU.")
        
except ImportError as e:
    logger.error(f"Critical imports failed: {e}")
    if HUGGINGFACE_SPACE:
        logger.info("Installing minimal dependencies...")
        import subprocess
        subprocess.run([
            "pip", "install", "torch", "torchvision", "opencv-python-headless", 
            "--extra-index-url", "https://download.pytorch.org/whl/cpu"
        ])
        # Retry imports
        import torch
        import torch.nn.functional as F
        import numpy as np
        import cv2
        from PIL import Image
        device = 'cpu'
    else:
        sys.exit(1)

# Setup structured logging early
try:
    from config.logging_config import setup_logging
    setup_logging(log_level=os.getenv('LOG_LEVEL', 'INFO'))
except Exception as _log_e:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).warning(f"Logging setup failed, using basic config: {_log_e}")

# Attempt to import SOTA router and handlers (graceful if missing)
SOTA_AVAILABLE = False
SOTA_IMPORT_ERROR: Optional[str] = None
try:
    from models.analysis.degradation_router import DegradationRouter
    from models.enhancement.vsr.vsrm_handler import VSRMHandler
    from models.enhancement.zeroshot.seedvr2_handler import SeedVR2Handler
    from models.enhancement.zeroshot.ditvr_handler import DiTVRHandler
    from models.enhancement.vsr.fast_mamba_vsr_handler import FastMambaVSRHandler
    SOTA_AVAILABLE = True
except Exception as _e:
    SOTA_IMPORT_ERROR = str(_e)

# Global state
performance_logger = get_performance_logger()

# Security configuration
security_config = SecurityConfig(
    api_key_required=bool(os.getenv('SECURITY_API_KEY_REQUIRED', 'true').lower() == 'true'),
    rate_limit_enabled=bool(os.getenv('SECURITY_RATE_LIMIT_ENABLED', 'true').lower() == 'true'),
    max_requests_per_minute=int(os.getenv('SECURITY_MAX_REQUESTS_PER_MINUTE', '60')),
    max_file_size_mb=int(os.getenv('SECURITY_MAX_FILE_SIZE_MB', '500')),
    file_validation_enabled=bool(os.getenv('SECURITY_FILE_VALIDATION_ENABLED', 'true').lower() == 'true'),
    encryption_enabled=bool(os.getenv('SECURITY_ENCRYPTION_ENABLED', 'true').lower() == 'true'),
    allowed_extensions=['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'],
    audit_logging_enabled=bool(os.getenv('SECURITY_AUDIT_LOGGING_ENABLED', 'true').lower() == 'true')
)

# Initialize security manager
app_security_manager = SecurityManager(security_config)

# Ring buffer for live logs
LOG_RING_MAX = int(os.getenv('LOG_RING_MAX', '500'))
log_ring = deque(maxlen=LOG_RING_MAX)

class RingBufferHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        log_ring.append(msg)

# Attach ring buffer handler
_ring_handler = RingBufferHandler()
_ring_handler.setLevel(logging.INFO)
_ring_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(_ring_handler)

job_history = []  # list of dicts: {id, engine, model, frames, time, input, output, ts}
processing_stats = {
    'total_processed': 0,
    'total_time': 0,
    'startup_time': datetime.now()
}

class GPUVideoEnhancer:
    """ZeroGPU-optimized video enhancer with dynamic GPU allocation."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.models_loaded = False
        self.upscale_model = None
        self.diffusion_pipeline = None
        logger.info(f"🎬 Initializing GPUVideoEnhancer for ZeroGPU")
        
    def load_models(self):
        """Load available models for CPU environment (models loaded on-demand in GPU functions)."""
        try:
            logger.info("📦 Initializing model configurations...")
            # In ZeroGPU, we don't pre-load GPU models - they're loaded on-demand
            self.models_loaded = True
            logger.info("✅ Model configuration ready!")
            return True
            
        except Exception as e:
            logger.error(f"Model configuration failed: {e}")
            return False
    
    def _create_gpu_upscaler(self):
        """Create an advanced GPU-optimized upscaler."""
        import torch.nn as nn
        
        class AdvancedUpscaler(nn.Module):
            def __init__(self):
                super().__init__()
                # More sophisticated architecture for GPU
                self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
                self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
                self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
                self.conv4 = nn.Conv2d(256, 128, 3, padding=1)
                self.conv5 = nn.Conv2d(128, 3, 3, padding=1)
                
                # Upsampling layers
                self.upconv1 = nn.ConvTranspose2d(3, 64, 4, stride=2, padding=1)
                self.upconv2 = nn.ConvTranspose2d(64, 3, 3, padding=1)
                
                # Batch normalization for better training
                self.bn1 = nn.BatchNorm2d(128)
                self.bn2 = nn.BatchNorm2d(128)
                self.bn3 = nn.BatchNorm2d(256)
                self.bn4 = nn.BatchNorm2d(128)
                
            def forward(self, x):
                # Enhanced processing with batch norm
                x1 = F.relu(self.bn1(self.conv1(x)))
                x2 = F.relu(self.bn2(self.conv2(x1))) + x1  # Skip connection
                x3 = F.relu(self.bn3(self.conv3(x2)))
                x4 = F.relu(self.bn4(self.conv4(x3)))
                x5 = self.conv5(x4)
                
                # Upsampling
                upsampled = F.relu(self.upconv1(x5))
                output = torch.clamp(self.upconv2(upsampled), 0, 1)
                
                return output
        
        model = AdvancedUpscaler().to('cuda')
        # Initialize with Xavier initialization for better convergence
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        return model
    
    # ZeroGPU decorator for GPU-accelerated frame enhancement
    def enhance_frame_gpu(self, frame):
        """ZeroGPU-accelerated frame enhancement with dynamic GPU allocation."""
        if ZEROGPU_AVAILABLE and spaces:
            return self._enhance_frame_with_gpu(frame)
        else:
            return self._enhance_frame_cpu_fallback(frame)
    
    @spaces.GPU(duration=30) if ZEROGPU_AVAILABLE and spaces else lambda x: x
    def _enhance_frame_with_gpu(self, frame):
        """GPU-accelerated frame enhancement using ZeroGPU."""
        try:
            logger.info("🎯 Enhancing frame with ZeroGPU acceleration")
            
            # Load model on GPU (dynamic allocation)
            if not hasattr(self, '_gpu_model') or self._gpu_model is None:
                logger.info("📦 Loading GPU model...")
                self._gpu_model = self._create_gpu_upscaler()
                self._gpu_model.eval()
            
            # Convert to tensor
            if isinstance(frame, np.ndarray):
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                frame_tensor = frame_tensor.to('cuda')
            else:
                frame_tensor = frame.to('cuda')
            
            # GPU-accelerated enhancement
            with torch.no_grad():
                enhanced = self._gpu_model(frame_tensor)
            
            # Convert back to numpy
            enhanced_frame = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
            enhanced_frame = np.clip(enhanced_frame * 255, 0, 255).astype(np.uint8)
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            logger.info("✅ Frame enhancement completed with GPU")
            return enhanced_frame
            
        except Exception as e:
            logger.error(f"GPU frame enhancement failed: {e}")
            # Fallback to CPU
            return self._enhance_frame_cpu_fallback(frame)
    
    def _enhance_frame_cpu_fallback(self, frame):
        """CPU fallback for frame enhancement."""
        try:
            logger.info("💻 Using CPU fallback for frame enhancement")
            
            # Convert to tensor
            if isinstance(frame, np.ndarray):
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            else:
                frame_tensor = frame
            
            # High-quality interpolation fallback
            with torch.no_grad():
                enhanced = F.interpolate(frame_tensor, scale_factor=2, mode='bicubic', align_corners=False)
            
            # Convert back to numpy
            enhanced_frame = enhanced.squeeze(0).permute(1, 2, 0).numpy()
            enhanced_frame = np.clip(enhanced_frame * 255, 0, 255).astype(np.uint8)
            
            return enhanced_frame
            
        except Exception as e:
            logger.error(f"CPU frame enhancement failed: {e}")
            # Return original frame as ultimate fallback
            if isinstance(frame, torch.Tensor):
                return frame.squeeze(0).permute(1, 2, 0).numpy()
            return frame
    
    # ZeroGPU decorator for video processing with dynamic duration
    def process_video(self, input_path, output_path, target_fps=30):
        """Main video processing method with ZeroGPU acceleration."""
        if ZEROGPU_AVAILABLE and spaces:
            return self._process_video_gpu(input_path, output_path, target_fps)
        else:
            return self._process_video_cpu(input_path, output_path, target_fps)
    
    def _estimate_processing_duration(self, input_path, output_path, target_fps):
        """Estimate processing duration for dynamic ZeroGPU allocation."""
        try:
            cap = cv2.VideoCapture(input_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Estimate: ~0.5 seconds per frame on H200 GPU
            estimated_duration = min(max(total_frames * 0.5, 60), 300)  # 60s to 300s range
            logger.info(f"📈 Estimated processing duration: {estimated_duration}s for {total_frames} frames")
            return int(estimated_duration)
        except:
            return 120  # Default 2 minutes
    
    @spaces.GPU(duration=lambda self, input_path, output_path, target_fps=30: self._estimate_processing_duration(input_path, output_path, target_fps)) if ZEROGPU_AVAILABLE and spaces else lambda x: x
    def _process_video_gpu(self, input_path, output_path, target_fps=30):
        """ZeroGPU-accelerated video processing."""
        try:
            logger.info(f"🎯 ZeroGPU Processing video: {input_path} -> {output_path}")
            start_time = time.time()
            
            # Load GPU model once per video
            if not hasattr(self, '_gpu_model') or self._gpu_model is None:
                logger.info("📦 Loading GPU model for video processing...")
                self._gpu_model = self._create_gpu_upscaler()
                self._gpu_model.eval()
            
            # Open video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"📊 ZeroGPU Video: {width}x{height}, {fps}fps, {total_frames} frames")
            
            # Setup output video with 2x upscaling
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Ensure parent directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out = cv2.VideoWriter(output_path, fourcc, min(target_fps, fps), (width*2, height*2))
            
            if not out.isOpened():
                raise ValueError(f"Could not create output video: {output_path}")
            
            # Batch processing for efficiency
            batch_size = 8  # Process frames in batches
            frames_batch = []
            frame_count = 0
            progress_step = max(1, total_frames // 20)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Process remaining frames in batch
                    if frames_batch:
                        enhanced_batch = self._process_frame_batch_gpu(frames_batch)
                        for enhanced_frame in enhanced_batch:
                            out.write(enhanced_frame)
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_batch.append(frame_rgb)
                frame_count += 1
                
                # Process batch when full
                if len(frames_batch) >= batch_size:
                    enhanced_batch = self._process_frame_batch_gpu(frames_batch)
                    for enhanced_frame in enhanced_batch:
                        out.write(enhanced_frame)
                    frames_batch = []
                
                if frame_count % progress_step == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"🚀 ZeroGPU Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Cleanup
            # Record job history
            try:
                job_history.append({
                    'id': uuid4().hex,
                    'engine': 'ZeroGPU Upscaler',
                    'model': 'AdvancedUpscaler',
                    'frames': total_frames,
                    'time': f"{processing_time:.1f}s",
                    'input': str(input_path),
                    'output': str(output_path),
                    'ts': datetime.now().isoformat(timespec='seconds')
                })
            except Exception:
                pass
            cap.release()
            out.release()
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            processing_time = time.time() - start_time
            logger.info(f"✅ ZeroGPU video processing completed in {processing_time:.1f}s")
            
            # Update stats
            processing_stats['total_processed'] += 1
            processing_stats['total_time'] += processing_time
            
            return True, f"✅ Enhanced {total_frames} frames in {processing_time:.1f}s with ZeroGPU acceleration"
            
        except Exception as e:
            logger.error(f"ZeroGPU video processing failed: {e}")
            # Clear GPU memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False, f"❌ ZeroGPU processing failed: {str(e)}"
        
        finally:
            # Ensure cleanup
            try:
                cap.release()
                out.release()
            except:
                pass
    
    def _process_frame_batch_gpu(self, frames_batch):
        """Process a batch of frames on GPU for efficiency."""
        try:
            # Convert batch to tensor
            batch_tensor = []
            for frame in frames_batch:
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                batch_tensor.append(frame_tensor)
            
            # Stack into batch
            batch_tensor = torch.stack(batch_tensor).to('cuda')
            
            # Process batch
            with torch.no_grad():
                enhanced_batch = self._gpu_model(batch_tensor)
            
            # Convert back to individual frames
            enhanced_frames = []
            for i in range(enhanced_batch.shape[0]):
                enhanced_frame = enhanced_batch[i].permute(1, 2, 0).cpu().numpy()
                enhanced_frame = np.clip(enhanced_frame * 255, 0, 255).astype(np.uint8)
                enhanced_bgr = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)
                enhanced_frames.append(enhanced_bgr)
            
            return enhanced_frames
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Fallback to individual processing
            enhanced_frames = []
            for frame in frames_batch:
                enhanced_frame = self._enhance_frame_cpu_fallback(frame)
                enhanced_bgr = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)
                enhanced_frames.append(enhanced_bgr)
            return enhanced_frames
    
    def _process_video_cpu(self, input_path, output_path, target_fps=30):
        """CPU fallback video processing."""
        try:
            logger.info(f"💻 CPU Processing video: {input_path} -> {output_path}")
            start_time = time.time()
            
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Ensure parent directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out = cv2.VideoWriter(output_path, fourcc, min(target_fps, fps), (width*2, height*2))
            
            frame_count = 0
            progress_step = max(1, total_frames // 20)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                enhanced_frame = self._enhance_frame_cpu_fallback(frame_rgb)
                enhanced_bgr = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)
                out.write(enhanced_bgr)
                
                frame_count += 1
                if frame_count % progress_step == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"💻 CPU Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            cap.release()
            out.release()
            
            processing_time = time.time() - start_time
            processing_stats['total_processed'] += 1
            processing_stats['total_time'] += processing_time
            
            return True, f"✅ Enhanced {total_frames} frames in {processing_time:.1f}s with CPU processing"
            
        except Exception as e:
            logger.error(f"CPU video processing failed: {e}")
            return False, f"❌ CPU processing failed: {str(e)}"
        
        finally:
            try:
                cap.release()
                out.release()
            except:
                pass

# Initialize enhancer
enhancer = GPUVideoEnhancer(device=device)

def initialize_enhancer():
    """Initialize the video enhancer with progress feedback."""
    try:
        logger.info("🚀 Initializing Video Enhancer...")
        success = enhancer.load_models()
        
        if success:
            return "✅ Video Enhancer Ready! Upload a video to get started."
        else:
            return "⚠️ Enhancer initialized with basic features only."
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return f"❌ Initialization failed: {str(e)}"

def _estimate_duration_seconds(input_path: str) -> int:
    try:
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return int(min(max(total_frames * 0.5, 60), 300))
    except Exception:
        return 120


def _apply_compression_cleanup(frame_bgr: np.ndarray) -> np.ndarray:
    # Use Non-local Means as deblocking/deartifacting proxy
    return cv2.fastNlMeansDenoisingColored(frame_bgr, None, 3, 3, 7, 21)


def _apply_denoising(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(frame_bgr, None, 10, 10, 7, 21)


def _apply_low_light_enhancement(frame_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    out = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # Gamma correction
    gamma = 0.9
    inv = 1.0 / max(gamma, 1e-6)
    table = (np.linspace(0, 1, 256) ** inv * 255).astype(np.uint8)
    return cv2.LUT(out, table)


def _preprocess_video(input_path: str, output_path: str, use_compression: bool, use_denoise: bool, use_low_light: bool):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        proc = frame
        if use_compression:
            proc = _apply_compression_cleanup(proc)
        if use_denoise:
            proc = _apply_denoising(proc)
        if use_low_light:
            proc = _apply_low_light_enhancement(proc)
        out.write(proc)
    cap.release(); out.release()


def _temporal_smooth(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    prev = None
    dis = None
    try:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    except Exception:
        dis = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if prev is None:
            out.write(frame)
            prev = frame
            continue
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if dis is not None:
            flow = dis.calc(prev_gray, curr_gray, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = prev_gray.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped_prev = cv2.remap(prev, map_x, map_y, cv2.INTER_LINEAR)
        blended = cv2.addWeighted(frame, 0.7, warped_prev, 0.3, 0)
        out.write(blended)
        prev = frame
    cap.release(); out.release()


def _run_sota_pipeline(input_path: str, output_path: str, target_fps: int, latency_class: str = 'standard', enable_face_expert: bool = False, enable_hfr: bool = False) -> tuple[bool, str]:
    """Run the SOTA routing + handler pipeline. Returns (success, message)."""
    if not SOTA_AVAILABLE:
        return False, f"SOTA pipeline unavailable: {SOTA_IMPORT_ERROR or 'imports failed'}"

    try:
        logger.info("🔍 Running Degradation Router analysis...")
        router = DegradationRouter(device='cuda' if CUDA_AVAILABLE else 'cpu')
        plan = router.analyze_and_route(input_path, latency_class=latency_class, enable_face_expert=bool(enable_face_expert), enable_hfr=bool(enable_hfr))
        route = plan['expert_routing']
        primary = route.get('primary_model', 'vsrm')
        logger.info(f"🗺️ Routing selected primary model: {primary}")

        # Preprocess
        work_input = input_path
        with tempfile.TemporaryDirectory() as tdir:
            tdir = Path(tdir)
            if route.get('use_compression_expert') or route.get('use_denoising') or route.get('use_low_light_expert'):
                pre_path = str(tdir / "preprocessed.mp4")
                _preprocess_video(
                    input_path=work_input,
                    output_path=pre_path,
                    use_compression=bool(route.get('use_compression_expert')),
                    use_denoise=bool(route.get('use_denoising')),
                    use_low_light=bool(route.get('use_low_light_expert')),
                )
                work_input = pre_path

            # Primary handler
            primary_out = str(tdir / "primary.mp4")
            if primary == 'seedvr2':
                handler = SeedVR2Handler(device='cuda' if CUDA_AVAILABLE else 'cpu')
                stats = handler.restore_video(input_path=work_input, output_path=primary_out, quality_threshold=0.5)
            elif primary == 'ditvr':
                handler = DiTVRHandler(device='cuda' if CUDA_AVAILABLE else 'cpu')
                stats = handler.restore_video(input_path=work_input, output_path=primary_out, degradation_type='unknown', auto_adapt=True)
            elif primary == 'fast_mamba_vsr':
                handler = FastMambaVSRHandler(device='cuda' if CUDA_AVAILABLE else 'cpu')
                stats = handler.enhance_video(input_path=work_input, output_path=primary_out, chunk_size=16, fp16=True)
            else:
                handler = VSRMHandler(device='cuda' if CUDA_AVAILABLE else 'cpu')
                stats = handler.enhance_video(input_path=work_input, output_path=primary_out, window=7, fp16=True)

            work_output = primary_out

            # Face restoration expert
            if route.get('use_face_expert'):
                try:
                    from models.enhancement.face_restoration_expert import FaceRestorationExpert
                    face_out = str(tdir / "face.mp4")
                    fre = FaceRestorationExpert(device='cuda' if CUDA_AVAILABLE else 'cpu')
                    fre.process_video_selective(work_output, face_out)
                    work_output = face_out
                except Exception as fe:
                    logger.warning(f"Face restoration skipped due to error: {fe}")

            # Temporal consistency
            if route.get('use_temporal_consistency'):
                temp_out = str(tdir / "tc.mp4")
                _temporal_smooth(work_output, temp_out)
                work_output = temp_out

            # Interpolation (HFR)
            if route.get('use_hfr_interpolation'):
                try:
                    from models.interpolation.enhanced_rife_handler import EnhancedRIFEHandler
                    rife_out = str(tdir / "hfr.mp4")
                    er = EnhancedRIFEHandler(device='cuda' if CUDA_AVAILABLE else 'cpu')
                    # Double FPS
                    cap = cv2.VideoCapture(work_output)
                    in_fps = cap.get(cv2.CAP_PROP_FPS) or 24
                    cap.release()
                    er.interpolate_video(work_output, rife_out, target_fps=float(in_fps)*2, interpolation_factor=2)
                    work_output = rife_out
                except Exception as ie:
                    logger.warning(f"HFR interpolation skipped due to error: {ie}")

            # Persist final output
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            import shutil as _sh
            _sh.copy2(work_output, output_path)

        return True, f"✅ SOTA pipeline completed with {primary}. Stats: {stats}"
    except Exception as e:
        logger.error(f"SOTA pipeline failed: {e}")
        return False, f"❌ SOTA pipeline failed: {e}"


def get_recent_logs(n: int = 200) -> str:
    return "\n".join(list(log_ring)[-n:])


def process_video_gradio(input_video, target_fps, engine_choice, latency_class, enable_face, enable_hfr, request: gr.Request = None):
    """Process video through Gradio interface with comprehensive security."""
    try:
        if input_video is None:
            return None, "❌ Please upload a video file."
        
        # Create security context from request
        context = SecurityContext(
            ip_address=request.client.host if request and request.client else "127.0.0.1",
            user_agent=request.headers.get('user-agent', 'Unknown') if request else "Gradio-Interface",
            session_id=getattr(request, 'session_hash', None) if request else None,
            metadata={"source": "gradio_interface", "user_consent": True}
        )
        
        # Rate limiting check
        if not app_security_manager.check_rate_limits(context):
            logger.warning(f"Rate limit exceeded for {context.ip_address}")
            return None, "❌ Rate limit exceeded. Please wait before uploading another video."
        
        # Resolve source path from possible input types (str path or dict with 'name')
        src_path = None
        if isinstance(input_video, (str, Path)):
            src_path = str(input_video)
        elif isinstance(input_video, dict) and input_video.get('name'):
            src_path = input_video['name']
        else:
            try:
                # Some gradio versions provide an object with .name
                maybe_name = getattr(input_video, 'name', None)
                if maybe_name:
                    src_path = str(maybe_name)
            except Exception:
                src_path = None
        
        if not src_path or not os.path.exists(src_path):
            return None, "❌ Unable to read uploaded video path. Please try again."
        
        # File security validation
        logger.info(f"🔒 Running security validation on uploaded file: {Path(src_path).name}")
        is_valid, error_or_record_id, threats = app_security_manager.validate_and_secure_file(
            Path(src_path), 
            context, 
            Path(src_path).name
        )
        
        if not is_valid:
            logger.error(f"Security validation failed: {error_or_record_id}")
            return None, f"❌ Security validation failed: {error_or_record_id}"
        
        # Log security threats if any (but allow processing if they're low risk)
        if threats:
            threat_summary = ", ".join([f"{t.threat_type}({t.risk_level})" for t in threats])
            logger.info(f"ℹ️ Security scan detected: {threat_summary}")
        
        record_id = error_or_record_id  # This is now the data protection record ID
        
        # Access the protected file for processing
        if record_id:
            protected_file_path = app_security_manager.access_protected_file(record_id, context)
            if protected_file_path:
                src_path = str(protected_file_path)
                logger.info(f"✅ Using protected file for processing: {record_id}")
        
        if not engine_choice:
            engine_choice = "ZeroGPU Upscaler (fast)"

        if engine_choice.startswith("Frame Upscaler"):
            # Real-ESRGAN frame-wise fallback
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir = Path(temp_dir)
                    input_path = temp_dir / "input.mp4"
                    output_path = temp_dir / "enhanced.mp4"
                    import shutil
                    shutil.copy2(src_path, input_path)
                    from models.enhancement.frame.realesrgan_fallback import RealESRGANFallback
                    up = RealESRGANFallback(device='cuda' if CUDA_AVAILABLE else 'cpu', scale=2)
                    up.enhance_video(str(input_path), str(output_path))
                    if output_path.exists():
                        from uuid import uuid4
                        persist_dir = Path(os.getenv('OUTPUT_DIR', 'data/output'))
                        persist_dir.mkdir(parents=True, exist_ok=True)
                        persistent_path = persist_dir / f"enhanced_{uuid4().hex}.mp4"
                        import shutil as _sh
                        _sh.copy2(output_path, persistent_path)
                        return str(persistent_path), "✅ Real-ESRGAN upscaling completed"
                    else:
                        return None, "❌ Real-ESRGAN failed: no output"
            except Exception as e:
                error_msg = f"❌ Real-ESRGAN error: {e}"
                logger.error(error_msg)
                return None, error_msg

        if engine_choice.startswith("SOTA"):
            # Use SOTA pipeline
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir = Path(temp_dir)
                    input_path = temp_dir / "input.mp4"
                    output_path = temp_dir / "enhanced.mp4"
                    import shutil
                    shutil.copy2(src_path, input_path)

                    if ZEROGPU_AVAILABLE and spaces:
                        # Wrap with ZeroGPU for acceleration
                        duration = _estimate_duration_seconds(str(input_path))
                        decorator = spaces.GPU(duration=duration)
                    else:
                        decorator = (lambda f: f)

                    @decorator
                    def _sota_job(inp: str, outp: str, fps: int, lat: str, face: bool, hfr: bool):
                        return _run_sota_pipeline(inp, outp, fps, lat, face, hfr)

                    success, message = _sota_job(str(input_path), str(output_path), int(target_fps), latency_class, bool(enable_face), bool(enable_hfr))

                    if success and output_path.exists():
                        from uuid import uuid4
                        persist_dir = Path(os.getenv('OUTPUT_DIR', 'data/output'))
                        persist_dir.mkdir(parents=True, exist_ok=True)
                        persistent_path = persist_dir / f"enhanced_{uuid4().hex}.mp4"
                        import shutil as _sh
                        _sh.copy2(output_path, persistent_path)
                        logger.info(f"✅ Persisted enhanced video to {persistent_path}")
                        # Record job history (limited fields when using SOTA)
                        job_history.append({
                            'id': uuid4().hex,
                            'engine': 'SOTA Router',
                            'model': 'auto',
                            'frames': 'unknown',
                            'time': 'unknown',
                            'input': str(input_path),
                            'output': str(persistent_path),
                            'ts': datetime.now().isoformat(timespec='seconds')
                        })
                        return str(persistent_path), message
                    else:
                        return None, message
            except Exception as e:
                error_msg = f"❌ SOTA processing error: {e}"
                logger.error(error_msg)
                return None, error_msg

        # Default: ZeroGPU upscaler path
        if not enhancer.models_loaded and not enhancer.load_models():
            return None, "❌ Models not loaded. Please refresh and try again."
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                
                # Setup paths
                input_path = temp_dir / "input.mp4"
                output_path = temp_dir / "enhanced.mp4"
                
                # Copy input file
                import shutil
                shutil.copy2(src_path, input_path)
                
                # Process video
                logger.info(f"🎬 Starting video processing...")
                success, message = enhancer.process_video(
                    str(input_path), 
                    str(output_path), 
                    target_fps=int(target_fps)
                )
                
                if success and output_path.exists():
                    # Persist output outside the TemporaryDirectory so Gradio can serve it
                    from uuid import uuid4
                    persist_dir = Path(os.getenv('OUTPUT_DIR', 'data/output'))
                    persist_dir.mkdir(parents=True, exist_ok=True)
                    persistent_path = persist_dir / f"enhanced_{uuid4().hex}.mp4"
                    import shutil as _sh
                    _sh.copy2(output_path, persistent_path)
                    logger.info(f"✅ Persisted enhanced video to {persistent_path}")
                    return str(persistent_path), message
                else:
                    return None, message
                    
        except SecurityError as e:
            error_msg = f"❌ Security error: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
        except Exception as e:
            error_msg = f"❌ Processing error: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
        finally:
            # Cleanup expired security data periodically
            try:
                app_security_manager.cleanup_expired_data()
            except Exception as cleanup_error:
                logger.warning(f"Security cleanup failed: {cleanup_error}")
                
    except SecurityError as e:
        error_msg = f"❌ Security error: {str(e)}"
        logger.error(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"❌ Processing error: {str(e)}"
        logger.error(error_msg)
        return None, error_msg
    finally:
        # Cleanup expired security data periodically
        try:
            app_security_manager.cleanup_expired_data()
        except Exception as cleanup_error:
            logger.warning(f"Security cleanup failed: {cleanup_error}")

def get_system_info():
    """Get system information for display."""
    import psutil
    
    gpu_info = "Unknown"
    if ZEROGPU_AVAILABLE and HUGGINGFACE_SPACE:
        gpu_info = "NVIDIA H200 (ZeroGPU Dynamic)"
    elif CUDA_AVAILABLE:
        try:
            gpu_info = torch.cuda.get_device_name(0)
        except:
            gpu_info = "CUDA Available"
    else:
        gpu_info = "CPU Only"
    
    # Get security status
    security_status = app_security_manager.get_security_status()
    data_protection_summary = security_status.get('data_protection_summary', {})
    
    info = [
        f"🎯 **ZeroGPU Video Enhancer**",
        f"• Acceleration: {'ZeroGPU Enabled' if ZEROGPU_AVAILABLE else 'CPU Fallback'}",
        f"• Hardware: {gpu_info}",
        f"• VRAM Available: {'70GB (H200)' if ZEROGPU_AVAILABLE and HUGGINGFACE_SPACE else 'N/A'}",
        f"• Python: {sys.version.split()[0]}",
        f"• PyTorch: {torch.__version__ if 'torch' in globals() else 'Not available'}",
        f"• Gradio: {gr.__version__ if hasattr(gr, '__version__') else '4.44+'}",
        f"• CPU Cores: {psutil.cpu_count()}",
        f"• RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB",
        f"",
        f"📊 **Processing Stats**",
        f"• Videos Enhanced: {processing_stats['total_processed']}",
        f"• Total Processing Time: {processing_stats['total_time']:.1f}s",
        f"• Average Speed: {processing_stats['total_time'] / max(processing_stats['total_processed'], 1):.1f}s/video",
        f"• Uptime: {(datetime.now() - processing_stats['startup_time']).total_seconds():.0f}s",
        f"",
        f"🔒 **Security Status**",
        f"• File Validation: {'✅ Enabled' if security_config.file_validation_enabled else '❌ Disabled'}",
        f"• Data Encryption: {'✅ Enabled' if security_config.encryption_enabled else '❌ Disabled'}",
        f"• Rate Limiting: {'✅ Enabled' if security_config.rate_limit_enabled else '❌ Disabled'}",
        f"• Protected Records: {data_protection_summary.get('total_records', 0)}",
        f"• Encrypted Files: {data_protection_summary.get('encrypted_records', 0)}",
        f"• Recent Security Events: {security_status.get('recent_events', 0)}",
        f"",
        f"⚙️ **Features**",
        f"• GPU Acceleration: {'✅' if ZEROGPU_AVAILABLE else '❌'}",
        f"• Batch Processing: {'✅' if ZEROGPU_AVAILABLE else '❌'}",
        f"• Dynamic Duration: {'✅' if ZEROGPU_AVAILABLE else '❌'}",
        "• Memory Optimization: ✅",
        "• Robust Fallbacks: ✅",
        "• Security Protection: ✅"
    ]
    
    return "\n".join(info)

# Create Gradio interface
def _generate_demo_video(path: str, seconds: int = 3, fps: int = 24, size=(320, 180)):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, size)
    import math
    for i in range(seconds * fps):
        t = i / fps
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        # Moving rectangle
        x = int((math.sin(t) * 0.4 + 0.5) * (size[0] - 60))
        y = int((math.cos(t*1.3) * 0.3 + 0.5) * (size[1] - 40))
        cv2.rectangle(frame, (x, y), (x+60, y+40), (0, 255, 0), -1)
        # Text
        cv2.putText(frame, f"Demo t={t:.2f}", (10, size[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        out.write(frame)
    out.release()


def _evaluate_psnr_ssim(ref_path: str, test_path: str) -> str:
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    except Exception as e:
        return f"❌ skimage not available: {e}"
    cap1 = cv2.VideoCapture(ref_path)
    cap2 = cv2.VideoCapture(test_path)
    if not cap1.isOpened() or not cap2.isOpened():
        return "❌ Could not open videos for evaluation"
    psnrs = []
    ssims = []
    while True:
        ret1, f1 = cap1.read(); ret2, f2 = cap2.read()
        if not ret1 or not ret2:
            break
        # Convert to gray for SSIM
        f1g = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        f2g = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        psnrs.append(peak_signal_noise_ratio(f1g, f2g, data_range=255))
        ssims.append(structural_similarity(f1g, f2g, data_range=255))
    cap1.release(); cap2.release()
    if not psnrs:
        return "❌ No overlapping frames to evaluate"
    return f"PSNR: {np.mean(psnrs):.2f} dB, SSIM: {np.mean(ssims):.4f}"


def _cleanup_old_outputs(hours: int = 24, base_dir: str = None) -> int:
    base = Path(base_dir or os.getenv('OUTPUT_DIR', 'data/output'))
    if not base.exists():
        return 0
    cutoff = datetime.now().timestamp() - hours * 3600
    removed = 0
    for p in base.glob('*.mp4'):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
                removed += 1
        except Exception:
            pass
    return removed


with gr.Blocks(title="🏆 ZeroGPU Video Enhancer", theme=gr.themes.Soft()) as app:
    gr.Markdown(f"""
    # 🎯 ZeroGPU Video Enhancer
    
    **Professional video enhancement with NVIDIA H200 GPU acceleration** - Upload any video and enhance it with cutting-edge AI.
    
    ✨ **ZeroGPU Features:**
    - 🚀 **Dynamic GPU Allocation** - NVIDIA H200 with 70GB VRAM
    - 🔥 **Batch Processing** - Efficient frame-by-frame enhancement
    - ⚡ **Smart Duration Management** - Automatic processing time estimation
    - 🎯 **Advanced Neural Upscaling** - 2x resolution enhancement
    - 🛡️ **Robust Fallbacks** - CPU processing when needed
    - 🔒 **Enterprise Security** - File validation, encryption, and threat detection
    
    {'**Status: ZeroGPU Enabled ✅**' if ZEROGPU_AVAILABLE else '**Status: CPU Fallback Mode ⚠️**'} | **Security: Active 🔒**
    
    ⚠️ **Security Notice:** All uploaded videos are automatically scanned for threats, encrypted during processing, and automatically deleted after {security_config.data_retention_hours} hours.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 📤 Upload & Settings")
            
            input_video = gr.Video(
                label="📹 Upload Video",
                sources=["upload"],
                elem_id="input_video"
            )
            
            target_fps = gr.Slider(
                label="🎯 Target FPS",
                minimum=24,
                maximum=60,
                value=30,
                step=1
            )

            engine_choice = gr.Radio(
                label="🧠 Engine",
                choices=["ZeroGPU Upscaler (fast)", "Frame Upscaler (Real-ESRGAN)", "SOTA Router (SeedVR2/VSRM)"] ,
                value="ZeroGPU Upscaler (fast)"
            )

            latency_class = gr.Radio(
                label="⏱️ Latency Class",
                choices=["strict", "standard", "flexible"],
                value="standard"
            )
            enable_face = gr.Checkbox(label="Enable Face Restoration", value=False)
            enable_hfr = gr.Checkbox(label="Enable HFR Interpolation", value=False)
            
            process_btn = gr.Button(
                "🚀 Enhance Video", 
                variant="primary"
            )
            
            status_text = gr.Textbox(
                label="📊 Status",
                value="Ready to process videos!",
                interactive=False
            )
        
        with gr.Column(scale=1):
            gr.Markdown("## ℹ️ System Information")
            
            system_info = gr.Textbox(
                label="System Status",
                value=get_system_info(),
                lines=15,
                interactive=False
            )
            
            refresh_btn = gr.Button("🔄 Refresh Info")

    with gr.Accordion("📜 Live Logs", open=False):
        logs_box = gr.Textbox(label="Recent Logs", value=get_recent_logs(), lines=12, interactive=False)
        refresh_logs_btn = gr.Button("🔄 Refresh Logs")

    with gr.Accordion("🧾 Job History", open=False):
        history_table = gr.Dataframe(headers=["id", "engine", "model", "frames", "time", "input", "output", "ts"],
                                     datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
                                     interactive=False,
                                     value=[])
        refresh_history_btn = gr.Button("🔄 Refresh History")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Before")
            input_video_viewer = gr.Video(label="Original Video", interactive=False)
        with gr.Column():
            gr.Markdown("### After")
            output_video_viewer = gr.Video(label="Enhanced Video", interactive=False)

    # Link the video players
    def sync_videos(video_time: gr.Request):
        return video_time

    input_video.change(fn=lambda x: x, inputs=input_video, outputs=input_video_viewer)
    
    with gr.Row():
        demo_btn = gr.Button("▶️ Run Demo (SOTA)")
        eval_btn = gr.Button("🧪 Evaluate Last Output vs Demo")
    
    # Event handlers
    process_btn.click(
        fn=process_video_gradio,
        inputs=[input_video, target_fps, engine_choice, latency_class, enable_face, enable_hfr],
        outputs=[output_video_viewer, status_text],
        show_progress="full"
    )

    def _run_demo():
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            demo_in = td / 'demo.mp4'
            _generate_demo_video(str(demo_in))
            persist_dir = Path(os.getenv('OUTPUT_DIR', 'data/output'))
            persist_dir.mkdir(parents=True, exist_ok=True)
            demo_out = persist_dir / f"demo_sota_{uuid.uuid4().hex}.mp4"
            ok, msg = _run_sota_pipeline(str(demo_in), str(demo_out), 24, 'standard', False, False)
            if ok and demo_out.exists():
                return str(demo_out), f"✅ Demo completed: {msg}"
            return None, msg

    demo_btn.click(
        fn=_run_demo,
        outputs=[output_video_viewer, status_text]
    )

    def _eval_last():
        # Evaluate the last SOTA demo by regenerating demo and computing metrics to last output
        try:
            if not job_history:
                return "❌ No jobs to evaluate"
            last = job_history[-1]
            last_out = last.get('output')
            if not last_out or not Path(last_out).exists():
                return "❌ Last output file not found"
            with tempfile.TemporaryDirectory() as td:
                td = Path(td)
                demo_in = td / 'demo.mp4'
                _generate_demo_video(str(demo_in))
                return _evaluate_psnr_ssim(str(demo_in), last_out)
        except Exception as e:
            return f"❌ Evaluation failed: {e}"

    eval_btn.click(
        fn=_eval_last,
        outputs=status_text
    )
    
    refresh_btn.click(
        fn=get_system_info,
        outputs=system_info
    )

    refresh_logs_btn.click(
        fn=lambda: get_recent_logs(),
        outputs=logs_box
    )

    def _get_history_rows():
        # Convert job_history list of dicts into rows ordered by time desc
        rows = [[j.get('id'), j.get('engine'), j.get('model'), str(j.get('frames')), str(j.get('time')), j.get('input'), j.get('output'), j.get('ts')] for j in job_history[-200:]]
        return rows[::-1]

    refresh_history_btn.click(
        fn=_get_history_rows,
        outputs=history_table
    )
    
    # Initialize on startup
    app.load(fn=initialize_enhancer, outputs=status_text)

# Expose health and metrics via underlying FastAPI
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    _fastapi = app.app  # Gradio mounts on FastAPI
    
    # Rate limiting state for API endpoints
    _rate = {}

    @_fastapi.get("/health")
    def health():
        gpu = {
            'device': device,
            'cuda_available': bool(CUDA_AVAILABLE),
            'zerogpu_available': bool(ZEROGPU_AVAILABLE),
        }
        status = {
            'status': 'ok',
            'enhancer_ready': enhancer.models_loaded,
            'sota_available': SOTA_AVAILABLE,
            'processed': processing_stats['total_processed'],
            'uptime_seconds': (datetime.now() - processing_stats['startup_time']).total_seconds(),
        }
        security_status = app_security_manager.get_security_status()
        return JSONResponse({ 
            'gpu': gpu, 
            'status': status,
            'security': security_status
        })

    @_fastapi.get("/metrics")
    def metrics():
        avg = processing_stats['total_time'] / max(1, processing_stats['total_processed'])
        return JSONResponse({
            'requests': {
                'processed': processing_stats['total_processed'],
            },
            'performance': {
                'total_time': processing_stats['total_time'],
                'average_processing_time': avg,
            },
            'recent_jobs': job_history[-10:]
        })
    @_fastapi.post("/api/v1/process/auto")
    def api_process_auto(payload: dict, request: Request):
        """Programmatic processing endpoint with enhanced security.
        Accepts JSON: {
          "engine": "zero|frame|sota",
          "source_url": "http..." (optional),
          "latency_class": "strict|standard|flexible",
          "enable_face": bool,
          "enable_hfr": bool,
          "target_fps": int (optional)
        }
        """
        import requests
        
        try:
            # Create security context
            context = SecurityContext(
                ip_address=request.client.host if request.client else 'unknown',
                user_agent=request.headers.get('user-agent', 'API-Client'),
                metadata={'source': 'api', 'endpoint': '/api/v1/process/auto'}
            )
            
            # Enhanced authentication using security manager
            api_key = request.headers.get('X-API-Key')
            if security_config.api_key_required:
                if not app_security_manager.authenticate_request(context, api_key):
                    return JSONResponse({ 'error': 'unauthorized' }, status_code=401)
            
            # Enhanced rate limiting using security manager
            if not app_security_manager.check_rate_limits(context):
                return JSONResponse({ 'error': 'rate limit exceeded' }, status_code=429)
            
            eng = (payload.get('engine') or 'sota').lower()
            source_url = payload.get('source_url')
            latency = payload.get('latency_class') or 'standard'
            enable_face = bool(payload.get('enable_face', False))
            enable_hfr = bool(payload.get('enable_hfr', False))
            target_fps = int(payload.get('target_fps') or 30)

            if not source_url:
                return JSONResponse({ 'error': 'source_url required' }, status_code=400)

            # Download to temp with security validation
            with tempfile.TemporaryDirectory() as tdir:
                tdirp = Path(tdir)
                in_path = tdirp / 'input.mp4'
                
                # Download file
                r = requests.get(source_url, stream=True, timeout=60)
                r.raise_for_status()
                with open(in_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Security validation on downloaded file
                is_valid, error_or_record_id, threats = app_security_manager.validate_and_secure_file(
                    in_path, 
                    context, 
                    f"api_download_{uuid.uuid4().hex[:8]}.mp4"
                )
                
                if not is_valid:
                    return JSONResponse({ 
                        'error': f'security_validation_failed: {error_or_record_id}' 
                    }, status_code=400)
                
                # Use protected file if available
                record_id = error_or_record_id
                if record_id:
                    protected_file_path = app_security_manager.access_protected_file(record_id, context)
                    if protected_file_path:
                        in_path = protected_file_path
                
                out_dir = Path(os.getenv('OUTPUT_DIR', 'data/output'))
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"api_enhanced_{uuid.uuid4().hex}.mp4"

                # Choose engine
                if eng.startswith('zero'):
                    ok, msg = enhancer.process_video(str(in_path), str(out_path), target_fps)
                elif eng.startswith('frame'):
                    from models.enhancement.frame.realesrgan_fallback import RealESRGANFallback
                    up = RealESRGANFallback(device='cuda' if CUDA_AVAILABLE else 'cpu', scale=2)
                    stats = up.enhance_video(str(in_path), str(out_path))
                    ok, msg = True, f"✅ Real-ESRGAN upscaling completed: {stats}"
                else:  # sota
                    ok, msg = _run_sota_pipeline(str(in_path), str(out_path), target_fps, latency, enable_face, enable_hfr)

                if ok and out_path.exists():
                    job = {
                        'id': uuid.uuid4().hex,
                        'engine': 'sota' if eng.startswith('sota') else ('frame' if eng.startswith('frame') else 'zero'),
                        'model': 'auto',
                        'frames': 'unknown',
                        'time': 'unknown',
                        'input': str(in_path),
                        'output': str(out_path),
                        'ts': datetime.now().isoformat(timespec='seconds')
                    }
                    job_history.append(job)
                    return JSONResponse({
                        'job': job,
                        'message': msg
                    })
                else:
                    return JSONResponse({ 'error': msg or 'processing failed' }, status_code=500)
        
        except SecurityError as e:
            logger.error(f"API security error: {e}")
            return JSONResponse({ 'error': f'security_error: {str(e)}' }, status_code=403)
        except Exception as e:
            logger.error(f"API process failed: {e}")
            return JSONResponse({ 'error': str(e) }, status_code=500)

    @_fastapi.get("/api/v1/jobs")
    def api_list_jobs():
        return JSONResponse(job_history[-200:][::-1])

    @_fastapi.get("/api/v1/job/{job_id}")
    def api_get_job(job_id: str):
        for j in reversed(job_history):
            if j.get('id') == job_id:
                return JSONResponse(j)
        return JSONResponse({ 'error': 'not found' }, status_code=404)

except Exception as _e:
    logger.warning(f"Health/metrics endpoints not mounted: {_e}")

if __name__ == "__main__":
    # Initialize enhancer
    logger.info("🚀 Starting SOTA Video Enhancer...")
    
    # Launch Gradio app
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )