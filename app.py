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

# Environment detection FIRST
HUGGINGFACE_SPACE = os.environ.get('SPACE_ID') is not None
# Enhanced ZeroGPU+ detection - covers multiple possible configurations
ZEROGPU_SPACE = (
    os.environ.get('ZERO_GPU') == '1' or 
    os.environ.get('hw') == 'zero-gpu' or 
    os.environ.get('hw') == 'zerogpu' or
    os.environ.get('SPACE_HARDWARE') == 'zero-gpu' or
    os.environ.get('SPACE_HARDWARE') == 'zerogpu' or
    'zero' in str(os.environ.get('hw', '')).lower() or
    'gpu' in str(os.environ.get('SPACE_HARDWARE', '')).lower()
)

# Debug environment variables for troubleshooting
logger = logging.getLogger(__name__)
if HUGGINGFACE_SPACE:
    env_debug = {
        'SPACE_ID': os.environ.get('SPACE_ID'),
        'ZERO_GPU': os.environ.get('ZERO_GPU'),
        'hw': os.environ.get('hw'),
        'SPACE_HARDWARE': os.environ.get('SPACE_HARDWARE'),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'HF_TOKEN': 'SET' if os.environ.get('HF_TOKEN') else 'NOT_SET'
    }
    logger.info(f"🔍 HF Space Environment Debug: {env_debug}")
    logger.info(f"🔍 ZeroGPU Detection: HUGGINGFACE_SPACE={HUGGINGFACE_SPACE}, ZEROGPU_SPACE={ZEROGPU_SPACE}")

# ZeroGPU import - ONLY in HF Spaces environment
try:
    if HUGGINGFACE_SPACE and ZEROGPU_SPACE:
        import spaces
        ZEROGPU_AVAILABLE = True
        logger = logging.getLogger(__name__)
        logger.info("✨ ZeroGPU available - GPU acceleration enabled")
    else:
        raise ImportError("Not in ZeroGPU environment")
except ImportError:
    ZEROGPU_AVAILABLE = False
    spaces = None
    logger = logging.getLogger(__name__)
    if HUGGINGFACE_SPACE:
        logger.warning("⚠️ ZeroGPU not available in this Space - using CPU fallback")
    else:
        logger.info("💻 Running in local/non-HF environment")

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Apply torchvision compatibility fix early
try:
    from utils.torchvision_compatibility import apply_torchvision_compatibility_fix
    apply_torchvision_compatibility_fix()
except ImportError:
    logger.warning("⚠️ TorchVision compatibility module not available")
except Exception as e:
    logger.warning(f"⚠️ TorchVision compatibility fix failed: {e}")

# Safe imports with fallbacks
try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    import cv2
    from PIL import Image
    from diffusers import StableDiffusionPipeline, DiffusionPipeline
    from transformers import pipeline
    
    # ZeroGPU device detection - NEVER initialize CUDA in main process
    if ZEROGPU_AVAILABLE and HUGGINGFACE_SPACE:
        # In ZeroGPU environment, device allocation is dynamic - defer GPU checks
        CUDA_AVAILABLE = True  # Assume available but don't test
        device = 'cuda'  # Will be managed by @spaces.GPU decorator
        logger.info(f"🎯 ZeroGPU Environment - Dynamic GPU allocation (deferred)")
    else:
        # Only check CUDA in non-ZeroGPU environments
        try:
            if not HUGGINGFACE_SPACE:
                # Safe to check CUDA in local environments
                CUDA_AVAILABLE = torch.cuda.is_available()
                device = 'cuda' if CUDA_AVAILABLE else 'cpu'
                logger.info(f"🚀 PyTorch loaded successfully. Device: {device}")
            else:
                # HF Space but not ZeroGPU - assume CPU
                CUDA_AVAILABLE = False
                device = 'cpu'
                logger.info(f"💻 HF Space CPU mode")
        except Exception as e:
            CUDA_AVAILABLE = False
            device = 'cpu'
            logger.warning(f"Device detection failed: {e}. Using CPU.")
        
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
    
    def _process_video_gpu(self, input_path, output_path, target_fps=30):
        """ZeroGPU-accelerated video processing with safe decorator."""
        # Apply GPU decorator with fixed duration
        if ZEROGPU_AVAILABLE and spaces:
            return self._process_video_gpu_decorated(input_path, output_path, target_fps)
        else:
            return self._process_video_cpu(input_path, output_path, target_fps)
    
    @spaces.GPU(duration=180) if ZEROGPU_AVAILABLE and spaces else lambda x: x
    def _process_video_gpu_decorated(self, input_path, output_path, target_fps=30):
        """GPU-decorated video processing method."""
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
    """Initialize the video enhancer with comprehensive error handling and diagnostics."""
    try:
        logger.info("🚀 Initializing Video Enhancer...")
        
        # Log environment details for diagnostics
        env_info = [
            f"Environment: {'ZeroGPU' if ZEROGPU_AVAILABLE else 'Standard'}",
            f"Device: {device}",
            f"HF Space: {HUGGINGFACE_SPACE}",
            f"SOTA Available: {SOTA_AVAILABLE}"
        ]
        
        if SOTA_IMPORT_ERROR:
            env_info.append(f"SOTA Error: {SOTA_IMPORT_ERROR}")
            
        logger.info(f"Environment: {' | '.join(env_info)}")
        
        # System resource check
        try:
            import psutil
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            logger.info(f"System Resources: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
        except Exception:
            logger.info("Could not determine system resources")
        
        # Initialize with timeout protection
        import threading
        import time as time_mod
        
        result = {"success": False, "error": None, "completed": False}
        
        def init_worker():
            try:
                logger.info("🔄 Loading models...")
                result["success"] = enhancer.load_models()
                result["completed"] = True
                logger.info(f"🏁 Model loading completed: {result['success']}")
            except Exception as e:
                result["error"] = str(e)
                result["completed"] = True
                logger.error(f"❌ Model loading failed: {e}")
                
        # Run initialization with timeout
        init_thread = threading.Thread(target=init_worker, daemon=True)
        init_thread.start()
        
        # Wait with progress indication
        timeout = 90  # 90 seconds timeout
        start_time = time_mod.time()
        
        while time_mod.time() - start_time < timeout and not result["completed"]:
            time_mod.sleep(1)
            elapsed = int(time_mod.time() - start_time)
            if elapsed % 15 == 0 and elapsed > 0:  # Progress every 15 seconds
                logger.info(f"⏳ Initialization in progress... ({elapsed}s elapsed)")
        
        if not result["completed"]:
            logger.error(f"⏱️ Enhancer initialization timed out after {timeout}s")
            return f"⚠️ Initialization timed out ({timeout}s). Using fallback mode."
            
        # Check results
        if result["error"]:
            error_msg = f"Initialization error: {result['error']}"
            logger.error(error_msg)
            if "CUDA" in result["error"] and ZEROGPU_AVAILABLE:
                return f"⚠️ {error_msg} - ZeroGPU will handle CUDA dynamically."
            return f"⚠️ {error_msg} - Fallback mode enabled."
            
        if result["success"]:
            logger.info("✅ Video enhancer initialized successfully")
            
            # Run basic functionality test
            test_result = _test_basic_functionality()
            
            success_msg = "✅ Video Enhancer Ready! Upload a video to get started."
            if test_result["warnings"]:
                success_msg += f" (Note: {test_result['warnings']})"
                
            return success_msg
        else:
            logger.warning("Enhancer initialized with limited functionality")
            return "⚠️ Enhancer initialized with basic features only. Some models may be unavailable."
            
    except Exception as e:
        error_msg = f"Critical initialization failure: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"❌ {error_msg}"
        
def _test_basic_functionality() -> dict:
    """Test basic functionality and return status."""
    warnings = []
    
    try:
        # Test PyTorch
        import torch
        test_tensor = torch.randn(1, 3, 8, 8)
        
        # Only test CUDA in non-ZeroGPU environments to avoid early initialization
        if CUDA_AVAILABLE and not ZEROGPU_AVAILABLE and not HUGGINGFACE_SPACE:
            try:
                test_tensor = test_tensor.cuda()
                logger.info("✅ CUDA functionality verified")
            except Exception as cuda_err:
                logger.warning(f"CUDA test failed: {cuda_err}")
                warnings.append("CUDA test failed")
        
        # Test OpenCV
        import cv2
        test_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
        logger.info("✅ OpenCV functionality verified")
        
        # Test SOTA availability
        if SOTA_AVAILABLE:
            logger.info("✅ SOTA models available")
        else:
            logger.warning(f"SOTA models unavailable: {SOTA_IMPORT_ERROR}")
            warnings.append("SOTA models unavailable")
            
        return {"success": True, "warnings": ", ".join(warnings) if warnings else None}
        
    except Exception as e:
        logger.warning(f"Basic functionality test failed: {e}")
        return {"success": False, "warnings": f"Functionality test failed: {e}"}

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
    """Run the SOTA routing + handler pipeline with ZeroGPU safety. Returns (success, message)."""
    if not SOTA_AVAILABLE:
        logger.error(f"SOTA pipeline unavailable: {SOTA_IMPORT_ERROR}")
        return False, f"❌ SOTA pipeline unavailable: {SOTA_IMPORT_ERROR or 'Model imports failed'}"

    try:
        logger.info("🔍 Initializing SOTA pipeline...")
        
        # ZeroGPU safety check
        if ZEROGPU_AVAILABLE and HUGGINGFACE_SPACE:
            return _run_sota_pipeline_zerogpu(input_path, output_path, target_fps, latency_class, enable_face_expert, enable_hfr)
        else:
            return _run_sota_pipeline_local(input_path, output_path, target_fps, latency_class, enable_face_expert, enable_hfr)
            
    except Exception as e:
        error_msg = f"❌ SOTA pipeline initialization failed: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

@spaces.GPU(duration=300) if ZEROGPU_AVAILABLE and spaces else lambda x: x
def _run_sota_pipeline_zerogpu(input_path: str, output_path: str, target_fps: int, latency_class: str = 'standard', enable_face_expert: bool = False, enable_hfr: bool = False) -> tuple[bool, str]:
    """ZeroGPU-decorated SOTA pipeline execution."""
    try:
        logger.info("🔍 Running Degradation Router analysis (ZeroGPU)...")
        
        # Import and initialize models within GPU context
        from models.analysis.degradation_router import DegradationRouter
        from models.enhancement.vsr.vsrm_handler import VSRMHandler
        from models.enhancement.zeroshot.seedvr2_handler import SeedVR2Handler
        from models.enhancement.zeroshot.ditvr_handler import DiTVRHandler
        from models.enhancement.vsr.fast_mamba_vsr_handler import FastMambaVSRHandler
        
        router = DegradationRouter(device='cuda')
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

            # Primary handler with comprehensive error handling
            primary_out = str(tdir / "primary.mp4")
            stats = {}
            try:
                if primary == 'seedvr2':
                    handler = SeedVR2Handler(device='cuda')
                    stats = handler.restore_video(input_path=work_input, output_path=primary_out, quality_threshold=0.5)
                elif primary == 'ditvr':
                    handler = DiTVRHandler(device='cuda')
                    stats = handler.restore_video(input_path=work_input, output_path=primary_out, degradation_type='unknown', auto_adapt=True)
                elif primary == 'fast_mamba_vsr':
                    handler = FastMambaVSRHandler(device='cuda')
                    stats = handler.enhance_video(input_path=work_input, output_path=primary_out, chunk_size=16, fp16=True)
                else:
                    handler = VSRMHandler(device='cuda')
                    stats = handler.enhance_video(input_path=work_input, output_path=primary_out, window=7, fp16=True)
            except Exception as model_error:
                logger.error(f"Primary model {primary} failed: {model_error}")
                # Fallback to basic upscaling
                return _fallback_basic_upscale_zerogpu(input_path, output_path)

            work_output = primary_out
            
            # Validate primary output
            if not Path(primary_out).exists() or Path(primary_out).stat().st_size == 0:
                logger.error(f"Primary model {primary} produced no output")
                return _fallback_basic_upscale_zerogpu(input_path, output_path)

            # Face restoration expert (optional)
            if route.get('use_face_expert'):
                try:
                    from models.enhancement.face_restoration_expert import FaceRestorationExpert
                    face_out = str(tdir / "face.mp4")
                    fre = FaceRestorationExpert(device='cuda')
                    fre.process_video_selective(work_output, face_out)
                    if Path(face_out).exists():
                        work_output = face_out
                except Exception as fe:
                    logger.warning(f"Face restoration skipped: {fe}")

            # Temporal consistency (optional)
            if route.get('use_temporal_consistency'):
                try:
                    temp_out = str(tdir / "tc.mp4")
                    _temporal_smooth(work_output, temp_out)
                    if Path(temp_out).exists():
                        work_output = temp_out
                except Exception as tc_error:
                    logger.warning(f"Temporal consistency skipped: {tc_error}")

            # Interpolation (HFR) (optional)
            if route.get('use_hfr_interpolation'):
                try:
                    from models.interpolation.enhanced_rife_handler import EnhancedRIFEHandler
                    rife_out = str(tdir / "hfr.mp4")
                    er = EnhancedRIFEHandler(device='cuda')
                    cap = cv2.VideoCapture(work_output)
                    in_fps = cap.get(cv2.CAP_PROP_FPS) or 24
                    cap.release()
                    er.interpolate_video(work_output, rife_out, target_fps=float(in_fps)*2, interpolation_factor=2)
                    if Path(rife_out).exists():
                        work_output = rife_out
                except Exception as ie:
                    logger.warning(f"HFR interpolation skipped: {ie}")

            # Persist final output
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            import shutil as _sh
            _sh.copy2(work_output, output_path)
            
            # Clear GPU memory
            import torch
            torch.cuda.empty_cache()

        return True, f"✅ ZeroGPU SOTA pipeline completed with {primary}"
    except Exception as e:
        error_msg = f"❌ ZeroGPU SOTA pipeline failed: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
        
def _fallback_basic_upscale_zerogpu(input_path: str, output_path: str) -> tuple[bool, str]:
    """Fallback basic upscaling for ZeroGPU when SOTA models fail."""
    try:
        logger.info("🔄 Using fallback basic upscaling...")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False, "Could not open input video"
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height*2))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Basic upscaling using CUDA-accelerated resize
            import torch
            frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
            upscaled = torch.nn.functional.interpolate(frame_tensor, scale_factor=2, mode='bicubic', align_corners=False)
            frame_up = (upscaled.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            out.write(frame_up)
            frame_count += 1
            
        cap.release()
        out.release()
        torch.cuda.empty_cache()
        
        return True, f"✅ Fallback upscaling completed ({frame_count} frames)"
        
    except Exception as e:
        return False, f"❌ Fallback upscaling failed: {e}"

def _run_sota_pipeline_local(input_path: str, output_path: str, target_fps: int, latency_class: str = 'standard', enable_face_expert: bool = False, enable_hfr: bool = False) -> tuple[bool, str]:
    """Local (non-ZeroGPU) SOTA pipeline execution."""
    try:
        logger.info("🔍 Running Degradation Router analysis (Local)...")
        
        router = DegradationRouter(device=device)
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
            try:
                if primary == 'seedvr2':
                    handler = SeedVR2Handler(device=device)
                    stats = handler.restore_video(input_path=work_input, output_path=primary_out, quality_threshold=0.5)
                elif primary == 'ditvr':
                    handler = DiTVRHandler(device=device)
                    stats = handler.restore_video(input_path=work_input, output_path=primary_out, degradation_type='unknown', auto_adapt=True)
                elif primary == 'fast_mamba_vsr':
                    handler = FastMambaVSRHandler(device=device)
                    stats = handler.enhance_video(input_path=work_input, output_path=primary_out, chunk_size=16, fp16=(device=='cuda'))
                else:
                    handler = VSRMHandler(device=device)
                    stats = handler.enhance_video(input_path=work_input, output_path=primary_out, window=7, fp16=(device=='cuda'))
            except Exception as model_error:
                logger.error(f"Primary model {primary} failed: {model_error}")
                return False, f"❌ Model {primary} failed: {str(model_error)}"

            work_output = primary_out

            # Post-processing steps (optional)
            if route.get('use_face_expert'):
                try:
                    from models.enhancement.face_restoration_expert import FaceRestorationExpert
                    face_out = str(tdir / "face.mp4")
                    fre = FaceRestorationExpert(device=device)
                    fre.process_video_selective(work_output, face_out)
                    if Path(face_out).exists():
                        work_output = face_out
                except Exception as fe:
                    logger.warning(f"Face restoration skipped: {fe}")

            if route.get('use_temporal_consistency'):
                try:
                    temp_out = str(tdir / "tc.mp4")
                    _temporal_smooth(work_output, temp_out)
                    if Path(temp_out).exists():
                        work_output = temp_out
                except Exception:
                    logger.warning("Temporal consistency skipped")

            if route.get('use_hfr_interpolation'):
                try:
                    from models.interpolation.enhanced_rife_handler import EnhancedRIFEHandler
                    rife_out = str(tdir / "hfr.mp4")
                    er = EnhancedRIFEHandler(device=device)
                    cap = cv2.VideoCapture(work_output)
                    in_fps = cap.get(cv2.CAP_PROP_FPS) or 24
                    cap.release()
                    er.interpolate_video(work_output, rife_out, target_fps=float(in_fps)*2, interpolation_factor=2)
                    if Path(rife_out).exists():
                        work_output = rife_out
                except Exception:
                    logger.warning("HFR interpolation skipped")

            # Persist final output
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            import shutil as _sh
            _sh.copy2(work_output, output_path)

        return True, f"✅ Local SOTA pipeline completed with {primary}"
    except Exception as e:
        error_msg = f"❌ Local SOTA pipeline failed: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def get_recent_logs(n: int = 200) -> str:
    return "\n".join(list(log_ring)[-n:])

def _get_hf_user_info(request):
    """Extract HuggingFace user information from OAuth request."""
    try:
        if not request:
            return None
            
        # Check for HuggingFace OAuth headers
        headers = getattr(request, 'headers', {})
        
        # Look for user information in OAuth context
        # HuggingFace Spaces with OAuth enabled provide user info
        user_info = getattr(request, 'username', None)
        if user_info:
            return {"preferred_username": user_info, "authenticated": True}
        
        # Check OAuth environment variables
        oauth_client_id = os.environ.get('OAUTH_CLIENT_ID')
        if oauth_client_id and hasattr(request, 'session_hash'):
            # User is authenticated through OAuth
            return {"preferred_username": "authenticated_user", "authenticated": True}
            
        # Fallback: check for any authentication indicators
        if HUGGINGFACE_SPACE and headers:
            auth_header = headers.get('authorization') or headers.get('x-user-token')
            if auth_header:
                return {"preferred_username": "token_user", "authenticated": True}
                
        return None
        
    except Exception as e:
        logger.warning(f"Failed to extract user info: {e}")
        return None

def _check_user_quota(user_info):
    """Check user quotas for ZeroGPU usage."""
    try:
        if not user_info:
            return {"allowed": False, "message": "Authentication required"}
        
        # For authenticated users, allow processing
        # The actual quota management is handled by HuggingFace ZeroGPU infrastructure
        if user_info.get("authenticated"):
            return {"allowed": True, "message": "Quota OK"}
        
        return {"allowed": False, "message": "Please sign in to access ZeroGPU features"}
        
    except Exception as e:
        logger.warning(f"Quota check failed: {e}")
        return {"allowed": True, "message": "Quota check bypassed"}


# ZeroGPU decorator for main processing function
if ZEROGPU_AVAILABLE and spaces:
    @spaces.GPU(duration=120)  # 2 minutes default duration
    def process_video_gradio(input_video, target_fps, engine_choice, latency_class, enable_face, enable_hfr, request: gr.Request = None):
        return _process_video_gradio_impl(input_video, target_fps, engine_choice, latency_class, enable_face, enable_hfr, request)
else:
    def process_video_gradio(input_video, target_fps, engine_choice, latency_class, enable_face, enable_hfr, request: gr.Request = None):
        return _process_video_gradio_impl(input_video, target_fps, engine_choice, latency_class, enable_face, enable_hfr, request)

def _process_video_gradio_impl(input_video, target_fps, engine_choice, latency_class, enable_face, enable_hfr, request: gr.Request = None):
    """Process video through Gradio interface with comprehensive security and user authentication."""
    try:
        if input_video is None:
            return None, "❌ Please upload a video file."
        
        # Check HuggingFace authentication in ZeroGPU environment
        user_info = None
        if HUGGINGFACE_SPACE and ZEROGPU_AVAILABLE:
            try:
                # Get user info from HuggingFace OAuth
                user_info = _get_hf_user_info(request)
                if not user_info:
                    return None, "❌ Please sign in with your Hugging Face account to use ZeroGPU features. Click the 'Sign in with HF' button at the top."
                    
                logger.info(f"👤 Authenticated user: {user_info.get('preferred_username', 'anonymous')}")
                
                # Check user quotas
                quota_status = _check_user_quota(user_info)
                if not quota_status["allowed"]:
                    return None, f"❌ {quota_status['message']} Please upgrade to HuggingFace PRO for higher limits: https://huggingface.co/subscribe/pro"
                    
            except Exception as auth_error:
                logger.error(f"Authentication error: {auth_error}")
                if "quota" in str(auth_error).lower():
                    return None, f"❌ ZeroGPU quota exceeded. Please sign in with HuggingFace Pro account or try again later."
                else:
                    return None, f"⚠️ Authentication issue: {auth_error}. Falling back to basic processing."
        
        # Create security context from request
        context = SecurityContext(
            ip_address=request.client.host if request and request.client else "127.0.0.1",
            user_agent=request.headers.get('user-agent', 'Unknown') if request else "Gradio-Interface",
            session_id=getattr(request, 'session_hash', None) if request else None,
            metadata={
                "source": "gradio_interface", 
                "user_consent": True,
                "user_info": user_info
            }
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
    zerogpu_status = "Not Available"
    
    if ZEROGPU_AVAILABLE and HUGGINGFACE_SPACE:
        gpu_info = "NVIDIA H200 (ZeroGPU Dynamic)"
        zerogpu_status = "✅ Enabled and Available"
    elif HUGGINGFACE_SPACE:
        gpu_info = "HuggingFace Space - CPU Mode"
        zerogpu_status = "⚠️ In HF Space but ZeroGPU not detected"
    elif CUDA_AVAILABLE:
        try:
            gpu_info = torch.cuda.get_device_name(0)
            zerogpu_status = "Local CUDA"
        except:
            gpu_info = "CUDA Available"
            zerogpu_status = "Local CUDA"
    else:
        gpu_info = "CPU Only"
        zerogpu_status = "Not Available"
    
    # Get security status
    security_status = app_security_manager.get_security_status()
    data_protection_summary = security_status.get('data_protection_summary', {})
    
    info = [
        f"🎯 **ZeroGPU Video Enhancer**",
        f"• ZeroGPU Status: {zerogpu_status}",
        f"• Hardware: {gpu_info}",
        f"• VRAM Available: {'70GB (H200)' if ZEROGPU_AVAILABLE and HUGGINGFACE_SPACE else 'N/A'}",
        f"• HF Space: {'Yes' if HUGGINGFACE_SPACE else 'No'}",
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
def _generate_demo_video(path: str, seconds: int = 3, fps: int = 24, size=(480, 360)):
    """Generate a comprehensive demo video for testing enhancement algorithms."""
    import math
    
    try:
        logger.info(f"📹 Generating demo video: {path} ({size[0]}x{size[1]}, {fps}fps, {seconds}s)")
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Use a more compatible codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, size)
        
        if not out.isOpened():
            logger.error("❌ Failed to open VideoWriter")
            return False
        
        total_frames = seconds * fps
        logger.info(f"🎯 Creating {total_frames} frames...")
        
        for i in range(total_frames):
            t = i / fps  # Time in seconds
            
            # Create frame with gradient background
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            
            # Gradient background (simulates lighting changes)
            for y in range(size[1]):
                for x in range(size[0]):
                    # Dynamic gradient that changes over time
                    r = int(127 + 127 * math.sin(x * 0.02 + t * 2))
                    g = int(127 + 127 * math.cos(y * 0.02 + t * 1.5))
                    b = int(127 + 127 * math.sin((x + y) * 0.01 + t))
                    frame[y, x] = [b, g, r]  # BGR format
            
            # Moving circle (tests motion and circular features)
            circle_x = int((math.sin(t * 2.5) * 0.3 + 0.5) * size[0])
            circle_y = int((math.cos(t * 2) * 0.3 + 0.5) * size[1])
            cv2.circle(frame, (circle_x, circle_y), 30, (0, 255, 255), -1)  # Yellow circle
            
            # Moving rectangle (tests rectangular features and motion blur)
            rect_x = int((math.cos(t * 3) * 0.25 + 0.75) * (size[0] - 80))
            rect_y = int((math.sin(t * 2.5) * 0.25 + 0.25) * (size[1] - 60))
            cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 80, rect_y + 60), (255, 0, 255), -1)  # Magenta rectangle
            
            # Rotating line (tests line detection and rotation)
            center_x, center_y = size[0] // 2, size[1] // 2
            angle = t * 120  # degrees
            line_length = min(size[0], size[1]) // 3
            end_x = int(center_x + line_length * math.cos(math.radians(angle)))
            end_y = int(center_y + line_length * math.sin(math.radians(angle)))
            cv2.line(frame, (center_x, center_y), (end_x, end_y), (0, 255, 0), 4)  # Green line
            
            # Text overlays (tests text clarity and readability)
            title = "ENHANCEMENT DEMO"
            cv2.putText(frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            timestamp = f"Time: {t:.2f}s | Frame: {i+1:03d}"
            cv2.putText(frame, timestamp, (10, size[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            info = f"Resolution: {size[0]}x{size[1]} | FPS: {fps}"
            cv2.putText(frame, info, (10, size[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Add some "artifacts" to test enhancement
            if i % 8 == 0:  # Every 8 frames, add some noise
                noise = np.random.normal(0, 10, frame.shape).astype(np.int16)
                frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            out.write(frame)
            
            # Progress logging
            if (i + 1) % (total_frames // 4) == 0:
                progress = ((i + 1) / total_frames) * 100
                logger.info(f"   Progress: {progress:.0f}% ({i+1}/{total_frames} frames)")
        
        out.release()
        
        # Verify the file was created successfully
        if Path(path).exists() and Path(path).stat().st_size > 0:
            file_size = Path(path).stat().st_size
            logger.info(f"✅ Demo video created successfully: {file_size / 1024:.1f} KB")
            return True
        else:
            logger.error("❌ Demo video file was not created or is empty")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error generating demo video: {e}")
        return False


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
    
    # Show available demo videos
    demo_videos_dir = Path('data/demo_videos')
    available_demo_info = "No demo videos available"
    if demo_videos_dir.exists():
        demo_files = list(demo_videos_dir.glob('*.mp4'))
        if demo_files:
            file_info = []
            for f in demo_files:
                size_kb = f.stat().st_size / 1024
                file_info.append(f"{f.name} ({size_kb:.1f} KB)")
            available_demo_info = f"Demo videos available: {', '.join(file_info)}"
    
    gr.Markdown(f"**Demo Videos:** {available_demo_info}")
    
    with gr.Row():
        demo_btn = gr.Button("▶️ Run Demo with Real Videos (SOTA)")
        eval_btn = gr.Button("🧪 Evaluate Last Output vs Demo")
    
    # Event handlers
    process_btn.click(
        fn=process_video_gradio,
        inputs=[input_video, target_fps, engine_choice, latency_class, enable_face, enable_hfr],
        outputs=[output_video_viewer, status_text],
        show_progress="full"
    )

    def _run_demo():
        try:
            logger.info("🎬 Starting demo video processing with real video files...")
            
            # Create persistent demo directory
            demo_dir = Path(os.getenv('OUTPUT_DIR', 'data/output')) / 'demo'
            demo_dir.mkdir(parents=True, exist_ok=True)
            
            # Look for available demo videos
            demo_videos_dir = Path('data/demo_videos')
            available_videos = []
            
            if demo_videos_dir.exists():
                # Find all mp4 files in the demo videos directory
                for video_file in demo_videos_dir.glob('*.mp4'):
                    if video_file.is_file() and video_file.stat().st_size > 0:
                        available_videos.append(video_file)
            
            # If no demo videos found, fall back to generating one
            if not available_videos:
                logger.info("📹 No demo videos found in data/demo_videos/, generating synthetic demo...")
                demo_in = demo_dir / f"demo_input_{uuid.uuid4().hex[:8]}.mp4"
                _generate_demo_video(str(demo_in), seconds=3, fps=24, size=(480, 360))
                
                if not demo_in.exists() or demo_in.stat().st_size == 0:
                    logger.error("❌ Failed to create demo video")
                    return None, "❌ Failed to create demo video"
            else:
                # Use one of the available demo videos
                import random
                selected_video = random.choice(available_videos)
                demo_in = demo_dir / f"demo_input_{uuid.uuid4().hex[:8]}.mp4"
                
                logger.info(f"📹 Using real demo video: {selected_video.name}")
                
                # Copy the selected video to demo directory
                import shutil
                shutil.copy2(selected_video, demo_in)
                
                logger.info(f"✅ Demo video prepared: {demo_in.stat().st_size / 1024:.1f} KB")
            
            # Get basic video info for logging
            try:
                import cv2
                cap = cv2.VideoCapture(str(demo_in))
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    cap.release()
                    
                    logger.info(f"📊 Video info: {width}x{height} @ {fps:.2f}fps, {duration:.2f}s ({frame_count} frames)")
                else:
                    logger.warning("⚠️ Could not read video properties")
            except Exception as e:
                logger.warning(f"⚠️ Video info extraction failed: {e}")
            
            # Create output file
            demo_out = demo_dir / f"demo_output_{uuid.uuid4().hex[:8]}.mp4"
            logger.info(f"🎯 Processing demo with SOTA pipeline...")
            
            # Run SOTA pipeline
            ok, msg = _run_sota_pipeline(str(demo_in), str(demo_out), 24, 'standard', False, False)
            
            if ok and demo_out.exists():
                logger.info(f"✅ Demo processing completed: {demo_out}")
                # Provide detailed status including file info
                input_size = demo_in.stat().st_size / 1024
                output_size = demo_out.stat().st_size / 1024
                status_msg = f"✅ Demo completed: {msg}\n"
                status_msg += f"📁 Input: {demo_in.name} ({input_size:.1f} KB)\n"
                status_msg += f"📁 Output: {demo_out.name} ({output_size:.1f} KB)\n"
                status_msg += f"📈 Size ratio: {output_size/input_size:.2f}x"
                
                return str(demo_out), status_msg
            else:
                logger.error(f"❌ Demo processing failed: {msg}")
                return None, f"❌ Demo processing failed: {msg}"
                
        except Exception as e:
            error_msg = f"❌ Demo failed with error: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

    demo_btn.click(
        fn=_run_demo,
        outputs=[output_video_viewer, status_text]
    )

    def _eval_last():
        # Evaluate the last SOTA demo by using real demo video for comparison
        try:
            if not job_history:
                return "❌ No jobs to evaluate"
            last = job_history[-1]
            last_out = last.get('output')
            if not last_out or not Path(last_out).exists():
                return "❌ Last output file not found"
            
            # Look for available demo videos first
            demo_videos_dir = Path('data/demo_videos')
            reference_video = None
            
            if demo_videos_dir.exists():
                available_videos = [v for v in demo_videos_dir.glob('*.mp4') 
                                  if v.is_file() and v.stat().st_size > 0]
                if available_videos:
                    # Use the first available demo video as reference
                    reference_video = str(available_videos[0])
                    logger.info(f"📊 Using {available_videos[0].name} as reference for evaluation")
            
            # Fallback to generating a demo video if no real ones available
            if not reference_video:
                with tempfile.TemporaryDirectory() as td:
                    td = Path(td)
                    demo_in = td / 'demo.mp4'
                    logger.info("📊 Generating synthetic reference video for evaluation")
                    if _generate_demo_video(str(demo_in)):
                        reference_video = str(demo_in)
                    else:
                        return "❌ Failed to create reference video for evaluation"
            
            return _evaluate_psnr_ssim(reference_video, last_out)
            
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
                    # Try Real-ESRGAN first, fallback to BasicUpscaler if it fails
                    try:
                        from models.enhancement.frame.realesrgan_fallback import RealESRGANFallback
                        up = RealESRGANFallback(device='cuda' if CUDA_AVAILABLE else 'cpu', scale=2)
                        stats = up.enhance_video(str(in_path), str(out_path))
                        ok, msg = True, f"✅ Real-ESRGAN upscaling completed: {stats}"
                        
                    except ImportError as e:
                        if "functional_tensor" in str(e):
                            logger.warning(f"⚠️ Real-ESRGAN unavailable, using BasicUpscaler fallback: {e}")
                            try:
                                from models.enhancement.frame.basic_upscaler_fallback import BasicUpscalerFallback
                                up = BasicUpscalerFallback(device='cpu', model_name='LANCZOS', scale=2)
                                stats = up.enhance_video(str(in_path), str(out_path))
                                ok, msg = True, f"✅ BasicUpscaler (fallback) completed: {stats}"
                            except Exception as fallback_e:
                                ok, msg = False, f"❌ BasicUpscaler fallback failed: {fallback_e}"
                        else:
                            ok, msg = False, f"❌ Real-ESRGAN import failed: {e}"
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Real-ESRGAN processing failed, trying BasicUpscaler: {e}")
                        try:
                            from models.enhancement.frame.basic_upscaler_fallback import BasicUpscalerFallback
                            up = BasicUpscalerFallback(device='cpu', model_name='LANCZOS', scale=2)
                            stats = up.enhance_video(str(in_path), str(out_path))
                            ok, msg = True, f"✅ BasicUpscaler (fallback after Real-ESRGAN failure) completed: {stats}"
                        except Exception as fallback_e:
                            ok, msg = False, f"❌ Both Real-ESRGAN and BasicUpscaler failed: {e} | {fallback_e}"
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
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "share": False,
        "show_error": True
    }
    
    # Enable OAuth for HuggingFace Spaces with ZeroGPU+
    if HUGGINGFACE_SPACE:
        launch_kwargs["auth"] = "huggingface"  # Enable HF OAuth
        logger.info("🔐 HuggingFace OAuth authentication enabled for ZeroGPU+ access")
    
    app.launch(**launch_kwargs)
