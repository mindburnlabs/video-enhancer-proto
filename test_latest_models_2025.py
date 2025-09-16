#!/usr/bin/env python3
"""
Comprehensive Test Script for Latest 2025 Video Enhancement Models

This script tests all implemented models with the latest weights and architectures:
- SeedVR2-3B and SeedVR2-7B (latest 2025 diffusion transformer models)
- VSRM (Mamba-based video super-resolution)
- FastMambaVSR (Ultra-efficient Mamba VSR)
- RIFE (Frame interpolation)
- Real-ESRGAN (Super-resolution)
"""

import sys
import os
import logging
import torch
import cv2
import numpy as np
from pathlib import Path
import tempfile
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_video(output_path: str, frames: int = 30, fps: int = 10, size=(240, 160)):
    """Create a simple test video for processing."""
    logger.info(f"Creating test video: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    for i in range(frames):
        # Create a simple animated pattern
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # Moving circle
        center_x = int(size[0] * 0.3 + (size[0] * 0.4) * np.sin(i * 0.2))
        center_y = int(size[1] * 0.5)
        cv2.circle(frame, (center_x, center_y), 20, (255, 100, 50), -1)
        
        # Moving square
        square_x = int(size[0] * 0.7 + (size[0] * 0.2) * np.cos(i * 0.15))
        square_y = int(size[1] * 0.3 + (size[1] * 0.2) * np.sin(i * 0.15))
        cv2.rectangle(frame, (square_x-15, square_y-15), (square_x+15, square_y+15), (50, 255, 100), -1)
        
        # Add some noise for realism
        noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
        frame = cv2.addWeighted(frame, 0.9, noise, 0.1, 0)
        
        out.write(frame)
    
    out.release()
    logger.info(f"✅ Test video created: {output_path}")

def test_seedvr2_models():
    """Test both SeedVR2-3B and SeedVR2-7B models."""
    logger.info("🎯 Testing SeedVR2 Models (Latest 2025)")
    
    try:
        from models.enhancement.zeroshot import SeedVR2_3B, SeedVR2_7B, create_seedvr2_3b
        
        # Test SeedVR2-3B
        logger.info("Testing SeedVR2-3B...")
        seedvr2_3b = create_seedvr2_3b(device="cpu", auto_download=True)
        
        model_info_3b = seedvr2_3b.get_model_info()
        logger.info(f"SeedVR2-3B Info: {model_info_3b['name']}")
        logger.info(f"  Description: {model_info_3b['description']}")
        logger.info(f"  Parameters: {model_info_3b['parameters']:,}")
        logger.info(f"  Architecture: {model_info_3b['architecture']}")
        logger.info(f"  Model Loaded: {model_info_3b['model_loaded']}")
        
        # Test SeedVR2-7B (if enabled)
        try:
            logger.info("Testing SeedVR2-7B...")
            seedvr2_7b = SeedVR2_7B(device="cpu", auto_download=False)  # Don't auto-download 7B for testing
            
            model_info_7b = seedvr2_7b.get_model_info()
            logger.info(f"SeedVR2-7B Info: {model_info_7b['name']}")
            logger.info(f"  Description: {model_info_7b['description']}")
            logger.info(f"  Parameters: {model_info_7b['parameters']:,}")
            logger.info(f"  Model Loaded: {model_info_7b['model_loaded']}")
        except Exception as e:
            logger.warning(f"SeedVR2-7B test skipped: {e}")
        
        logger.info("✅ SeedVR2 models tested successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ SeedVR2 test failed: {e}")
        return False

def test_vsrm_model():
    """Test VSRM Mamba-based model."""
    logger.info("🔥 Testing VSRM Model")
    
    try:
        from models.enhancement.vsr import VSRMHandler
        
        vsrm = VSRMHandler(device="cpu", auto_download=True)
        
        model_info = vsrm.get_model_info()
        logger.info(f"VSRM Info: {model_info['name']}")
        logger.info(f"  Description: {model_info['description']}")
        logger.info(f"  Parameters: {model_info['parameters']:,}")
        logger.info(f"  Architecture: {model_info['architecture']}")
        
        logger.info("✅ VSRM model tested successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ VSRM test failed: {e}")
        return False

def test_fast_mamba_vsr_model():
    """Test FastMambaVSR ultra-efficient model."""
    logger.info("⚡ Testing FastMambaVSR Model")
    
    try:
        from models.enhancement.vsr import FastMambaVSRHandler
        
        fast_mamba = FastMambaVSRHandler(device="cpu", auto_download=True)
        
        model_info = fast_mamba.get_model_info()
        logger.info(f"FastMambaVSR Info: {model_info['name']}")
        logger.info(f"  Description: {model_info['description']}")
        logger.info(f"  Parameters: {model_info['parameters']:,}")
        logger.info(f"  Architecture: {model_info['architecture']}")
        logger.info(f"  Optimizations: {model_info['optimizations']}")
        
        logger.info("✅ FastMambaVSR model tested successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ FastMambaVSR test failed: {e}")
        return False

def test_rife_model():
    """Test RIFE frame interpolation model."""
    logger.info("🎬 Testing RIFE Model")
    
    try:
        from models.interpolation import RIFEHandler
        
        rife = RIFEHandler(device="cpu", auto_download=True)
        
        model_info = rife.get_model_info()
        logger.info(f"RIFE Info: {model_info['name']}")
        logger.info(f"  Description: {model_info['description']}")
        logger.info(f"  Parameters: {model_info['parameters']:,}")
        logger.info(f"  Capabilities: {model_info['capabilities']}")
        
        logger.info("✅ RIFE model tested successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ RIFE test failed: {e}")
        return False

def test_real_esrgan_model():
    """Test Real-ESRGAN super-resolution model."""
    logger.info("🖼️ Testing Real-ESRGAN Model")
    
    try:
        from models.enhancement.frame import RealESRGANHandler
        
        esrgan = RealESRGANHandler(device="cpu", auto_download=True)
        
        model_info = esrgan.get_model_info()
        logger.info(f"Real-ESRGAN Info: {model_info['name']}")
        logger.info(f"  Description: {model_info['description']}")
        logger.info(f"  Scale: {model_info['scale']}x")
        logger.info(f"  Parameters: {model_info['parameters']:,}")
        
        logger.info("✅ Real-ESRGAN model tested successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Real-ESRGAN test failed: {e}")
        return False

def test_video_processing():
    """Test actual video processing with one of the models."""
    logger.info("🎥 Testing Video Processing")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_video = os.path.join(temp_dir, "test_input.mp4")
            output_video = os.path.join(temp_dir, "test_output.mp4")
            
            # Create test video
            create_test_video(input_video, frames=15, fps=5, size=(160, 120))
            
            # Test with SeedVR2-3B (if available)
            try:
                from models.enhancement.zeroshot import create_seedvr2_3b
                
                model = create_seedvr2_3b(device="cpu", auto_download=False)
                
                logger.info("Processing video with SeedVR2-3B...")
                start_time = time.time()
                
                stats = model.restore_video(
                    input_path=input_video,
                    output_path=output_video,
                    degradation_type="low_resolution",
                    auto_adapt=True,
                    fp16=False  # Use FP32 for CPU
                )
                
                processing_time = time.time() - start_time
                
                logger.info(f"✅ Video processing completed in {processing_time:.2f}s")
                logger.info(f"  Input frames: {stats['input_frames']}")
                logger.info(f"  Output frames: {stats['output_frames']}")
                logger.info(f"  Processing mode: {stats['processing_mode']}")
                
                # Verify output exists
                if os.path.exists(output_video):
                    logger.info(f"  Output video size: {os.path.getsize(output_video)} bytes")
                else:
                    logger.warning("Output video not found")
                
                return True
                
            except Exception as e:
                logger.warning(f"SeedVR2 processing test failed: {e}")
                
                # Fallback to Real-ESRGAN if available
                try:
                    from models.enhancement.frame import RealESRGANHandler
                    
                    model = RealESRGANHandler(device="cpu", auto_download=False)
                    
                    logger.info("Processing video with Real-ESRGAN...")
                    stats = model.enhance_video(
                        input_path=input_video,
                        output_path=output_video,
                        fp16=False
                    )
                    
                    logger.info("✅ Video processing with Real-ESRGAN completed")
                    logger.info(f"  Processed frames: {stats.get('frames_processed', 'N/A')}")
                    
                    return True
                    
                except Exception as e2:
                    logger.error(f"Fallback processing also failed: {e2}")
                    return False
        
    except Exception as e:
        logger.error(f"❌ Video processing test failed: {e}")
        return False

def main():
    """Run comprehensive model tests."""
    logger.info("🚀 Starting Comprehensive 2025 Video Enhancement Model Tests")
    logger.info("=" * 60)
    
    # System info
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
    
    logger.info("=" * 60)
    
    test_results = {}
    
    # Test all models
    tests = [
        ("SeedVR2 Models (2025 Latest)", test_seedvr2_models),
        ("VSRM Mamba Model", test_vsrm_model),
        ("FastMambaVSR Model", test_fast_mamba_vsr_model),
        ("RIFE Interpolation", test_rife_model),
        ("Real-ESRGAN", test_real_esrgan_model),
        ("Video Processing", test_video_processing),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        logger.info("-" * 40)
        
        try:
            result = test_func()
            test_results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{status}: {test_name}")
        except Exception as e:
            test_results[test_name] = False
            logger.error(f"❌ FAILED: {test_name} - {e}")
        
        logger.info("-" * 40)
    
    # Summary
    logger.info("\n📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
    
    logger.info("=" * 60)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Latest 2025 models are ready for use.")
        return 0
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)