#!/usr/bin/env python3
"""
Test script to verify model implementations work with real weights
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_realesrgan():
    """Test Real-ESRGAN handler with weight downloading"""
    logger.info("🎯 Testing Real-ESRGAN handler...")
    
    try:
        from models.enhancement.vsr.realesrgan_handler import RealESRGANHandler
        
        # Initialize handler
        handler = RealESRGANHandler(device="cpu", scale=2)  # Smaller scale for testing
        
        # Test a small demo video
        input_path = "data/demo_videos/test.mp4"
        if not os.path.exists(input_path):
            logger.warning(f"Demo video not found: {input_path}")
            return False
            
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name
        
        try:
            # Process video
            logger.info("Processing video with Real-ESRGAN...")
            stats = handler.restore_video(input_path, output_path)
            
            logger.info(f"✅ Real-ESRGAN processing completed!")
            logger.info(f"   Input frames: {stats['input_frames']}")
            logger.info(f"   Output frames: {stats['output_frames']}")
            logger.info(f"   Scale: {stats['scale_factor']}x")
            
            # Check output file exists
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"✅ Output file created: {os.path.getsize(output_path)} bytes")
                return True
            else:
                logger.error("❌ Output file not created or empty")
                return False
                
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
                
    except Exception as e:
        logger.error(f"❌ Real-ESRGAN test failed: {e}")
        return False

def test_rife():
    """Test RIFE handler with weight downloading"""
    logger.info("🎬 Testing RIFE handler...")
    
    try:
        from models.interpolation.rife_handler import RIFEHandler
        
        # Initialize handler
        handler = RIFEHandler(device="cpu", fp16=False)  # CPU mode
        
        # Test a small demo video
        input_path = "data/demo_videos/test.mp4"
        if not os.path.exists(input_path):
            logger.warning(f"Demo video not found: {input_path}")
            return False
            
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name
        
        try:
            # Process video (only interpolate first few frames to save time)
            logger.info("Processing video with RIFE...")
            
            # Create a shorter test clip first
            import cv2
            cap = cv2.VideoCapture(input_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            short_path = "/tmp/test_short.mp4"
            out = cv2.VideoWriter(short_path, fourcc, fps, (width, height))
            
            # Write only first 10 frames for quick test
            frame_count = 0
            while frame_count < 10:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frame_count += 1
            
            cap.release()
            out.release()
            
            if frame_count > 1:
                stats = handler.interpolate_video(short_path, output_path, interpolation_factor=2)
                
                logger.info(f"✅ RIFE processing completed!")
                logger.info(f"   Input frames: {stats['input_frames']}")
                logger.info(f"   Output frames: {stats['output_frames']}")
                logger.info(f"   Interpolated frames: {stats['interpolated_frames']}")
                
                # Check output file exists
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    logger.info(f"✅ Output file created: {os.path.getsize(output_path)} bytes")
                    return True
                else:
                    logger.error("❌ Output file not created or empty")
                    return False
            else:
                logger.warning("⚠️ Not enough frames for interpolation test")
                return False
                
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
            if os.path.exists("/tmp/test_short.mp4"):
                os.unlink("/tmp/test_short.mp4")
                
    except Exception as e:
        logger.error(f"❌ RIFE test failed: {e}")
        return False

def test_seedvr2_improved():
    """Test improved SeedVR2 handler"""
    logger.info("🌱 Testing improved SeedVR2 handler...")
    
    try:
        from models.enhancement.zeroshot.seedvr2_handler import SeedVR2Handler
        
        # Initialize handler
        handler = SeedVR2Handler(device="cpu", num_frames=4)  # Smaller frames for testing
        
        # Test model info (this should not crash now)
        model_info = handler.get_model_info()
        logger.info(f"✅ SeedVR2 model info retrieved: {model_info['name']}")
        logger.info(f"   Parameters: {model_info['parameters']:,}")
        
        return True
                
    except Exception as e:
        logger.error(f"❌ SeedVR2 test failed: {e}")
        return False

def main():
    """Run model integration tests"""
    logger.info("🚀 Starting model integration tests...")
    
    # Track results
    results = {}
    
    # Test Real-ESRGAN
    results['realesrgan'] = test_realesrgan()
    
    # Test RIFE 
    results['rife'] = test_rife()
    
    # Test improved SeedVR2
    results['seedvr2'] = test_seedvr2_improved()
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    logger.info(f"\n📊 Model Integration Test Results:")
    logger.info(f"=" * 50)
    for model, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{model.upper()}: {status}")
    
    logger.info(f"=" * 50)
    logger.info(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All model integration tests PASSED!")
        return 0
    else:
        logger.error("❌ Some model integration tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())