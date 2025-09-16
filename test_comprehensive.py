#!/usr/bin/env python3
"""
Comprehensive Testing Framework for Video Enhancement Pipeline
Tests all components, models, and edge cases with real video data.
"""

import os
import sys
import tempfile
import logging
import traceback
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple, Optional
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_comprehensive.log')
    ]
)
logger = logging.getLogger(__name__)

# Test Results Storage
test_results = {
    "passed": [],
    "failed": [],
    "warnings": [],
    "performance": {},
    "environment": {}
}

class TestFramework:
    """Comprehensive testing framework for video enhancement."""
    
    def __init__(self):
        self.temp_dir = None
        self.demo_videos = []
        self.setup_test_environment()
        
    def setup_test_environment(self):
        """Set up test environment and discover test videos."""
        try:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="video_enhancer_test_"))
            logger.info(f"🔧 Test directory: {self.temp_dir}")
            
            # Discover demo videos
            demo_paths = [
                Path("data/demo_videos"),
                Path("/Users/ivan/Downloads"),  # Fallback location
            ]
            
            for demo_path in demo_paths:
                if demo_path.exists():
                    videos = list(demo_path.glob("*.mp4"))
                    if videos:
                        self.demo_videos.extend(videos)
                        logger.info(f"📹 Found {len(videos)} demo videos in {demo_path}")
                        
            if not self.demo_videos:
                logger.warning("⚠️ No demo videos found, will create synthetic test videos")
                self._create_synthetic_test_videos()
                
            # Collect environment info
            self._collect_environment_info()
            
        except Exception as e:
            logger.error(f"❌ Test environment setup failed: {e}")
            raise
            
    def _create_synthetic_test_videos(self):
        """Create synthetic test videos for testing."""
        try:
            import cv2
            import numpy as np
            
            logger.info("🎬 Creating synthetic test videos...")
            
            # Create different test scenarios
            test_scenarios = [
                {"name": "simple", "size": (320, 240), "duration": 2, "fps": 24},
                {"name": "hd", "size": (1280, 720), "duration": 1, "fps": 30},
                {"name": "noisy", "size": (480, 360), "duration": 2, "fps": 24, "noise": True},
            ]
            
            for scenario in test_scenarios:
                video_path = self.temp_dir / f"synthetic_{scenario['name']}.mp4"
                self._create_test_video(
                    str(video_path),
                    scenario["size"], 
                    scenario["duration"], 
                    scenario["fps"],
                    noise=scenario.get("noise", False)
                )
                self.demo_videos.append(video_path)
                
        except Exception as e:
            logger.error(f"❌ Synthetic video creation failed: {e}")
            
    def _create_test_video(self, path: str, size: tuple, duration: int, fps: int, noise: bool = False):
        """Create a single test video."""
        try:
            import cv2
            import numpy as np
            import math
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(path, fourcc, fps, size)
            
            total_frames = duration * fps
            for i in range(total_frames):
                t = i / fps
                
                # Create colorful moving pattern
                frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                
                # Moving circle
                cx = int((math.sin(t * 2) * 0.3 + 0.5) * size[0])
                cy = int((math.cos(t * 1.5) * 0.3 + 0.5) * size[1])
                cv2.circle(frame, (cx, cy), min(size[0], size[1]) // 8, (0, 255, 255), -1)
                
                # Moving rectangle
                rx = int((math.cos(t * 3) * 0.25 + 0.75) * size[0] - 40)
                ry = int((math.sin(t * 2.5) * 0.25 + 0.25) * size[1] - 30)
                cv2.rectangle(frame, (rx, ry), (rx + 80, ry + 60), (255, 0, 255), -1)
                
                # Add noise if requested
                if noise and i % 3 == 0:
                    noise_arr = np.random.normal(0, 25, frame.shape).astype(np.int16)
                    frame = np.clip(frame.astype(np.int16) + noise_arr, 0, 255).astype(np.uint8)
                
                out.write(frame)
                
            out.release()
            logger.info(f"✅ Created test video: {Path(path).name}")
            
        except Exception as e:
            logger.error(f"❌ Test video creation failed: {e}")
            
    def _collect_environment_info(self):
        """Collect environment information for diagnostics."""
        try:
            import torch
            import cv2
            import psutil
            
            test_results["environment"] = {
                "python_version": sys.version,
                "pytorch_version": torch.__version__,
                "opencv_version": cv2.__version__,
                "cpu_count": psutil.cpu_count(),
                "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "cuda_available": torch.cuda.is_available() if not os.environ.get('SPACE_ID') else "deferred",
                "huggingface_space": os.environ.get('SPACE_ID') is not None,
                "zerogpu_available": os.environ.get('ZERO_GPU') == '1'
            }
            
            logger.info(f"🔍 Environment: Python {sys.version.split()[0]}, PyTorch {torch.__version__}")
            
        except Exception as e:
            logger.warning(f"Environment info collection failed: {e}")
            
    def test_imports_and_basic_functionality(self):
        """Test all critical imports and basic functionality."""
        logger.info("🧪 Testing imports and basic functionality...")
        
        try:
            # Test core imports
            import torch
            import cv2
            import numpy as np
            from PIL import Image
            self._mark_test_passed("core_imports")
            
            # Test PyTorch basic operations
            tensor = torch.randn(2, 3, 4, 4)
            result = torch.nn.functional.relu(tensor)
            assert result.shape == tensor.shape
            self._mark_test_passed("pytorch_operations")
            
            # Test OpenCV operations
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            assert gray.shape == (100, 100)
            self._mark_test_passed("opencv_operations")
            
        except Exception as e:
            self._mark_test_failed("basic_functionality", str(e))
            
    def test_app_initialization(self):
        """Test app initialization without errors."""
        logger.info("🧪 Testing app initialization...")
        
        try:
            # Import main app components
            from app import initialize_enhancer, enhancer, SOTA_AVAILABLE
            
            # Test enhancer initialization
            status = initialize_enhancer()
            if "✅" in status or "⚠️" in status:
                self._mark_test_passed("app_initialization")
                logger.info(f"App status: {status}")
            else:
                self._mark_test_failed("app_initialization", status)
                
            # Check SOTA availability
            if SOTA_AVAILABLE:
                self._mark_test_passed("sota_imports")
            else:
                self._mark_test_warning("sota_imports", "SOTA models not available")
                
        except Exception as e:
            self._mark_test_failed("app_initialization", str(e))
            
    def test_video_processing_pipeline(self):
        """Test the complete video processing pipeline."""
        logger.info("🧪 Testing video processing pipeline...")
        
        if not self.demo_videos:
            self._mark_test_failed("video_processing", "No test videos available")
            return
            
        try:
            from app import enhancer
            
            test_video = self.demo_videos[0]
            output_path = self.temp_dir / f"enhanced_{test_video.name}"
            
            logger.info(f"📹 Processing test video: {test_video.name}")
            start_time = time.time()
            
            # Test video processing
            success, message = enhancer.process_video(
                str(test_video), 
                str(output_path), 
                target_fps=24
            )
            
            processing_time = time.time() - start_time
            test_results["performance"]["video_processing_time"] = processing_time
            
            if success:
                if output_path.exists() and output_path.stat().st_size > 0:
                    self._mark_test_passed("video_processing_pipeline")
                    logger.info(f"✅ Video processed in {processing_time:.2f}s: {message}")
                else:
                    self._mark_test_failed("video_processing_pipeline", "Output file missing or empty")
            else:
                self._mark_test_failed("video_processing_pipeline", message)
                
        except Exception as e:
            self._mark_test_failed("video_processing_pipeline", str(e))
            
    def test_sota_pipeline(self):
        """Test SOTA pipeline if available."""
        logger.info("🧪 Testing SOTA pipeline...")
        
        try:
            from app import _run_sota_pipeline, SOTA_AVAILABLE
            
            if not SOTA_AVAILABLE:
                self._mark_test_warning("sota_pipeline", "SOTA not available for testing")
                return
                
            if not self.demo_videos:
                self._mark_test_failed("sota_pipeline", "No test videos available")
                return
                
            test_video = self.demo_videos[0]
            output_path = self.temp_dir / f"sota_enhanced_{test_video.name}"
            
            logger.info(f"🚀 Testing SOTA pipeline with: {test_video.name}")
            start_time = time.time()
            
            success, message = _run_sota_pipeline(
                str(test_video),
                str(output_path),
                target_fps=24,
                latency_class='standard',
                enable_face_expert=False,
                enable_hfr=False
            )
            
            processing_time = time.time() - start_time
            test_results["performance"]["sota_processing_time"] = processing_time
            
            if success:
                if output_path.exists() and output_path.stat().st_size > 0:
                    self._mark_test_passed("sota_pipeline")
                    logger.info(f"✅ SOTA pipeline completed in {processing_time:.2f}s")
                else:
                    self._mark_test_failed("sota_pipeline", "SOTA output missing")
            else:
                self._mark_test_failed("sota_pipeline", message)
                
        except Exception as e:
            self._mark_test_failed("sota_pipeline", str(e))
            
    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        logger.info("🧪 Testing error handling...")
        
        try:
            from app import process_video_gradio
            
            # Test with None input
            result, message = process_video_gradio(None, 30, "ZeroGPU Upscaler (fast)", "standard", False, False)
            if result is None and "❌" in message:
                self._mark_test_passed("error_handling_none_input")
            else:
                self._mark_test_failed("error_handling_none_input", f"Unexpected response: {message}")
                
            # Test with non-existent file
            fake_path = "/nonexistent/fake_video.mp4"
            result, message = process_video_gradio(fake_path, 30, "ZeroGPU Upscaler (fast)", "standard", False, False)
            if result is None and ("❌" in message or "not found" in message.lower()):
                self._mark_test_passed("error_handling_missing_file")
            else:
                self._mark_test_failed("error_handling_missing_file", f"Unexpected response: {message}")
                
        except Exception as e:
            self._mark_test_failed("error_handling", str(e))
            
    def test_demo_functionality(self):
        """Test demo functionality with real demo videos."""
        logger.info("🧪 Testing demo functionality...")
        
        try:
            # Import demo functions
            sys.path.insert(0, str(Path(__file__).parent))
            from app import _run_demo
            
            logger.info("🎬 Running demo test...")
            start_time = time.time()
            
            # Test demo execution
            result_video, status_message = _run_demo()
            
            demo_time = time.time() - start_time
            test_results["performance"]["demo_time"] = demo_time
            
            if result_video and "✅" in status_message:
                self._mark_test_passed("demo_functionality")
                logger.info(f"✅ Demo completed in {demo_time:.2f}s")
            else:
                self._mark_test_failed("demo_functionality", status_message or "Demo returned no result")
                
        except Exception as e:
            self._mark_test_failed("demo_functionality", str(e))
            
    def test_gradio_interface(self):
        """Test Gradio interface components."""
        logger.info("🧪 Testing Gradio interface...")
        
        try:
            import gradio as gr
            from app import app
            
            # Check if app is properly created
            if hasattr(app, 'interface') or hasattr(app, 'app'):
                self._mark_test_passed("gradio_interface_creation")
            else:
                self._mark_test_warning("gradio_interface_creation", "Gradio interface structure unknown")
                
            # Test basic Gradio functionality
            test_interface = gr.Interface(
                fn=lambda x: x,
                inputs="text",
                outputs="text"
            )
            self._mark_test_passed("gradio_basic_functionality")
            
        except Exception as e:
            self._mark_test_failed("gradio_interface", str(e))
            
    def test_memory_usage(self):
        """Test memory usage during processing."""
        logger.info("🧪 Testing memory usage...")
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run a processing test
            if self.demo_videos:
                from app import enhancer
                test_video = self.demo_videos[0]
                output_path = self.temp_dir / f"memory_test_{test_video.name}"
                
                enhancer.process_video(str(test_video), str(output_path), target_fps=24)
                
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = final_memory - initial_memory
                
                test_results["performance"]["memory_usage_mb"] = {
                    "initial": round(initial_memory, 2),
                    "final": round(final_memory, 2),
                    "increase": round(memory_increase, 2)
                }
                
                if memory_increase < 1000:  # Less than 1GB increase
                    self._mark_test_passed("memory_usage")
                else:
                    self._mark_test_warning("memory_usage", f"High memory usage: {memory_increase:.1f}MB")
            else:
                self._mark_test_warning("memory_usage", "No test videos for memory test")
                
        except Exception as e:
            self._mark_test_failed("memory_usage", str(e))
            
    def run_all_tests(self):
        """Run all tests in sequence."""
        logger.info("🚀 Starting comprehensive test suite...")
        
        test_methods = [
            self.test_imports_and_basic_functionality,
            self.test_app_initialization,
            self.test_gradio_interface,
            self.test_video_processing_pipeline,
            self.test_sota_pipeline,
            self.test_error_handling,
            self.test_demo_functionality,
            self.test_memory_usage,
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                logger.error(f"❌ Test {test_method.__name__} crashed: {e}")
                self._mark_test_failed(test_method.__name__, f"Test crashed: {e}")
                
        self.generate_report()
        
    def _mark_test_passed(self, test_name: str):
        """Mark a test as passed."""
        test_results["passed"].append(test_name)
        logger.info(f"✅ {test_name}: PASSED")
        
    def _mark_test_failed(self, test_name: str, reason: str):
        """Mark a test as failed."""
        test_results["failed"].append({"test": test_name, "reason": reason})
        logger.error(f"❌ {test_name}: FAILED - {reason}")
        
    def _mark_test_warning(self, test_name: str, reason: str):
        """Mark a test with a warning."""
        test_results["warnings"].append({"test": test_name, "reason": reason})
        logger.warning(f"⚠️ {test_name}: WARNING - {reason}")
        
    def generate_report(self):
        """Generate comprehensive test report."""
        total_tests = len(test_results["passed"]) + len(test_results["failed"])
        passed_count = len(test_results["passed"])
        failed_count = len(test_results["failed"])
        warning_count = len(test_results["warnings"])
        
        success_rate = (passed_count / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "="*80)
        print("🧪 COMPREHENSIVE TEST REPORT")
        print("="*80)
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"⚠️ Warnings: {warning_count}")
        print(f"Success Rate: {success_rate:.1f}%")
        print()
        
        if test_results["failed"]:
            print("❌ FAILED TESTS:")
            for failure in test_results["failed"]:
                print(f"   • {failure['test']}: {failure['reason']}")
            print()
            
        if test_results["warnings"]:
            print("⚠️ WARNINGS:")
            for warning in test_results["warnings"]:
                print(f"   • {warning['test']}: {warning['reason']}")
            print()
            
        if test_results["performance"]:
            print("📊 PERFORMANCE METRICS:")
            for metric, value in test_results["performance"].items():
                if isinstance(value, dict):
                    print(f"   • {metric}:")
                    for k, v in value.items():
                        print(f"     - {k}: {v}")
                else:
                    print(f"   • {metric}: {value}")
            print()
            
        print("🔍 ENVIRONMENT:")
        for key, value in test_results["environment"].items():
            print(f"   • {key}: {value}")
        
        print("="*80)
        
        # Save detailed report
        report_path = Path("test_comprehensive_report.json")
        with open(report_path, "w") as f:
            json.dump(test_results, f, indent=2, default=str)
        print(f"📄 Detailed report saved to: {report_path}")
        
        # Return success status
        return success_rate >= 70  # Consider 70%+ success rate as acceptable
        
    def cleanup(self):
        """Clean up test resources."""
        try:
            if self.temp_dir and self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info(f"🧹 Cleaned up test directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

def main():
    """Main test execution."""
    framework = TestFramework()
    
    try:
        success = framework.run_all_tests()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"❌ Test framework failed: {e}")
        traceback.print_exc()
        return 1
    finally:
        framework.cleanup()

if __name__ == "__main__":
    sys.exit(main())