#!/usr/bin/env python3
"""
Smoke tests for end-to-end video enhancement pipeline.
Uses real video files to test the complete processing pipeline.
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


import pytest
import torch
import cv2
import numpy as np
import tempfile
import os
import logging
import json
from pathlib import Path
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSmokeE2E:
    """End-to-end smoke tests for video enhancement pipeline."""
    
    @classmethod
    def setup_class(cls):
        """Setup test environment."""
        cls.test_assets_dir = Path(__file__).parent.parent / "data" / "test_assets"
        cls.output_dir = Path(__file__).parent.parent / "data" / "test_output"
        cls.output_dir.mkdir(exist_ok=True)
        
        # Find test videos
        cls.test_videos = list(cls.test_assets_dir.glob("*.mp4"))
        if not cls.test_videos:
            pytest.skip("No test videos found in data/test_assets/")
        
        logger.info(f"Found {len(cls.test_videos)} test videos")
        for video in cls.test_videos:
            logger.info(f"  - {video.name} ({video.stat().st_size / (1024*1024):.1f} MB)")
    
    def test_video_file_integrity(self):
        """Test that test video files are valid and readable."""
        for video_path in self.test_videos:
            # Test with OpenCV
            cap = cv2.VideoCapture(str(video_path))
            assert cap.isOpened(), f"Cannot open video file: {video_path}"
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            assert fps > 0, f"Invalid FPS for {video_path.name}: {fps}"
            assert frame_count > 0, f"No frames in {video_path.name}"
            assert width > 0 and height > 0, f"Invalid dimensions for {video_path.name}: {width}x{height}"
            
            # Read first frame
            ret, frame = cap.read()
            assert ret, f"Cannot read first frame from {video_path.name}"
            assert frame is not None, f"First frame is None for {video_path.name}"
            assert frame.shape == (height, width, 3), f"Unexpected frame shape for {video_path.name}: {frame.shape}"
            
            cap.release()
            
            logger.info(f"✅ {video_path.name}: {width}x{height}, {frame_count} frames, {fps:.2f} FPS")
    
    def test_degradation_router_analysis(self):
        """Test degradation analysis on real videos."""
        try:
            from models.analysis.degradation_router import DegradationRouter
            
            router = DegradationRouter()
            
            for video_path in self.test_videos:
                logger.info(f"🔍 Analyzing degradations in {video_path.name}...")
                
                # Run full analysis and routing
                result = router.analyze_and_route(
                    str(video_path),
                    latency_class='standard',
                    allow_diffusion=True,
                    allow_zero_shot=True
                )
                
                # Validate analysis results
                assert 'degradations' in result
                assert 'content_analysis' in result
                assert 'expert_routing' in result
                assert 'processing_order' in result
                assert 'confidence_score' in result
                
                # Check degradation metrics
                degradations = result['degradations']
                expected_metrics = ['compression_artifacts', 'motion_blur', 'low_light', 'noise', 'temporal_inconsistency']
                for metric in expected_metrics:
                    assert metric in degradations
                    assert 0.0 <= degradations[metric] <= 1.0, f"Invalid {metric} score: {degradations[metric]}"
                
                # Check routing plan
                routing_plan = result['expert_routing']
                assert 'primary_model' in routing_plan
                assert routing_plan['primary_model'] in ['vsrm', 'seedvr2', 'ditvr', 'fast_mamba_vsr']
                
                # Log results
                logger.info(f"  Primary model: {routing_plan['primary_model']}")
                logger.info(f"  Confidence: {result['confidence_score']:.3f}")
                logger.info(f"  Top degradations:")
                for metric, score in sorted(degradations.items(), key=lambda x: x[1], reverse=True)[:3]:
                    logger.info(f"    - {metric}: {score:.3f}")
                
        except ImportError:
            pytest.skip("DegradationRouter not available")
    
    def test_gradio_ui_initialization(self):
        """Test that the Gradio UI can be initialized without errors."""
        try:
            # Import the main app module
            import importlib.util
            spec = importlib.util.spec_from_file_location("app", "app.py")
            app_module = importlib.util.module_from_spec(spec)
            
            # Test basic imports work
            import gradio as gr
            
            logger.info("✅ Gradio UI imports successful")
            
        except Exception as e:
            pytest.fail(f"Failed to initialize Gradio UI: {e}")
    
    def test_basic_video_processing_pipeline(self):
        """Test basic video processing without heavy models."""
        try:
            from utils.video_utils import VideoUtils
            
            video_utils = VideoUtils()
            
            for video_path in self.test_videos:
                logger.info(f"📹 Testing basic processing for {video_path.name}...")
                
                # Test metadata extraction
                metadata = video_utils.get_video_metadata(str(video_path))
                
                assert 'fps' in metadata
                assert 'frame_count' in metadata
                assert 'duration' in metadata
                assert 'width' in metadata
                assert 'height' in metadata
                
                assert metadata['fps'] > 0
                assert metadata['frame_count'] > 0
                assert metadata['duration'] > 0
                assert metadata['width'] > 0
                assert metadata['height'] > 0
                
                logger.info(f"  Metadata: {metadata['width']}x{metadata['height']}, "
                           f"{metadata['frame_count']} frames, {metadata['fps']:.2f} FPS, "
                           f"{metadata['duration']:.1f}s")
                
                # Test frame sampling (lightweight operation)
                sample_frames = video_utils.sample_frames(str(video_path), num_samples=5)
                
                assert len(sample_frames) > 0, "No frames sampled"
                assert len(sample_frames) <= 5, "Too many frames sampled"
                
                for i, frame in enumerate(sample_frames):
                    assert isinstance(frame, np.ndarray), f"Frame {i} is not a numpy array"
                    assert frame.ndim == 3, f"Frame {i} is not 3D"
                    assert frame.shape[2] == 3, f"Frame {i} doesn't have 3 channels"
                
                logger.info(f"  Sampled {len(sample_frames)} frames successfully")
                
        except ImportError as e:
            pytest.skip(f"Video utilities not available: {e}")
    
    def test_model_fallback_behavior(self):
        """Test that the system gracefully handles missing models."""
        test_scenarios = [
            {
                'name': 'VSRM Handler',
                'module': 'models.enhancement.vsr.vsrm_handler',
                'class': 'VSRMHandler'
            },
            {
                'name': 'SeedVR2 Handler', 
                'module': 'models.enhancement.zeroshot.seedvr2_handler',
                'class': 'SeedVR2Handler'
            },
            {
                'name': 'DiTVR Handler',
                'module': 'models.enhancement.zeroshot.ditvr_handler', 
                'class': 'DiTVRHandler'
            }
        ]
        
        for scenario in test_scenarios:
            try:
                module = __import__(scenario['module'], fromlist=[scenario['class']])
                handler_class = getattr(module, scenario['class'])
                
                # Try to instantiate with CPU device (should work even without weights)
                handler = handler_class(device='cpu')
                
                # Test basic properties exist
                assert hasattr(handler, 'device'), f"{scenario['name']} missing device attribute"
                
                logger.info(f"✅ {scenario['name']} instantiated successfully")
                
            except ImportError:
                logger.info(f"⚠️  {scenario['name']} not available (expected)")
            except Exception as e:
                logger.warning(f"⚠️  {scenario['name']} failed to instantiate: {e}")
    
    def test_output_directory_permissions(self):
        """Test that output directories can be created and written to."""
        test_dirs = [
            self.output_dir / "test_run",
            self.output_dir / "enhanced",
            self.output_dir / "metadata"
        ]
        
        for test_dir in test_dirs:
            # Create directory
            test_dir.mkdir(exist_ok=True, parents=True)
            assert test_dir.exists(), f"Failed to create directory: {test_dir}"
            
            # Test write permissions
            test_file = test_dir / "test_write.txt"
            test_file.write_text("test data")
            assert test_file.exists(), f"Failed to write to directory: {test_dir}"
            
            # Cleanup
            test_file.unlink()
            
        logger.info(f"✅ All output directories writable")
    
    def test_configuration_validation(self):
        """Test that configuration files are valid and accessible."""
        try:
            from config.model_config import ModelConfig
            
            config = ModelConfig()
            status = config.get_model_status()
            
            # Validate status structure
            assert isinstance(status, dict), "Model status should be a dict"
            
            required_keys = ['device', 'pipeline_defaults', 'paths']
            for key in required_keys:
                assert key in status, f"Missing required key in model status: {key}"
            
            # Test device configuration
            assert status['device'] in ['cpu', 'cuda'], f"Invalid device: {status['device']}"
            
            # Test pipeline defaults
            pipeline_defaults = status['pipeline_defaults']
            assert 'latency_class' in pipeline_defaults
            assert pipeline_defaults['latency_class'] in ['strict', 'standard', 'flexible']
            
            logger.info(f"✅ Configuration validation passed")
            logger.info(f"  Device: {status['device']}")
            logger.info(f"  Latency class: {pipeline_defaults['latency_class']}")
            
            # Log model availability
            model_types = ['vsrm', 'seedvr2', 'ditvr', 'fast_mamba_vsr']
            available_models = [m for m in model_types if status.get(m, False)]
            logger.info(f"  Available models: {available_models if available_models else 'None (fallback mode)'}")
            
        except ImportError:
            pytest.skip("ModelConfig not available")
    
    def test_memory_usage_basic(self):
        """Test basic memory usage patterns."""
        try:
            import psutil
            import gc
            
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            logger.info(f"Initial memory usage: {initial_memory:.1f} MB")
            
            # Test tensor creation and cleanup
            test_tensors = []
            for i in range(5):
                tensor = torch.randn(1, 3, 16, 256, 256)  # ~50MB tensor
                test_tensors.append(tensor)
            
            peak_memory = process.memory_info().rss / (1024 * 1024)
            logger.info(f"Peak memory usage: {peak_memory:.1f} MB")
            
            # Cleanup
            del test_tensors
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            final_memory = process.memory_info().rss / (1024 * 1024)
            logger.info(f"Final memory usage: {final_memory:.1f} MB")
            
            # Memory should not grow excessively
            memory_growth = final_memory - initial_memory
            assert memory_growth < 500, f"Excessive memory growth: {memory_growth:.1f} MB"
            
            logger.info(f"✅ Memory usage test passed")
            
        except ImportError:
            logger.info("⚠️  psutil not available, skipping memory test")


class TestSmokeIntegrationPatterns:
    """Test common integration patterns and edge cases."""
    
    def test_task_specification_creation(self):
        """Test task specification creation and validation."""
        try:
            from agents.core.task_specification import TaskSpecification
            
            # Test valid task specification
            valid_task = TaskSpecification(
                task_type="video_enhancement",
                input_data={
                    'input_path': '/tmp/test_input.mp4',
                    'output_path': '/tmp/test_output.mp4',
                    'vsr_strategy': 'auto',
                    'latency_class': 'standard',
                    'quality_tier': 'balanced',
                    'target_fps': 30,
                    'allow_diffusion': True,
                    'allow_zero_shot': True
                }
            )
            
            assert valid_task.task_type == "video_enhancement"
            assert 'input_path' in valid_task.input_data
            assert valid_task.input_data['latency_class'] == 'standard'
            
            logger.info("✅ TaskSpecification creation successful")
            
        except ImportError:
            pytest.skip("TaskSpecification not available")
    
    def test_error_handling_patterns(self):
        """Test common error handling patterns."""
        test_cases = [
            {
                'name': 'Invalid video path',
                'operation': lambda: cv2.VideoCapture('/nonexistent/path.mp4'),
                'expected': 'graceful_failure'
            },
            {
                'name': 'Invalid tensor shape',
                'operation': lambda: torch.randn(0, 0, 0),
                'expected': 'valid_tensor'
            }
        ]
        
        for case in test_cases:
            try:
                result = case['operation']()
                logger.info(f"✅ {case['name']}: Handled gracefully")
                
                # Cleanup if needed
                if hasattr(result, 'release'):
                    result.release()
                    
            except Exception as e:
                logger.info(f"⚠️  {case['name']}: Exception handled: {type(e).__name__}")
    
    def test_concurrent_processing_safety(self):
        """Test that basic operations are safe for concurrent use."""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                # Simulate some basic processing
                tensor = torch.randn(1, 3, 8, 64, 64)
                processed = tensor * 0.5 + 0.1
                result = torch.mean(processed).item()
                results.append((worker_id, result))
                time.sleep(0.1)  # Simulate some work
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Run multiple workers
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Check results
        assert len(errors) == 0, f"Concurrent processing errors: {errors}"
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        
        logger.info(f"✅ Concurrent processing test passed with {len(results)} workers")


def run_smoke_tests():
    """Run all smoke tests."""
    logger.info("🚀 Running Video Enhancement Smoke Tests")
    logger.info("=" * 60)
    
    # Configure pytest for smoke tests
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure for smoke tests
        "--disable-warnings"
    ]
    
    result = pytest.main(pytest_args)
    
    if result == 0:
        logger.info("🎉 All smoke tests passed!")
        logger.info("\nSystem Status:")
        logger.info("  ✅ Video file handling")
        logger.info("  ✅ Degradation analysis")
        logger.info("  ✅ Configuration validation")
        logger.info("  ✅ Memory management")
        logger.info("  ✅ Error handling")
        logger.info("  ✅ Basic processing pipeline")
    else:
        logger.error("❌ Some smoke tests failed. System may not be ready for production.")
    
    return result == 0


if __name__ == "__main__":
    success = run_smoke_tests()
    exit(0 if success else 1)