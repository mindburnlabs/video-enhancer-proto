#!/usr/bin/env python3
"""
Video Enhancement Agent - 2025 SOTA Edition
Intelligent video enhancement using VSRM, SeedVR2, and DiTVR
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


import asyncio
import logging
import time
import os
import tempfile
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

from agents.core import (
    BaseVideoProcessingAgent, AgentCapabilities, TaskSpecification,
    ProcessingResult, TaskStatus, Priority
)

# 2025 SOTA handlers
from models.enhancement.vsr.vsrm_handler import VSRMHandler
from models.enhancement.zeroshot.seedvr2_handler import SeedVR2Handler
from models.enhancement.zeroshot.ditvr_handler import DiTVRHandler
from models.enhancement.vsr.fast_mamba_vsr_handler import FastMambaVSRHandler
from models.enhancement.vsr.realesrgan_handler import RealESRGANHandler
from models.interpolation.rife_handler import RIFEHandler

logger = logging.getLogger(__name__)

class VideoEnhancementAgent(BaseVideoProcessingAgent):
    """
    Video Enhancement Agent using 2025 SOTA models:
    - VSRM for primary video super-resolution
    - SeedVR2 for one-step diffusion restoration
    - DiTVR for zero-shot transformer restoration
    - Fast Mamba VSR for ultra-efficient processing
    """
    
    def __init__(self, name: str = "video_enhancer_sota", device: str = "cuda", **kwargs):
        capabilities = AgentCapabilities(
            agent_type="enhancer",
            capabilities=[
                "video-super-resolution", "diffusion-restoration", 
                "zero-shot-enhancement", "mamba-processing"
            ],
            max_concurrent_tasks=1,
            cpu_cores=8,
            gpu_memory=16,  # More efficient than legacy models
            specialized_models=["VSRM", "SeedVR2", "DiTVR", "FastMambaVSR"]
        )
        
        super().__init__(name=name, capabilities=capabilities, **kwargs)
        
        # Initialize SOTA handlers
        self.vsrm_handler = VSRMHandler(device=device)
        self.seedvr2_handler = SeedVR2Handler(device=device)
        self.ditvr_handler = DiTVRHandler(device=device)
        self.fast_mamba_handler = FastMambaVSRHandler(device=device)
        self.realesrgan_handler = RealESRGANHandler(device=device)
        self.rife_handler = RIFEHandler(device=device)
        self.device = device
        
        # Enhancement configuration
        self.default_config = {
            'primary_model': 'vsrm',
            'quality_threshold': 0.7,
            'use_diffusion_restoration': True,
            'enable_zero_shot': True,
            'fast_mode': False,
            'enhancement_factor': 4
        }
        
        # Model selection strategy
        self.model_strategy = {
            'high_quality': 'vsrm',
            'fast_processing': 'fast_mamba_vsr',
            'restoration': 'seedvr2',
            'unknown_degradation': 'ditvr',
            'super_resolution': 'realesrgan',
            'frame_interpolation': 'rife'
        }
        
        # Enhancement statistics
        self.enhancement_stats = {
            'total_enhancements': 0,
            'successful_enhancements': 0,
            'failed_enhancements': 0,
            'model_usage': {
                'vsrm': 0, 'seedvr2': 0, 'ditvr': 0, 'fast_mamba_vsr': 0,
                'realesrgan': 0, 'rife': 0
            },
            'average_processing_time': 0.0,
            'total_frames_processed': 0
        }
        
        logger.info(f"SOTA Video Enhancement Agent initialized on {device}")
    
    async def process_task(self, task: TaskSpecification) -> ProcessingResult:
        """Process video enhancement tasks using SOTA models"""
        start_time = time.time()
        
        try:
            if task.task_type == "video_enhancement":
                return await self._handle_video_enhancement(task)
            elif task.task_type == "quality_restoration":
                return await self._handle_quality_restoration(task)
            elif task.task_type == "zero_shot_enhancement":
                return await self._handle_zero_shot_enhancement(task)
            elif task.task_type == "fast_enhancement":
                return await self._handle_fast_enhancement(task)
            else:
                return ProcessingResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error_message=f"Unknown enhancement task type: {task.task_type}",
                    processing_time=time.time() - start_time
                )
                
        except Exception as e:
            logger.error(f"SOTA enhancement task failed: {e}")
            self.enhancement_stats['failed_enhancements'] += 1
            return ProcessingResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    async def _handle_video_enhancement(self, task: TaskSpecification) -> ProcessingResult:
        """Handle video enhancement with automatic model selection"""
        start_time = time.time()
        input_data = task.input_data
        
        video_path = input_data.get('video_path', '')
        output_path = input_data.get('output_path', '')
        analysis_result = input_data.get('analysis_result', {})
        user_preferences = input_data.get('user_preferences', {})
        
        if not video_path or not os.path.exists(video_path):
            return ProcessingResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error_message=f"Video file not found: {video_path}",
                processing_time=time.time() - start_time
            )
        
        if not output_path:
            output_path = self._generate_output_path(video_path)
        
        try:
            self.enhancement_stats['total_enhancements'] += 1
            
            # Select optimal model based on analysis
            selected_model = self._select_model(analysis_result, user_preferences)
            
            logger.info(f"Starting SOTA enhancement with {selected_model}")
            
            # Execute enhancement
            enhancement_result = await self._execute_sota_enhancement(
                video_path, output_path, selected_model, analysis_result
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self.enhancement_stats['successful_enhancements'] += 1
            self.enhancement_stats['model_usage'][selected_model] += 1
            
            return ProcessingResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                output_data=enhancement_result,
                processing_time=processing_time,
                metadata={
                    'model_used': selected_model,
                    'enhancement_type': '2025_sota',
                    'frames_processed': enhancement_result.get('output_frames', 0)
                }
            )
            
        except Exception as e:
            self.enhancement_stats['failed_enhancements'] += 1
            logger.error(f"SOTA enhancement failed: {e}")
            raise e
    
    async def _handle_quality_restoration(self, task: TaskSpecification) -> ProcessingResult:
        """Handle quality restoration using SeedVR2"""
        input_data = task.input_data
        video_path = input_data['video_path']
        output_path = input_data.get('output_path', self._generate_output_path(video_path))
        
        logger.info("Using SeedVR2 for quality restoration")
        
        result = self.seedvr2_handler.restore_video(
            input_path=video_path,
            output_path=output_path,
            quality_threshold=input_data.get('quality_threshold', 0.5)
        )
        
        return ProcessingResult(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            output_data={'output_path': output_path, 'stats': result},
            metadata={'model_used': 'seedvr2', 'task_type': 'quality_restoration'}
        )
    
    async def _handle_zero_shot_enhancement(self, task: TaskSpecification) -> ProcessingResult:
        """Handle zero-shot enhancement using DiTVR"""
        input_data = task.input_data
        video_path = input_data['video_path']
        output_path = input_data.get('output_path', self._generate_output_path(video_path))
        
        logger.info("Using DiTVR for zero-shot enhancement")
        
        result = self.ditvr_handler.restore_video(
            input_path=video_path,
            output_path=output_path,
            degradation_type=input_data.get('degradation_type', 'unknown'),
            auto_adapt=True
        )
        
        return ProcessingResult(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            output_data={'output_path': output_path, 'stats': result},
            metadata={'model_used': 'ditvr', 'task_type': 'zero_shot_enhancement'}
        )
    
    async def _handle_fast_enhancement(self, task: TaskSpecification) -> ProcessingResult:
        """Handle fast enhancement using Fast Mamba VSR"""
        input_data = task.input_data
        video_path = input_data['video_path']
        output_path = input_data.get('output_path', self._generate_output_path(video_path))
        
        logger.info("Using Fast Mamba VSR for efficient enhancement")
        
        result = self.fast_mamba_handler.enhance_video(
            input_path=video_path,
            output_path=output_path,
            chunk_size=input_data.get('chunk_size', 16),
            fp16=True
        )
        
        return ProcessingResult(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            output_data={'output_path': output_path, 'stats': result},
            metadata={'model_used': 'fast_mamba_vsr', 'task_type': 'fast_enhancement'}
        )
    
    def _select_model(self, analysis_result: Dict[str, Any], user_preferences: Dict[str, Any]) -> str:
        """Select the optimal SOTA model based on analysis and preferences"""
        
        # User preference override
        if user_preferences.get('force_model'):
            return user_preferences['force_model']
        
        # Fast mode preference
        if user_preferences.get('fast_mode', False):
            return 'fast_mamba_vsr'
        
        # Analysis-based selection
        quality_score = analysis_result.get('quality_score', 0.5)
        motion_intensity = analysis_result.get('motion_intensity', 0.5)
        degradation_type = analysis_result.get('degradation_type', 'unknown')
        task_type = analysis_result.get('task_type', 'enhancement')
        
        # Frame interpolation request
        if task_type == 'interpolation' or user_preferences.get('interpolation', False):
            return 'rife'
        
        # Low quality videos need restoration
        if quality_score < 0.4:
            return 'seedvr2'
        
        # Super-resolution focused tasks
        if task_type == 'super_resolution' or user_preferences.get('super_resolution', False):
            return 'realesrgan'
        
        # Unknown degradation needs zero-shot capability
        if degradation_type == 'unknown' or degradation_type not in ['blur', 'noise', 'compression']:
            return 'ditvr'
        
        # High motion videos need advanced processing
        if motion_intensity > 0.7:
            return 'vsrm'
        
        # Default to Real-ESRGAN for general super-resolution (reliable weights)
        return 'realesrgan'
    
    async def _execute_sota_enhancement(self, video_path: str, output_path: str, 
                                      model_name: str, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhancement using selected SOTA model"""
        
        if model_name == 'vsrm':
            return self.vsrm_handler.enhance_video(
                input_path=video_path,
                output_path=output_path,
                window=7,
                fp16=True
            )
        
        elif model_name == 'seedvr2':
            return self.seedvr2_handler.restore_video(
                input_path=video_path,
                output_path=output_path,
                quality_threshold=0.6
            )
        
        elif model_name == 'ditvr':
            degradation_type = analysis_result.get('degradation_type', 'unknown')
            return self.ditvr_handler.restore_video(
                input_path=video_path,
                output_path=output_path,
                degradation_type=degradation_type,
                auto_adapt=True
            )
        
        elif model_name == 'fast_mamba_vsr':
            return self.fast_mamba_handler.enhance_video(
                input_path=video_path,
                output_path=output_path,
                chunk_size=16,
                fp16=True
            )
        
        elif model_name == 'realesrgan':
            return self.realesrgan_handler.restore_video(
                input_path=video_path,
                output_path=output_path
            )
        
        elif model_name == 'rife':
            return self.rife_handler.interpolate_video(
                input_path=video_path,
                output_path=output_path,
                interpolation_factor=2
            )
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _generate_output_path(self, input_path: str) -> str:
        """Generate output path for enhanced video"""
        input_path = Path(input_path)
        return str(input_path.parent / f"{input_path.stem}_sota_enhanced{input_path.suffix}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        return {
            'enhancement_stats': self.enhancement_stats,
            'model_info': {
                'vsrm': self.vsrm_handler.get_model_info(),
                'seedvr2': self.seedvr2_handler.get_model_info(),
                'ditvr': self.ditvr_handler.get_model_info(),
                'fast_mamba_vsr': self.fast_mamba_handler.get_model_info(),
                'realesrgan': self.realesrgan_handler.get_model_info(),
                'rife': self.rife_handler.get_model_info()
            },
            'agent_capabilities': self.capabilities.__dict__
        }
    
    async def benchmark_models(self) -> Dict[str, Any]:
        """Benchmark all SOTA models"""
        results = {}
        
        # Benchmark Fast Mamba VSR
        try:
            results['fast_mamba_vsr'] = self.fast_mamba_handler.benchmark_performance(test_frames=50)
        except Exception as e:
            results['fast_mamba_vsr'] = {'error': str(e)}
        
        return results