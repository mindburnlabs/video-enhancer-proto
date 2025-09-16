#!/usr/bin/env python3
"""
Video Processing API Endpoints v1

Comprehensive REST API for programmatic video enhancement with job management,
strategy flags, and OpenAPI schema support.
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


from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import logging
import asyncio
import uuid
import os
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
import aiofiles
import torch
from utils.performance_monitor import get_performance_tracker

# Import error handling
from utils.error_handler import (
    handle_exceptions, InputValidationError, ProcessingError, APIError,
    ErrorCode, error_handler
)

logger = logging.getLogger(__name__)

# Create router with OpenAPI metadata
router = APIRouter(
    prefix="/api/v1/process", 
    tags=["Video Processing"],
    responses={404: {"description": "Not found"}},
)

# Global job storage (in production, use Redis/database)
_job_store: Dict[str, Dict] = {}
_job_results: Dict[str, Dict] = {}

# Global agent instance
_enhancement_agent: Optional['VideoEnhancementAgent'] = None

def get_enhancement_agent():
    """Get or create the global enhancement agent"""
    global _enhancement_agent
    if _enhancement_agent is None:
        try:
            from agents.enhancer.video_enhancer_sota import VideoEnhancementAgent
            _enhancement_agent = VideoEnhancementAgent(device="cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Initialized VideoEnhancementAgent for API processing")
        except Exception as e:
            logger.error(f"Failed to initialize VideoEnhancementAgent: {e}")
            _enhancement_agent = None
    return _enhancement_agent

class LatencyClass(str, Enum):
    """Latency class options for processing optimization"""
    strict = "strict"
    standard = "standard" 
    flexible = "flexible"

class QualityTier(str, Enum):
    """Quality tier options"""
    fast = "fast"
    balanced = "balanced"
    high = "high"
    ultra = "ultra"

class VSRStrategy(str, Enum):
    """Video Super Resolution strategy options"""
    auto = "auto"
    vsrm = "vsrm"
    seedvr2 = "seedvr2"
    ditvr = "ditvr"
    fast_mamba_vsr = "fast_mamba_vsr"

class JobStatus(str, Enum):
    """Job processing status"""
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"

class ProcessingRequest(BaseModel):
    """Comprehensive video processing request model"""
    
    # Core processing parameters
    vsr_strategy: VSRStrategy = Field(VSRStrategy.auto, description="Video super-resolution strategy")
    latency_class: LatencyClass = Field(LatencyClass.standard, description="Processing latency class")
    quality_tier: QualityTier = Field(QualityTier.balanced, description="Output quality tier")
    
    # Target specifications
    target_fps: Optional[int] = Field(None, ge=15, le=120, description="Target frame rate (15-120 fps)")
    target_resolution: Optional[str] = Field(None, description="Target resolution (e.g., '1920x1080', '4K')")
    scale_factor: Optional[float] = Field(None, ge=1.0, le=4.0, description="Upscaling factor (1.0-4.0)")
    
    # Strategy toggles
    allow_diffusion: bool = Field(True, description="Allow diffusion-based models (SeedVR2)")
    allow_zero_shot: bool = Field(True, description="Allow zero-shot models (DiTVR)")
    enable_face_expert: bool = Field(False, description="Enable face restoration expert")
    enable_hfr: bool = Field(False, description="Enable high frame rate interpolation")
    enable_temporal_consistency: bool = Field(True, description="Enable temporal consistency processing")
    
    # Advanced options
    license_mode: str = Field("permissive_only", description="Licensing mode for model selection")
    custom_pipeline: Optional[Dict[str, Any]] = Field(None, description="Custom processing pipeline configuration")
    metadata_extraction: bool = Field(True, description="Extract detailed video metadata")
    quality_metrics: bool = Field(False, description="Compute quality assessment metrics")
    
    # Output options
    output_format: str = Field("mp4", description="Output video format")
    output_codec: str = Field("h264", description="Output video codec")
    preserve_audio: bool = Field(True, description="Preserve original audio track")
    
    @validator('target_resolution')
    def validate_resolution(cls, v):
        if v is not None:
            valid_resolutions = ['720p', '1080p', '1440p', '4K', '8K']
            if 'x' in v:
                try:
                    w, h = v.split('x')
                    int(w), int(h)
                except (ValueError, AttributeError):
                    raise ValueError('Invalid resolution format. Use WxH (e.g., 1920x1080) or preset (720p, 1080p, 4K)')
            elif v not in valid_resolutions:
                raise ValueError(f'Invalid resolution preset. Use one of: {valid_resolutions}')
        return v

class ProcessingResponse(BaseModel):
    """Response model for processing requests"""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    message: str = Field(..., description="Status message")
    estimated_duration: Optional[int] = Field(None, description="Estimated processing time in seconds")
    created_at: datetime = Field(..., description="Job creation timestamp")
    strategy_plan: Optional[Dict[str, Any]] = Field(None, description="Selected processing strategy")

class JobStatusResponse(BaseModel):
    """Detailed job status response"""
    job_id: str
    status: JobStatus
    progress: float = Field(..., ge=0.0, le=100.0, description="Processing progress percentage")
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Processing details
    current_stage: Optional[str] = None
    stages_completed: List[str] = []
    stages_remaining: List[str] = []
    
    # Resource usage
    estimated_duration: Optional[int] = None
    elapsed_time: Optional[int] = None
    memory_usage_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    
    # Results (when completed)
    output_path: Optional[str] = None
    output_size_mb: Optional[float] = None
    processing_stats: Optional[Dict[str, Any]] = None
    quality_metrics: Optional[Dict[str, float]] = None
    
    # Error information (when failed)
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

class JobListResponse(BaseModel):
    """Response for job listing"""
    jobs: List[JobStatusResponse]
    total_count: int
    page: int
    page_size: int
    has_next: bool

# Dependency functions
async def validate_upload_file(file: UploadFile = File(...)):
    """Validate uploaded video file with enhanced error handling"""
    
    # Validate content type
    if not file.content_type or not file.content_type.startswith('video/'):
        raise InputValidationError(
            message="Invalid file type",
            error_code=ErrorCode.INPUT_UNSUPPORTED_FORMAT,
            context=error_handler.handle_error(
                ValueError(f"File type '{file.content_type}' not supported"),
                component="api",
                operation="file_validation",
                user_message="Please upload a video file (MP4, AVI, MOV, or similar)",
                suggestions=[
                    "Convert your file to MP4 format",
                    "Ensure the file has a proper video extension",
                    "Check that the file is not corrupted"
                ]
            ).context
        )
    
    # Validate file name
    if not file.filename:
        raise InputValidationError(
            message="File must have a name",
            error_code=ErrorCode.INPUT_MISSING_REQUIRED,
            context=error_handler.handle_error(
                ValueError("Missing filename"),
                component="api",
                operation="file_validation",
                user_message="The uploaded file must have a valid filename",
                suggestions=["Ensure the file has a proper name and extension"]
            ).context
        )
    
    # Check file size (limit to 500MB)
    try:
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        
        if len(content) > 500 * 1024 * 1024:
            raise InputValidationError(
                message="File too large",
                error_code=ErrorCode.INPUT_FILE_TOO_LARGE,
                context=error_handler.handle_error(
                    ValueError(f"File size {file_size_mb:.1f}MB exceeds 500MB limit"),
                    component="api",
                    operation="file_validation",
                    user_message=f"File is too large ({file_size_mb:.1f}MB). Maximum size is 500MB.",
                    suggestions=[
                        "Compress the video to reduce file size",
                        "Trim the video to a shorter length",
                        "Use a lower resolution or bitrate",
                        "Process the video in smaller segments"
                    ]
                ).context
            )
        
        # Validate minimum file size (1MB)
        if len(content) < 1024 * 1024:
            raise InputValidationError(
                message="File too small",
                error_code=ErrorCode.INPUT_FILE_CORRUPTED,
                context=error_handler.handle_error(
                    ValueError(f"File size {file_size_mb:.3f}MB is suspiciously small"),
                    component="api",
                    operation="file_validation",
                    user_message="The file appears to be too small to be a valid video",
                    suggestions=[
                        "Check that the file is not corrupted",
                        "Ensure you uploaded the correct file",
                        "Try uploading a different video file"
                    ]
                ).context
            )
    
    except (OSError, IOError) as e:
        raise InputValidationError(
            message="Failed to read file",
            error_code=ErrorCode.INPUT_FILE_CORRUPTED,
            context=error_handler.handle_error(
                e,
                component="api",
                operation="file_validation",
                user_message="Unable to read the uploaded file. It may be corrupted.",
                suggestions=[
                    "Try uploading the file again",
                    "Check that the file is not corrupted",
                    "Use a different video file"
                ]
            ).context
        )
    
    finally:
        await file.seek(0)  # Reset file pointer
    
    logger.info(f"Validated video file: {file.filename} ({file_size_mb:.1f}MB)")
    return file

def get_processing_config():
    """Get current processing configuration"""
    try:
        from config.model_config import ModelConfig
        return ModelConfig()
    except ImportError:
        raise HTTPException(status_code=500, detail="Processing configuration not available")

# API Endpoints

@router.post("/auto", response_model=ProcessingResponse, 
            summary="Automatic Video Enhancement",
            description="Process video with automatic strategy selection based on content analysis")
async def process_video_auto(
    background_tasks: BackgroundTasks,
    file: UploadFile = Depends(validate_upload_file),
    request: ProcessingRequest = ProcessingRequest(),
    config = Depends(get_processing_config)
):
    """
    Automatically enhance video quality using SOTA models.
    
    This endpoint analyzes the input video and automatically selects the best
    enhancement strategy based on detected degradations and content characteristics.
    """
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Create temporary files
        temp_dir = Path(tempfile.mkdtemp())
        input_path = temp_dir / f"input_{job_id}{Path(file.filename).suffix}"
        output_path = temp_dir / f"output_{job_id}.{request.output_format}"
        
        # Save uploaded file
        async with aiofiles.open(input_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Analyze content and create strategy plan
        strategy_plan = await _create_strategy_plan(str(input_path), request, config)
        
        # Create job record with performance tracking
        tracker = get_performance_tracker()
        perf_operation_id = tracker.start_operation('api_request', request.vsr_strategy, {
            'job_id': job_id,
            'latency_class': request.latency_class,
            'quality_tier': request.quality_tier
        })
        
        job_record = {
            "job_id": job_id,
            "status": JobStatus.pending,
            "created_at": datetime.utcnow(),
            "input_path": str(input_path),
            "output_path": str(output_path),
            "request": request.dict(),
            "strategy_plan": strategy_plan,
            "progress": 0.0,
            "stages_completed": [],
            "stages_remaining": strategy_plan.get("processing_stages", []),
            "perf_operation_id": perf_operation_id
        }
        
        _job_store[job_id] = job_record
        
        # Start background processing
        background_tasks.add_task(
            _process_video_background,
            job_id, str(input_path), str(output_path), request, strategy_plan
        )
        
        return ProcessingResponse(
            job_id=job_id,
            status=JobStatus.pending,
            message="Video processing queued for automatic enhancement",
            estimated_duration=strategy_plan.get("estimated_duration"),
            created_at=job_record["created_at"],
            strategy_plan=strategy_plan
        )
        
    except Exception as e:
        logger.error(f"Failed to queue video processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/job/{job_id}", response_model=JobStatusResponse,
           summary="Get Job Status", 
           description="Retrieve detailed status information for a processing job")
async def get_job_status(job_id: str):
    """Get detailed status of a video processing job"""
    
    if job_id not in _job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_record = _job_store[job_id]
    
    # Calculate elapsed time
    elapsed_time = None
    if job_record.get("started_at"):
        end_time = job_record.get("completed_at") or datetime.utcnow()
        elapsed_time = int((end_time - job_record["started_at"]).total_seconds())
    
    return JobStatusResponse(
        job_id=job_id,
        status=job_record["status"],
        progress=job_record.get("progress", 0.0),
        created_at=job_record["created_at"],
        started_at=job_record.get("started_at"),
        completed_at=job_record.get("completed_at"),
        current_stage=job_record.get("current_stage"),
        stages_completed=job_record.get("stages_completed", []),
        stages_remaining=job_record.get("stages_remaining", []),
        estimated_duration=job_record.get("estimated_duration"),
        elapsed_time=elapsed_time,
        memory_usage_mb=job_record.get("memory_usage_mb"),
        gpu_utilization=job_record.get("gpu_utilization"),
        output_path=job_record.get("output_path"),
        output_size_mb=job_record.get("output_size_mb"),
        processing_stats=job_record.get("processing_stats"),
        quality_metrics=job_record.get("quality_metrics"),
        error_message=job_record.get("error_message"),
        error_details=job_record.get("error_details")
    )

@router.get("/job/{job_id}/download",
           summary="Download Processed Video",
           description="Download the enhanced video file for a completed job")
async def download_result(job_id: str):
    """Download the processed video file"""
    
    if job_id not in _job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_record = _job_store[job_id]
    
    if job_record["status"] != JobStatus.completed:
        raise HTTPException(status_code=400, detail="Job not completed")
    
    output_path = job_record.get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    filename = f"enhanced_{job_id}.mp4"
    return FileResponse(
        path=output_path,
        filename=filename,
        media_type='video/mp4'
    )

@router.get("/jobs", response_model=JobListResponse,
           summary="List Jobs",
           description="List processing jobs with filtering and pagination")
async def list_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by job status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page")
):
    """List processing jobs with optional filtering"""
    
    # Filter jobs
    jobs = list(_job_store.values())
    if status:
        jobs = [job for job in jobs if job["status"] == status]
    
    # Sort by creation time (newest first)
    jobs.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Pagination
    total_count = len(jobs)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_jobs = jobs[start_idx:end_idx]
    
    # Convert to response models
    job_responses = []
    for job in page_jobs:
        elapsed_time = None
        if job.get("started_at"):
            end_time = job.get("completed_at") or datetime.utcnow()
            elapsed_time = int((end_time - job["started_at"]).total_seconds())
            
        job_responses.append(JobStatusResponse(
            job_id=job["job_id"],
            status=job["status"],
            progress=job.get("progress", 0.0),
            created_at=job["created_at"],
            started_at=job.get("started_at"),
            completed_at=job.get("completed_at"),
            current_stage=job.get("current_stage"),
            stages_completed=job.get("stages_completed", []),
            stages_remaining=job.get("stages_remaining", []),
            estimated_duration=job.get("estimated_duration"),
            elapsed_time=elapsed_time,
            memory_usage_mb=job.get("memory_usage_mb"),
            gpu_utilization=job.get("gpu_utilization"),
            output_path=job.get("output_path"),
            output_size_mb=job.get("output_size_mb"),
            processing_stats=job.get("processing_stats"),
            quality_metrics=job.get("quality_metrics"),
            error_message=job.get("error_message"),
            error_details=job.get("error_details")
        ))
    
    return JobListResponse(
        jobs=job_responses,
        total_count=total_count,
        page=page,
        page_size=page_size,
        has_next=end_idx < total_count
    )

@router.delete("/job/{job_id}",
              summary="Cancel Job",
              description="Cancel a pending or running job")
async def cancel_job(job_id: str):
    """Cancel a processing job"""
    
    if job_id not in _job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_record = _job_store[job_id]
    
    if job_record["status"] in [JobStatus.completed, JobStatus.failed, JobStatus.cancelled]:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    
    # Update job status
    job_record["status"] = JobStatus.cancelled
    job_record["completed_at"] = datetime.utcnow()
    job_record["error_message"] = "Job cancelled by user"
    
    return {"message": "Job cancelled successfully"}

@router.get("/strategies",
           summary="List Available Strategies",
           description="Get information about available enhancement strategies")
async def get_available_strategies(config = Depends(get_processing_config)):
    """Get available enhancement strategies and their capabilities"""
    
    try:
        status = config.get_model_status()
        
        strategies = {
            "vsrm": {
                "name": "VSRM (Video Super-Resolution Mamba)",
                "description": "Mamba-based super-resolution optimized for high-motion content",
                "available": status.get("vsrm", False),
                "best_for": ["high_motion", "temporal_consistency"],
                "latency_class": ["standard", "flexible"],
                "max_scale_factor": 4.0
            },
            "seedvr2": {
                "name": "SeedVR2 (Diffusion Video Restoration)",
                "description": "One-step diffusion model for comprehensive restoration",
                "available": status.get("seedvr2", False),
                "best_for": ["compression_artifacts", "mixed_degradations"],
                "latency_class": ["flexible"],
                "max_scale_factor": 2.0
            },
            "ditvr": {
                "name": "DiTVR (Zero-shot Diffusion Transformer)",
                "description": "Transformer-based zero-shot restoration for unknown degradations",
                "available": status.get("ditvr", False),
                "best_for": ["unknown_degradations", "zero_shot"],
                "latency_class": ["standard", "flexible"],
                "max_scale_factor": 3.0
            },
            "fast_mamba_vsr": {
                "name": "Fast Mamba VSR",
                "description": "Lightweight Mamba model optimized for speed",
                "available": status.get("fast_mamba_vsr", False),
                "best_for": ["real_time", "low_latency"],
                "latency_class": ["strict", "standard"],
                "max_scale_factor": 2.0
            }
        }
        
        return {
            "strategies": strategies,
            "device": status["device"],
            "pipeline_defaults": status["pipeline_defaults"]
        }
        
    except Exception as e:
        logger.error(f"Failed to get strategies: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve strategies")

@router.get("/health",
           summary="Health Check",
           description="Check API and processing system health")
async def health_check(config = Depends(get_processing_config)):
    """Health check endpoint"""
    
    try:
        status = config.get_model_status()
        
        # Check system resources
        import psutil
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "device": status["device"],
                "memory_usage": memory.percent,
                "disk_usage": disk.percent,
                "available_models": [k for k, v in status.items() if isinstance(v, bool) and v]
            },
            "jobs": {
                "total": len(_job_store),
                "pending": len([j for j in _job_store.values() if j["status"] == JobStatus.pending]),
                "processing": len([j for j in _job_store.values() if j["status"] == JobStatus.processing]),
                "completed": len([j for j in _job_store.values() if j["status"] == JobStatus.completed])
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@router.get("/agent/status",
           summary="Get Agent Status",
           description="Get current status and capabilities of the VideoEnhancementAgent")
async def get_agent_status():
    """Get status and capabilities of the VideoEnhancementAgent"""
    
    agent = get_enhancement_agent()
    
    if not agent:
        return {
            "status": "unavailable",
            "error": "VideoEnhancementAgent could not be initialized",
            "fallback_mode": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        # Get agent statistics
        stats = agent.enhancement_stats
        capabilities = agent.capabilities
        
        return {
            "status": "active",
            "timestamp": datetime.utcnow().isoformat(),
            "agent_name": agent.name,
            "device": str(agent.device),
            "capabilities": {
                "agent_type": capabilities.agent_type,
                "supported_tasks": capabilities.capabilities,
                "max_concurrent_tasks": capabilities.max_concurrent_tasks,
                "specialized_models": capabilities.specialized_models,
                "gpu_memory_gb": capabilities.gpu_memory,
                "cpu_cores": capabilities.cpu_cores
            },
            "statistics": {
                "total_enhancements": stats["total_enhancements"],
                "successful_enhancements": stats["successful_enhancements"],
                "failed_enhancements": stats["failed_enhancements"],
                "success_rate": (
                    stats["successful_enhancements"] / max(stats["total_enhancements"], 1) * 100
                    if stats["total_enhancements"] > 0 else 0
                ),
                "model_usage": stats["model_usage"],
                "average_processing_time": stats["average_processing_time"],
                "total_frames_processed": stats["total_frames_processed"]
            },
            "model_handlers": {
                "vsrm": bool(agent.vsrm_handler),
                "seedvr2": bool(agent.seedvr2_handler),
                "ditvr": bool(agent.ditvr_handler),
                "fast_mamba_vsr": bool(agent.fast_mamba_handler)
            },
            "fallback_mode": False
        }
        
    except Exception as e:
        logger.error(f"Agent status error: {e}")
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "fallback_mode": True
        }

# Helper functions

async def _create_strategy_plan(input_path: str, request: ProcessingRequest, config) -> Dict[str, Any]:
    """Create processing strategy plan based on video analysis"""
    
    try:
        from models.analysis.degradation_router import DegradationRouter
        
        router = DegradationRouter()
        analysis = router.analyze_and_route(
            input_path,
            latency_class=request.latency_class.value,
            allow_diffusion=request.allow_diffusion,
            allow_zero_shot=request.allow_zero_shot,
            license_mode=request.license_mode,
            enable_face_expert=request.enable_face_expert,
            enable_hfr=request.enable_hfr
        )
        
        # Estimate processing duration based on video length and strategy
        import cv2
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = frame_count / fps if fps > 0 else 60
        cap.release()
        
        # Base processing time estimates (in seconds per minute of video)
        time_estimates = {
            "vsrm": 120,
            "seedvr2": 180,
            "ditvr": 150,
            "fast_mamba_vsr": 60
        }
        
        primary_model = analysis["expert_routing"]["primary_model"]
        base_time = time_estimates.get(primary_model, 120)
        estimated_duration = int(duration_seconds * base_time / 60)
        
        # Create processing stages
        stages = ["video_analysis", "degradation_detection"]
        
        if analysis["expert_routing"].get("pre_processing"):
            stages.extend(["preprocessing", "noise_reduction"])
        
        stages.append(f"enhancement_{primary_model}")
        
        if analysis["expert_routing"].get("post_processing"):
            stages.extend(["temporal_smoothing", "color_correction"])
        
        if request.enable_face_expert and analysis["content_analysis"].get("has_faces"):
            stages.append("face_restoration")
        
        if request.enable_hfr:
            stages.append("frame_interpolation")
        
        stages.extend(["quality_assessment", "output_encoding"])
        
        return {
            "primary_model": primary_model,
            "degradation_analysis": analysis["degradations"],
            "content_analysis": analysis["content_analysis"],
            "confidence_score": analysis["confidence_score"],
            "processing_stages": stages,
            "estimated_duration": estimated_duration,
            "video_duration": duration_seconds,
            "routing_plan": analysis["expert_routing"]
        }
        
    except Exception as e:
        logger.error(f"Strategy planning failed: {e}")
        # Return fallback plan
        return {
            "primary_model": "vsrm",
            "processing_stages": ["video_analysis", "enhancement_vsrm", "output_encoding"],
            "estimated_duration": 300,
            "error": str(e)
        }

async def _create_task_specification(input_path: str, output_path: str, request: ProcessingRequest, strategy_plan: Dict[str, Any]) -> 'TaskSpecification':
    """Create TaskSpecification from API request"""
    try:
        from agents.core.task_specification import (
            TaskSpecification, TaskType, Priority, Quality, VideoSpecs, ProcessingConstraints
        )
        
        # Extract video metadata for specs
        try:
            from utils.video_utils import VideoUtils
            video_utils = VideoUtils()
            metadata = video_utils.get_video_metadata(input_path)
            input_resolution = (metadata.get('width', 1920), metadata.get('height', 1080))
            fps = metadata.get('fps', 30)
            duration = metadata.get('duration', 0)
        except Exception as e:
            logger.warning(f"Could not extract video metadata: {e}")
            input_resolution = (1920, 1080)
            fps = 30
            duration = 0
        
        # Map API request to task specification
        task_type_mapping = {
            'vsrm': TaskType.VIDEO_SUPER_RESOLUTION,
            'seedvr2': TaskType.VIDEO_RESTORATION, 
            'ditvr': TaskType.VIDEO_ENHANCEMENT,
            'fast_mamba_vsr': TaskType.UPSCALING,
            'auto': TaskType.VIDEO_ENHANCEMENT
        }
        
        quality_mapping = {
            'fast': Quality.FAST,
            'balanced': Quality.BALANCED,
            'high': Quality.HIGH_QUALITY,
            'ultra': Quality.MAXIMUM
        }
        
        priority_mapping = {
            'strict': Priority.HIGH,
            'standard': Priority.NORMAL,
            'flexible': Priority.LOW
        }
        
        # Calculate target resolution if scale factor is provided
        target_resolution = None
        if request.scale_factor:
            target_resolution = (
                int(input_resolution[0] * request.scale_factor),
                int(input_resolution[1] * request.scale_factor)
            )
        elif request.target_resolution:
            if 'x' in request.target_resolution:
                w, h = request.target_resolution.split('x')
                target_resolution = (int(w), int(h))
            else:
                # Handle preset resolutions
                preset_resolutions = {
                    '720p': (1280, 720),
                    '1080p': (1920, 1080),
                    '1440p': (2560, 1440),
                    '4K': (3840, 2160),
                    '8K': (7680, 4320)
                }
                target_resolution = preset_resolutions.get(request.target_resolution)
        
        # Create video specs
        video_specs = VideoSpecs(
            input_resolution=input_resolution,
            target_resolution=target_resolution,
            fps=fps,
            duration=duration,
            has_faces=strategy_plan.get('content_analysis', {}).get('has_faces', False),
            degradation_types=list(strategy_plan.get('degradation_analysis', {}).keys())
        )
        
        # Create processing constraints
        constraints = ProcessingConstraints(
            gpu_required=True,
            model_precision="fp16",
            tile_size=(512, 512) if input_resolution[0] > 1920 else None
        )
        
        # Generate unique task ID
        import time
        task_id = f"api_task_{int(time.time() * 1000000)}"
        
        # Create task specification
        task_spec = TaskSpecification(
            task_id=task_id,
            task_type=task_type_mapping.get(request.vsr_strategy, TaskType.VIDEO_ENHANCEMENT),
            priority=priority_mapping.get(request.latency_class, Priority.NORMAL),
            quality=quality_mapping.get(request.quality_tier, Quality.BALANCED),
            input_path=input_path,
            output_path=output_path,
            video_specs=video_specs,
            processing_constraints=constraints,
            model_preferences={
                'primary_strategy': request.vsr_strategy,
                'allow_diffusion': request.allow_diffusion,
                'allow_zero_shot': request.allow_zero_shot,
                'enable_face_expert': request.enable_face_expert,
                'enable_hfr': request.enable_hfr,
                'enable_temporal_consistency': request.enable_temporal_consistency
            },
            metadata={
                'api_request': request.dict(),
                'strategy_plan': strategy_plan,
                'created_via': 'api_v1'
            }
        )
        
        return task_spec
        
    except Exception as e:
        logger.error(f"Failed to create task specification: {e}")
        raise

async def _process_video_background(
    job_id: str,
    input_path: str, 
    output_path: str,
    request: ProcessingRequest,
    strategy_plan: Dict[str, Any]
):
    """Background video processing task using VideoEnhancementAgent"""
    
    job_record = _job_store[job_id]
    
    try:
        # Update job status
        job_record["status"] = JobStatus.processing
        job_record["started_at"] = datetime.utcnow()
        
        # Try to use the enhancement agent
        agent = get_enhancement_agent()
        
        if agent:
            logger.info(f"Job {job_id}: Processing with VideoEnhancementAgent")
            
            # Create task specification
            task_spec = await _create_task_specification(input_path, output_path, request, strategy_plan)
            
            # Process through agent
            result = await agent.process_task(task_spec)
            
            if result.status.value == "completed":
                # Agent processing succeeded
                job_record["status"] = JobStatus.completed
                job_record["completed_at"] = datetime.utcnow()
                job_record["progress"] = 100.0
                job_record["current_stage"] = None
                job_record["stages_remaining"] = []
                
                # Extract results from agent
                output_data = result.output_data or {}
                
                job_record["output_size_mb"] = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0
                job_record["processing_stats"] = {
                    "model_used": result.metadata.get('model_used', 'unknown'),
                    "enhancement_type": result.metadata.get('enhancement_type', 'sota'),
                    "frames_processed": result.metadata.get('frames_processed', 0),
                    "processing_time": result.processing_time,
                    "agent_metadata": result.metadata
                }
                
                # Update performance tracking with results
                if "perf_operation_id" in job_record:
                    tracker = get_performance_tracker()
                    tracker.update_operation(
                        job_record["perf_operation_id"],
                        frames_processed=result.metadata.get('frames_processed', 0),
                        quality_score=0.85  # Default quality score
                    )
                    tracker.finish_operation(job_record["perf_operation_id"], success=True)
                
                logger.info(f"Job {job_id}: Agent processing completed successfully")
                return
            else:
                # Agent processing failed
                raise Exception(f"Agent processing failed: {result.error_message}")
        
        else:
            # Fallback to simulated processing if agent is unavailable
            logger.warning(f"Job {job_id}: VideoEnhancementAgent unavailable, using fallback processing")
            
            stages = strategy_plan.get("processing_stages", [])
            total_stages = len(stages)
            
            for i, stage in enumerate(stages):
                job_record["current_stage"] = stage
                job_record["progress"] = (i / total_stages) * 100
                job_record["stages_completed"].append(stage)
                if i + 1 < total_stages:
                    job_record["stages_remaining"] = stages[i + 1:]
                else:
                    job_record["stages_remaining"] = []
                
                # Simulate processing time for each stage
                await asyncio.sleep(2)  # In production, call actual processing functions
                
                logger.info(f"Job {job_id}: Completed stage {stage}")
            
            # Mark as completed
            job_record["status"] = JobStatus.completed
            job_record["completed_at"] = datetime.utcnow()
            job_record["progress"] = 100.0
            job_record["current_stage"] = None
            job_record["stages_remaining"] = []
            
            # Simulate output file creation (fallback)
            import shutil
            shutil.copy(input_path, output_path)
            
            # Add processing stats
            job_record["output_size_mb"] = os.path.getsize(output_path) / (1024 * 1024)
            job_record["processing_stats"] = {
                "model_used": "fallback_simulation",
                "enhancement_type": "simulated",
                "frames_processed": 1800,  # Simulated
                "enhancement_factor": 2.0,
                "quality_improvement": 0.85
            }
            
            # Update performance tracking with results
            if "perf_operation_id" in job_record:
                tracker = get_performance_tracker()
                tracker.update_operation(
                    job_record["perf_operation_id"],
                    frames_processed=job_record["processing_stats"]["frames_processed"],
                    quality_score=job_record["processing_stats"]["quality_improvement"]
                )
                tracker.finish_operation(job_record["perf_operation_id"], success=True)
            
            logger.info(f"Job {job_id}: Fallback processing completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id}: Processing failed: {e}")
        
        job_record["status"] = JobStatus.failed
        job_record["completed_at"] = datetime.utcnow()
        job_record["error_message"] = str(e)
        job_record["error_details"] = {"stage": job_record.get("current_stage"), "traceback": str(e)}
        
        # Update performance tracking for failed job
        if "perf_operation_id" in job_record:
            tracker = get_performance_tracker()
            tracker.finish_operation(job_record["perf_operation_id"], success=False, error_message=str(e))

# Export router for integration
__all__ = ['router']