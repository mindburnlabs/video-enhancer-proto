#!/usr/bin/env python3
"""
Multi-Agent API Endpoints

FastAPI endpoints for integrating the multi-agent system with the existing API.
This allows the existing web service to leverage the multi-agent capabilities.
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


from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import asyncio
import uuid

# Multi-agent system imports
from main import VideoEnhancementSystem
from workflow import WorkflowStatus, create_video_enhancement_workflow

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/multi-agent", tags=["Multi-Agent"])

# Global system instance (initialized on startup)
_video_system: Optional[VideoEnhancementSystem] = None

class MultiAgentProcessRequest(BaseModel):
    """Request model for multi-agent video processing"""
    workflow_template: str = "video_enhancement_pipeline"
    processing_type: str = "standard"
    config: Optional[Dict[str, Any]] = None

class MultiAgentProcessResponse(BaseModel):
    """Response model for multi-agent video processing"""
    job_id: str
    status: str
    message: str
    workflow_template: str
    processing_type: str

class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status"""
    job_id: str
    status: str
    progress: float
    completed_nodes: List[str]
    failed_nodes: List[str]
    execution_time: Optional[float] = None
    success: bool
    error_messages: List[str] = []

async def get_video_system() -> VideoEnhancementSystem:
    """Get or initialize the video enhancement system"""
    global _video_system
    if _video_system is None:
        _video_system = VideoEnhancementSystem()
        await _video_system.initialize()
    return _video_system

@router.on_event("startup")
async def initialize_multi_agent_system():
    """Initialize the multi-agent system on startup"""
    try:
        await get_video_system()
        logger.info("Multi-agent system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize multi-agent system: {e}")

@router.on_event("shutdown")
async def shutdown_multi_agent_system():
    """Shutdown the multi-agent system"""
    global _video_system
    if _video_system:
        await _video_system.shutdown()
        _video_system = None

@router.post("/process", response_model=MultiAgentProcessResponse)
async def process_video_multi_agent(
    video: UploadFile = File(...),
    request: MultiAgentProcessRequest = None,
    background_tasks: BackgroundTasks = None
):
    """
    Process video using the multi-agent system
    
    This endpoint integrates with the existing API to provide
    multi-agent video enhancement capabilities.
    """
    try:
        # Get system instance
        system = await get_video_system()
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded video to temp location
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, f"input_{job_id}.mp4")
        output_path = os.path.join(temp_dir, f"output_{job_id}.mp4")
        
        # Write uploaded file
        with open(input_path, "wb") as f:
            content = await video.read()
            f.write(content)
        
        # Process video in background
        if background_tasks:
            background_tasks.add_task(
                _process_video_background,
                system, job_id, input_path, output_path,
                request.workflow_template if request else "video_enhancement_pipeline",
                request.processing_type if request else "standard"
            )
        
        return MultiAgentProcessResponse(
            job_id=job_id,
            status="processing",
            message="Video processing started with multi-agent system",
            workflow_template=request.workflow_template if request else "video_enhancement_pipeline",
            processing_type=request.processing_type if request else "standard"
        )
        
    except Exception as e:
        logger.error(f"Multi-agent video processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(job_id: str):
    """Get status of a multi-agent workflow execution"""
    try:
        system = await get_video_system()
        
        # Get workflow execution status
        execution = system.workflow_engine.executor.get_execution_status(job_id)
        
        if not execution:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Calculate progress
        total_nodes = len(execution.completed_nodes) + len(execution.failed_nodes) + len(execution.current_nodes)
        progress = len(execution.completed_nodes) / max(total_nodes, 1) * 100
        
        return WorkflowStatusResponse(
            job_id=job_id,
            status=execution.status.value,
            progress=progress,
            completed_nodes=execution.completed_nodes,
            failed_nodes=execution.failed_nodes,
            execution_time=execution.end_time - execution.start_time if execution.end_time else None,
            success=execution.status == WorkflowStatus.COMPLETED,
            error_messages=execution.error_messages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/status")
async def get_system_status():
    """Get comprehensive multi-agent system status"""
    try:
        system = await get_video_system()
        status = await system.get_system_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows")
async def list_available_workflows():
    """List available workflow templates"""
    try:
        system = await get_video_system()
        templates = system.workflow_engine.list_templates()
        
        return {
            "templates": [
                {
                    "id": template.template_id,
                    "name": template.name,
                    "description": template.description,
                    "nodes": len(template.nodes),
                    "entry_points": template.entry_points,
                    "exit_points": template.exit_points
                }
                for template in templates
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _process_video_background(
    system: VideoEnhancementSystem,
    job_id: str,
    input_path: str,
    output_path: str,
    workflow_template: str,
    processing_type: str
):
    """Background task for processing video with multi-agent system"""
    try:
        logger.info(f"Starting background processing for job {job_id}")
        
        # Process video using multi-agent system
        result = await system.process_video(
            video_path=input_path,
            output_path=output_path,
            workflow_template=workflow_template,
            processing_type=processing_type
        )
        
        logger.info(f"Multi-agent processing completed for job {job_id}: {result['success']}")
        
        # Store result for later retrieval (in production, use database)
        # For now, just log the result
        
    except Exception as e:
        logger.error(f"Background processing failed for job {job_id}: {e}")

# Export router for integration
__all__ = ['router']