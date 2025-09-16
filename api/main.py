#!/usr/bin/env python3
"""
FastAPI Application for Video Enhancement API

Main application that integrates all API endpoints and provides
OpenAPI documentation and testing interface.
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

from fastapi import FastAPI, HTTPException, Request, Query, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import time
from pathlib import Path
import tempfile
import os

# Import centralized error handling
from utils.error_handler import (
    error_handler, VideoEnhancementError, APIError, InputValidationError,
    ErrorCode, create_error_response, handle_exceptions
)

# Import authentication system  
from utils.auth import (
    auth_manager, get_current_user, require_admin, get_client_ip
)

# Import API routers
from api.v1.process_endpoints import router as process_router
try:
    from api.v1.multi_agent_endpoints import router as multi_agent_router
    multi_agent_available = True
except ImportError:
    multi_agent_available = False
    logging.warning("Multi-agent endpoints not available")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app with comprehensive metadata
app = FastAPI(
    title="Video Enhancement API",
    description="""
    ## Next-Generation AI Video Enhancement API

    Comprehensive REST API for programmatic video enhancement using state-of-the-art models.

    ### Features
    * **Automatic Strategy Selection**: AI-powered analysis selects optimal enhancement strategy
    * **Multiple SOTA Models**: VSRM, SeedVR2, DiTVR, FastMamba VSR support
    * **Job Management**: Asynchronous processing with detailed status tracking
    * **Flexible Configuration**: Extensive options for quality, latency, and processing preferences
    * **Quality Metrics**: Optional computation of enhancement quality assessments

    ### Processing Strategies
    * **VSRM**: Video Super-Resolution Mamba - optimized for high-motion content
    * **SeedVR2**: Diffusion-based restoration for compression artifacts
    * **DiTVR**: Zero-shot transformer for unknown degradations
    * **FastMamba VSR**: Lightweight model for real-time processing

    ### Usage Examples
    
    **Basic Enhancement:**
    ```bash
    curl -X POST "http://localhost:8000/api/v1/process/auto" \\
         -H "accept: application/json" \\
         -F "file=@video.mp4"
    ```

    **Advanced Configuration:**
    ```bash
    curl -X POST "http://localhost:8000/api/v1/process/auto" \\
         -H "accept: application/json" \\
         -F "file=@video.mp4" \\
         -F 'request={"vsr_strategy":"seedvr2","latency_class":"flexible","enable_face_expert":true}'
    ```

    **Check Job Status:**
    ```bash
    curl -X GET "http://localhost:8000/api/v1/process/job/{job_id}"
    ```
    """,
    version="1.0.0",
    contact={
        "name": "Video Enhancement API Support",
        "email": "support@videoenhancer.ai",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    terms_of_service="https://videoenhancer.ai/terms",
    openapi_tags=[
        {
            "name": "Video Processing",
            "description": "Core video enhancement endpoints with job management"
        },
        {
            "name": "Multi-Agent",
            "description": "Advanced multi-agent processing workflows"
        },
        {
            "name": "System",
            "description": "System status and health monitoring"
        }
    ]
)

# Middleware configuration - Secure CORS settings
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Restrict origins in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Explicit methods only
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],  # Explicit headers only
    expose_headers=["X-Process-Time", "X-Rate-Limit-Remaining"]
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing information"""
    start_time = time.time()
    
    # Log request
    logger.info(f"{request.method} {request.url}")
    
    try:
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.3f}s")
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"{request.method} {request.url} - ERROR: {str(e)} - {process_time:.3f}s")
        raise

# Enhanced exception handlers with centralized error handling
@app.exception_handler(VideoEnhancementError)
async def video_enhancement_exception_handler(request: Request, exc: VideoEnhancementError):
    """Handle custom video enhancement errors"""
    status_code = 400 if isinstance(exc, InputValidationError) else 500
    
    # Map specific error codes to HTTP status codes
    if isinstance(exc, APIError):
        if exc.error_code == ErrorCode.API_RATE_LIMITED:
            status_code = 429
        elif exc.error_code == ErrorCode.API_AUTHENTICATION_FAILED:
            status_code = 401
        elif exc.error_code == ErrorCode.API_SERVICE_UNAVAILABLE:
            status_code = 503
        elif exc.error_code == ErrorCode.API_INVALID_REQUEST:
            status_code = 400
    
    return JSONResponse(
        status_code=status_code,
        content=create_error_response(exc, status_code)
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle FastAPI request validation errors"""
    # Convert validation errors to user-friendly messages
    error_details = []
    for error in exc.errors():
        field = ' -> '.join(str(x) for x in error['loc'][1:])  # Skip 'body'
        error_details.append(f"{field}: {error['msg']}")
    
    enhanced_error = error_handler.handle_error(
        error=exc,
        component="api",
        operation="request_validation",
        user_message=f"Invalid request parameters: {'; '.join(error_details)}",
        suggestions=[
            "Check the request format and required fields",
            "Ensure all parameters are of the correct type",
            "Review the API documentation at /docs for correct usage",
            "Validate file uploads are in supported formats (MP4, AVI, MOV)"
        ]
    )
    
    return JSONResponse(
        status_code=422,
        content=create_error_response(enhanced_error, 422)
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with enhanced context"""
    error_code = ErrorCode.API_INVALID_REQUEST
    suggestions = ["Contact support for assistance"]
    
    if exc.status_code == 404:
        error_code = ErrorCode.API_INVALID_REQUEST
        suggestions = [
            "Check the URL path is correct",
            "Review the API documentation at /docs",
            "Ensure the endpoint exists"
        ]
    elif exc.status_code == 401:
        error_code = ErrorCode.API_AUTHENTICATION_FAILED
        suggestions = [
            "Check your API key or authentication credentials",
            "Ensure you have permission to access this endpoint"
        ]
    elif exc.status_code == 503:
        error_code = ErrorCode.API_SERVICE_UNAVAILABLE
        suggestions = [
            "The service is temporarily unavailable",
            "Please try again in a few minutes",
            "Check the system status page"
        ]
    
    enhanced_error = APIError(
        message=str(exc.detail),
        error_code=error_code
    )
    enhanced_error.context.suggestions = suggestions
    
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(enhanced_error, exc.status_code)
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with enhanced error reporting"""
    logger.error(f"Unhandled exception on {request.method} {request.url}: {str(exc)}")
    
    enhanced_error = error_handler.handle_error(
        error=exc,
        component="api",
        operation="request_processing",
        user_message="An unexpected error occurred while processing your request",
        suggestions=[
            "Please try again in a few moments",
            "Contact support if the problem persists",
            "Check the API documentation at /docs for correct usage",
            "Verify your request format and parameters"
        ]
    )
    
    return JSONResponse(
        status_code=500,
        content=create_error_response(enhanced_error, 500)
    )

# Include API routers
app.include_router(process_router)

if multi_agent_available:
    app.include_router(multi_agent_router)

# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """API root endpoint with basic information"""
    return {
        "name": "Video Enhancement API",
        "version": "1.0.0",
        "description": "Next-generation AI video enhancement API",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
        "endpoints": {
            "process": "/api/v1/process/auto",
            "job_status": "/api/v1/process/job/{job_id}",
            "strategies": "/api/v1/process/strategies",
            "agent_status": "/api/v1/process/agent/status",
            "health": "/api/v1/process/health",
            "performance": "/performance/stats",
            "storage": "/storage/stats",
            "metrics": "/metrics"
        },
        "multi_agent_available": multi_agent_available
    }

@app.get("/health", tags=["System"])
async def health_check():
    """Simple health check endpoint"""
    try:
        from config.model_config import ModelConfig
        config = ModelConfig()
        status = config.get_model_status()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "api_version": "1.0.0",
            "device": status["device"],
            "available_models": [k for k, v in status.items() if isinstance(v, bool) and v]
        }
    except Exception as e:
        return {
            "status": "degraded",
            "timestamp": time.time(),
            "error": str(e)
        }

@app.get("/metrics", tags=["System"])
async def get_metrics():
    """Basic system metrics endpoint"""
    try:
        import psutil
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Import job store to get job metrics
        from api.v1.process_endpoints import _job_store
        
        job_counts = {
            "total": len(_job_store),
            "pending": len([j for j in _job_store.values() if j["status"].value == "pending"]),
            "processing": len([j for j in _job_store.values() if j["status"].value == "processing"]),
            "completed": len([j for j in _job_store.values() if j["status"].value == "completed"]),
            "failed": len([j for j in _job_store.values() if j["status"].value == "failed"])
        }
        
        # Try to get performance metrics
        performance_stats = None
        try:
            from utils.performance_monitor import get_performance_stats
            performance_stats = get_performance_stats()
        except Exception as perf_err:
            logger.debug(f"Performance stats unavailable: {perf_err}")
        
        result = {
            "timestamp": time.time(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            },
            "jobs": job_counts
        }
        
        if performance_stats:
            result["performance"] = {
                "summary": performance_stats.get("summary", {}),
                "recent_operations": len(performance_stats.get("recent_activity", {}).get("last_10_operations", []))
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return {
            "timestamp": time.time(),
            "status": "error",
            "error": str(e)
        }

@app.get("/performance/stats", tags=["System"])
async def get_performance_stats_endpoint():
    """Get comprehensive performance statistics"""
    try:
        from utils.performance_monitor import get_performance_stats
        stats = get_performance_stats()
        
        return {
            "status": "success",
            "timestamp": time.time(),
            "data": stats
        }
    except Exception as e:
        logger.error(f"Performance stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance stats: {str(e)}")

@app.post("/performance/export", tags=["System"])
async def export_performance_data(
    background_tasks: BackgroundTasks,
    format: str = Query("json", regex="^(json|csv)$")
):
    """Export performance metrics to file"""
    try:
        from utils.performance_monitor import export_performance_metrics
        from datetime import datetime
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{format}', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        # Export metrics
        export_performance_metrics(temp_path, format)
        
        # Schedule cleanup
        background_tasks.add_task(os.unlink, temp_path)
        
        filename = f"performance_metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{format}"
        
        return FileResponse(
            path=temp_path,
            filename=filename,
            media_type="application/json" if format == "json" else "text/csv"
        )
        
    except Exception as e:
        logger.error(f"Performance export error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export performance data: {str(e)}")

@app.get("/storage/stats", tags=["System"])
async def get_storage_stats():
    """Get current storage usage and retention policy statistics"""
    try:
        from utils.storage_retention import get_storage_stats
        stats = get_storage_stats()
        
        return {
            "status": "success",
            "timestamp": time.time(),
            "data": stats
        }
    except Exception as e:
        logger.error(f"Storage stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get storage stats: {str(e)}")

@app.post("/storage/cleanup", tags=["System"])
async def trigger_storage_cleanup():
    """Manually trigger storage cleanup with current retention policies"""
    try:
        from utils.storage_retention import cleanup_storage
        results = cleanup_storage()
        
        total_files = sum(r.files_removed for r in results)
        total_mb_freed = sum(r.bytes_freed for r in results) / (1024 * 1024)
        
        return {
            "status": "success",
            "timestamp": time.time(),
            "summary": {
                "files_removed": total_files,
                "mb_freed": round(total_mb_freed, 2),
                "rules_executed": len(results)
            },
            "detailed_results": [
                {
                    "rule_name": r.rule_name,
                    "files_removed": r.files_removed,
                    "mb_freed": round(r.bytes_freed / (1024 * 1024), 2),
                    "execution_time": round(r.execution_time, 2),
                    "had_errors": len(r.errors) > 0
                }
                for r in results
            ]
        }
    except Exception as e:
        logger.error(f"Storage cleanup error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to perform storage cleanup: {str(e)}")

@app.post("/storage/emergency-cleanup", tags=["System"])
async def emergency_storage_cleanup(
    target_mb: int = Query(1024, ge=100, le=10240, description="Target MB to free")
):
    """Perform emergency storage cleanup to free specified disk space"""
    try:
        from utils.storage_retention import emergency_cleanup
        result = emergency_cleanup(target_mb)
        
        return {
            "status": "success" if result["success"] else "partial",
            "timestamp": time.time(),
            "data": result
        }
    except Exception as e:
        logger.error(f"Emergency cleanup error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to perform emergency cleanup: {str(e)}")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("🚀 Video Enhancement API starting up...")
    
    try:
        # Validate configuration
        from config.model_config import ModelConfig
        config = ModelConfig()
        status = config.get_model_status()
        
        logger.info(f"   Device: {status['device']}")
        logger.info(f"   Available models: {[k for k, v in status.items() if isinstance(v, bool) and v]}")
        
        # Create necessary directories
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        (data_dir / "test_output").mkdir(exist_ok=True)
        (data_dir / "uploads").mkdir(exist_ok=True)
        (data_dir / "temp").mkdir(exist_ok=True)
        (data_dir / "api_output").mkdir(exist_ok=True)
        
        # Initialize storage retention manager
        try:
            from utils.storage_retention import get_retention_manager, start_cleanup_scheduler
            retention_manager = get_retention_manager()
            
            # Start automatic cleanup scheduler (every 6 hours)
            start_cleanup_scheduler(6)
            
            logger.info("   Storage retention manager initialized")
            logger.info("   Automatic cleanup scheduler started (6h interval)")
            
        except Exception as e:
            logger.warning(f"Storage retention initialization failed: {e}")
        
        logger.info("✅ Video Enhancement API ready!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("🛑 Video Enhancement API shutting down...")
    
    # Cleanup temporary files
    import tempfile
    import shutil
    
    try:
        # In production, you'd want to gracefully handle running jobs
        from api.v1.process_endpoints import _job_store
        
        running_jobs = [j for j in _job_store.values() 
                       if j["status"].value in ["pending", "processing"]]
        
        if running_jobs:
            logger.info(f"⚠️  Shutting down with {len(running_jobs)} active jobs")
        
        logger.info("✅ Shutdown complete")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

if __name__ == "__main__":
    import uvicorn
    
    # Development server configuration
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )