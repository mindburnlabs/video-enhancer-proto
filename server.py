#!/usr/bin/env python3

"""
Simple Health and Metrics Server for Video Enhancer
Compatible with HuggingFace Spaces deployment
"""

from fastapi import FastAPI
from datetime import datetime
import psutil
import os
import sys

app = FastAPI(title="Video Enhancer Health API")

startup_time = datetime.now()
request_count = 0

@app.get("/health")
def health_check():
    """Basic health check endpoint."""
    global request_count
    request_count += 1
    
    try:
        memory = psutil.virtual_memory()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - startup_time).total_seconds(),
            "system": {
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "cpu_count": psutil.cpu_count(),
                "platform": sys.platform
            },
            "requests_served": request_count
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/ready")
def readiness_check():
    """Readiness check endpoint."""
    return {
        "ready": True,
        "timestamp": datetime.now().isoformat(),
        "service": "video-enhancer",
        "version": "1.0.0"
    }

@app.get("/metrics")
def get_metrics():
    """Simple metrics endpoint."""
    try:
        memory = psutil.virtual_memory()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - startup_time).total_seconds(),
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": memory.percent,
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
            },
            "requests": {
                "total": request_count
            }
        }
    except Exception as e:
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)