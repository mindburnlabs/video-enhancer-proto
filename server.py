#!/usr/bin/env python3

"""
Simple Health and Metrics Server for Video Enhancer
Compatible with HuggingFace Spaces deployment
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