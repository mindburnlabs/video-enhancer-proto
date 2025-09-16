#!/usr/bin/env python3
"""
Background Job Scheduler

Background task scheduler for running maintenance jobs like storage cleanup,
system health checks, and other periodic tasks.
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
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class BackgroundScheduler:
    """Background task scheduler for maintenance and monitoring"""
    
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.running = False
        self.task_results: Dict[str, Dict] = {}
        
    def add_task(self, 
                name: str, 
                func: Callable, 
                interval_hours: float = 24.0, 
                run_at_startup: bool = False,
                description: str = ""):
        """Add a scheduled background task"""
        
        self.tasks[name] = {
            "function": func,
            "interval_hours": interval_hours,
            "interval_seconds": interval_hours * 3600,
            "run_at_startup": run_at_startup,
            "description": description,
            "next_run": datetime.utcnow() if run_at_startup else datetime.utcnow() + timedelta(hours=interval_hours),
            "last_run": None,
            "run_count": 0,
            "error_count": 0,
            "last_error": None
        }
        
        logger.info(f"Scheduled task '{name}': {description} (every {interval_hours}h)")
        
    async def run_task(self, task_name: str) -> Dict:
        """Run a specific task and return results"""
        
        if task_name not in self.tasks:
            raise ValueError(f"Task '{task_name}' not found")
            
        task = self.tasks[task_name]
        
        logger.info(f"Running background task: {task_name}")
        
        start_time = datetime.utcnow()
        result = {
            "task_name": task_name,
            "started_at": start_time.isoformat(),
            "success": False,
            "duration_seconds": 0,
            "error": None,
            "result": None
        }
        
        try:
            # Run the task function
            task_result = await task["function"]()
            
            # Update task metadata
            task["last_run"] = start_time
            task["run_count"] += 1
            task["next_run"] = start_time + timedelta(hours=task["interval_hours"])
            
            # Store results
            result["success"] = True
            result["result"] = task_result
            result["duration_seconds"] = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"Task '{task_name}' completed successfully "
                       f"in {result['duration_seconds']:.2f} seconds")
            
        except Exception as e:
            task["error_count"] += 1
            task["last_error"] = str(e)
            
            result["error"] = str(e)
            result["duration_seconds"] = (datetime.utcnow() - start_time).total_seconds()
            
            logger.error(f"Task '{task_name}' failed after "
                        f"{result['duration_seconds']:.2f} seconds: {e}")
        
        result["completed_at"] = datetime.utcnow().isoformat()
        self.task_results[task_name] = result
        
        return result
    
    async def start(self):
        """Start the background scheduler"""
        
        self.running = True
        logger.info(f"Background scheduler started with {len(self.tasks)} tasks")
        
        # Run startup tasks
        for task_name, task in self.tasks.items():
            if task["run_at_startup"]:
                try:
                    await self.run_task(task_name)
                except Exception as e:
                    logger.error(f"Startup task '{task_name}' failed: {e}")
        
        # Main scheduling loop
        while self.running:
            try:
                now = datetime.utcnow()
                
                # Check which tasks need to run
                for task_name, task in self.tasks.items():
                    if now >= task["next_run"]:
                        try:
                            await self.run_task(task_name)
                        except Exception as e:
                            logger.error(f"Scheduled task '{task_name}' failed: {e}")
                
                # Sleep for 1 minute before checking again
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Scheduler main loop error: {e}")
                await asyncio.sleep(60)  # Continue despite errors
    
    async def stop(self):
        """Stop the background scheduler"""
        self.running = False
        logger.info("Background scheduler stopped")
    
    def get_status(self) -> Dict:
        """Get scheduler status and task information"""
        
        now = datetime.utcnow()
        
        status = {
            "running": self.running,
            "task_count": len(self.tasks),
            "total_runs": sum(task["run_count"] for task in self.tasks.values()),
            "total_errors": sum(task["error_count"] for task in self.tasks.values()),
            "tasks": {}
        }
        
        for task_name, task in self.tasks.items():
            task_status = {
                "description": task["description"],
                "interval_hours": task["interval_hours"],
                "run_count": task["run_count"],
                "error_count": task["error_count"],
                "last_run": task["last_run"].isoformat() if task["last_run"] else None,
                "next_run": task["next_run"].isoformat(),
                "minutes_until_next_run": int((task["next_run"] - now).total_seconds() / 60),
                "last_error": task["last_error"]
            }
            
            # Add last result if available
            if task_name in self.task_results:
                task_status["last_result"] = self.task_results[task_name]
            
            status["tasks"][task_name] = task_status
        
        return status

# Global scheduler instance
_scheduler: Optional[BackgroundScheduler] = None

def get_scheduler() -> BackgroundScheduler:
    """Get or create the global scheduler instance"""
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler()
    return _scheduler

async def setup_default_tasks():
    """Setup default background tasks for the application"""
    
    scheduler = get_scheduler()
    
    # Storage maintenance task
    async def storage_maintenance():
        try:
            from utils.storage_manager import StorageManager
            manager = StorageManager()
            result = await manager.run_maintenance()
            return result
        except Exception as e:
            logger.error(f"Storage maintenance failed: {e}")
            raise
    
    scheduler.add_task(
        name="storage_maintenance",
        func=storage_maintenance,
        interval_hours=24.0,  # Run daily
        run_at_startup=False,
        description="Clean up expired files and verify storage integrity"
    )
    
    # System health check
    async def system_health_check():
        try:
            import psutil
            from config.model_config import ModelConfig
            
            # System metrics
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Model config check
            config = ModelConfig()
            model_status = config.get_model_status()
            
            health_report = {
                "timestamp": datetime.utcnow().isoformat(),
                "system": {
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                    "cpu_percent": cpu_percent,
                    "disk_free_gb": disk.free / (1024**3)
                },
                "models": {
                    "device": model_status["device"],
                    "available_models": [k for k, v in model_status.items() 
                                       if isinstance(v, bool) and v]
                },
                "health_status": "healthy" if (
                    memory.percent < 90 and 
                    disk.percent < 95 and 
                    cpu_percent < 95
                ) else "degraded"
            }
            
            if health_report["health_status"] == "degraded":
                logger.warning(f"System health degraded: {health_report['system']}")
            
            return health_report
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    scheduler.add_task(
        name="system_health_check",
        func=system_health_check,
        interval_hours=6.0,  # Run every 6 hours
        run_at_startup=True,
        description="Monitor system resources and health"
    )
    
    # Temporary file cleanup
    async def temp_cleanup():
        try:
            import tempfile
            import shutil
            
            # Clean up old temp directories
            temp_base = Path(tempfile.gettempdir())
            cleaned = 0
            bytes_freed = 0
            cutoff_time = datetime.utcnow() - timedelta(hours=48)  # 48 hours old
            
            for temp_dir in temp_base.glob("tmp*"):
                if temp_dir.is_dir():
                    try:
                        # Check creation time
                        created = datetime.fromtimestamp(temp_dir.stat().st_ctime)
                        if created < cutoff_time:
                            # Get size before deletion
                            size = sum(f.stat().st_size for f in temp_dir.rglob('*') if f.is_file())
                            shutil.rmtree(temp_dir)
                            cleaned += 1
                            bytes_freed += size
                    except Exception as e:
                        logger.debug(f"Could not clean temp dir {temp_dir}: {e}")
            
            result = {
                "directories_cleaned": cleaned,
                "bytes_freed": bytes_freed,
                "mb_freed": bytes_freed / (1024 * 1024)
            }
            
            if cleaned > 0:
                logger.info(f"Cleaned {cleaned} temp directories, "
                           f"freed {result['mb_freed']:.2f} MB")
            
            return result
            
        except Exception as e:
            logger.error(f"Temp cleanup failed: {e}")
            raise
    
    scheduler.add_task(
        name="temp_cleanup",
        func=temp_cleanup,
        interval_hours=12.0,  # Run twice daily
        run_at_startup=False,
        description="Clean up old temporary files and directories"
    )
    
    # Job cleanup (clean up old job records from API)
    async def job_cleanup():
        try:
            from api.v1.process_endpoints import _job_store
            from datetime import timedelta
            
            now = datetime.utcnow()
            cutoff_time = now - timedelta(days=7)  # Keep jobs for 7 days
            
            jobs_before = len(_job_store)
            jobs_to_remove = []
            
            for job_id, job_record in _job_store.items():
                created_at = job_record.get("created_at")
                if created_at and created_at < cutoff_time:
                    # Only clean up completed or failed jobs
                    if job_record.get("status") in ["completed", "failed", "cancelled"]:
                        jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del _job_store[job_id]
            
            jobs_cleaned = len(jobs_to_remove)
            
            result = {
                "jobs_before": jobs_before,
                "jobs_cleaned": jobs_cleaned,
                "jobs_remaining": len(_job_store)
            }
            
            if jobs_cleaned > 0:
                logger.info(f"Cleaned up {jobs_cleaned} old job records")
            
            return result
            
        except Exception as e:
            logger.error(f"Job cleanup failed: {e}")
            raise
    
    scheduler.add_task(
        name="job_cleanup",
        func=job_cleanup,
        interval_hours=24.0,  # Run daily
        run_at_startup=False,
        description="Clean up old job records from API"
    )
    
    logger.info("Default background tasks configured")

async def start_scheduler():
    """Start the background scheduler with default tasks"""
    await setup_default_tasks()
    scheduler = get_scheduler()
    await scheduler.start()

def get_scheduler_status() -> Dict:
    """Get current scheduler status"""
    scheduler = get_scheduler()
    return scheduler.get_status()

# Export main functions
__all__ = ['get_scheduler', 'setup_default_tasks', 'start_scheduler', 'get_scheduler_status']