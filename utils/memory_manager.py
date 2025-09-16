"""
Memory Management Utility for SOTA Video Enhancer
GPU memory optimization, model caching, and cleanup strategies for production deployment.
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


import os
import gc
import time
import logging
import threading
from typing import Dict, Optional, Any, List, Callable
from datetime import datetime, timedelta
import psutil
import weakref
from pathlib import Path

logger = logging.getLogger(__name__)

class GPUMemoryManager:
    """GPU memory management and optimization utilities."""
    
    def __init__(self):
        self._torch_available = self._check_torch_availability()
        self._gpu_available = self._check_gpu_availability()
        self._memory_pool = {}
        self._peak_memory = 0.0
        
    def _check_torch_availability(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        if not self._torch_available:
            return False
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory information."""
        if not self._gpu_available:
            return {
                'available': False,
                'total_gb': 0.0,
                'allocated_gb': 0.0,
                'cached_gb': 0.0,
                'free_gb': 0.0
            }
        
        try:
            import torch
            
            total = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            
            total_gb = total / (1024**3)
            allocated_gb = allocated / (1024**3)
            cached_gb = cached / (1024**3)
            free_gb = (total - cached) / (1024**3)
            
            # Update peak memory tracking
            if allocated_gb > self._peak_memory:
                self._peak_memory = allocated_gb
            
            return {
                'available': True,
                'total_gb': round(total_gb, 2),
                'allocated_gb': round(allocated_gb, 2),
                'cached_gb': round(cached_gb, 2),
                'free_gb': round(free_gb, 2),
                'utilization_percent': round((cached_gb / total_gb) * 100, 1),
                'peak_allocated_gb': round(self._peak_memory, 2)
            }
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {e}")
            return {'available': False, 'error': str(e)}
    
    def cleanup_gpu_memory(self, aggressive: bool = False):
        """Clean up GPU memory."""
        if not self._gpu_available:
            return
        
        try:
            import torch
            
            # Clear cache
            torch.cuda.empty_cache()
            
            if aggressive:
                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache()
                
                # Clear memory pool if available
                if hasattr(torch.cuda, 'memory_pool'):
                    torch.cuda.memory_pool.empty_cache()
                
            logger.info("GPU memory cleanup completed")
            
        except Exception as e:
            logger.error(f"GPU memory cleanup failed: {e}")
    
    def optimize_memory_usage(self, max_memory_fraction: float = 0.9):
        """Optimize GPU memory usage."""
        if not self._gpu_available:
            return
        
        try:
            import torch
            
            # Set memory fraction if supported
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(max_memory_fraction)
            
            # Enable memory pool if available
            if hasattr(torch.cuda, 'set_memory_pool_config'):
                torch.cuda.set_memory_pool_config({'maxSplit': 8})
            
            logger.info(f"GPU memory optimization applied: max_fraction={max_memory_fraction}")
            
        except Exception as e:
            logger.error(f"GPU memory optimization failed: {e}")
    
    def monitor_memory_usage(self, threshold_gb: float = 12.0) -> bool:
        """Monitor memory usage and return True if under threshold."""
        memory_info = self.get_memory_info()
        
        if not memory_info.get('available', False):
            return True  # No GPU monitoring needed
        
        allocated = memory_info['allocated_gb']
        
        if allocated > threshold_gb:
            logger.warning(f"GPU memory usage high: {allocated:.1f}GB (threshold: {threshold_gb}GB)")
            return False
        
        return True

class ModelCache:
    """Model caching system for efficient memory usage."""
    
    def __init__(self, max_cache_size: int = 3, cleanup_threshold_mb: int = 1000):
        self._cache = {}
        self._access_times = {}
        self._max_cache_size = max_cache_size
        self._cleanup_threshold_mb = cleanup_threshold_mb
        self._lock = threading.RLock()
        
    def get_model(self, model_key: str) -> Optional[Any]:
        """Get model from cache."""
        with self._lock:
            if model_key in self._cache:
                # Update access time
                self._access_times[model_key] = datetime.now()
                logger.debug(f"Model cache hit: {model_key}")
                return self._cache[model_key]
            
            logger.debug(f"Model cache miss: {model_key}")
            return None
    
    def store_model(self, model_key: str, model: Any):
        """Store model in cache with LRU eviction."""
        with self._lock:
            # Check if we need to evict
            if len(self._cache) >= self._max_cache_size and model_key not in self._cache:
                self._evict_lru()
            
            # Store model with weak reference to avoid memory leaks
            self._cache[model_key] = model
            self._access_times[model_key] = datetime.now()
            
            logger.info(f"Model cached: {model_key} (cache size: {len(self._cache)})")
    
    def _evict_lru(self):
        """Evict least recently used model."""
        if not self._access_times:
            return
        
        # Find least recently used model
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        
        # Remove from cache
        if lru_key in self._cache:
            del self._cache[lru_key]
        del self._access_times[lru_key]
        
        logger.info(f"Evicted LRU model from cache: {lru_key}")
        
        # Force cleanup
        gc.collect()
    
    def clear_cache(self):
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            gc.collect()
            logger.info("Model cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'cached_models': list(self._cache.keys()),
                'cache_size': len(self._cache),
                'max_cache_size': self._max_cache_size,
                'access_times': {k: v.isoformat() for k, v in self._access_times.items()}
            }

class SystemMemoryManager:
    """System memory monitoring and management."""
    
    def __init__(self, warning_threshold_percent: float = 85.0, critical_threshold_percent: float = 95.0):
        self.warning_threshold = warning_threshold_percent
        self.critical_threshold = critical_threshold_percent
        self._monitoring_active = False
        self._monitor_thread = None
        
    def get_memory_info(self) -> Dict[str, Any]:
        """Get system memory information."""
        memory = psutil.virtual_memory()
        
        return {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2),
            'percent_used': round(memory.percent, 1),
            'warning_threshold': self.warning_threshold,
            'critical_threshold': self.critical_threshold,
            'status': self._get_memory_status(memory.percent)
        }
    
    def _get_memory_status(self, percent_used: float) -> str:
        """Get memory status based on usage."""
        if percent_used >= self.critical_threshold:
            return 'CRITICAL'
        elif percent_used >= self.warning_threshold:
            return 'WARNING'
        else:
            return 'OK'
    
    def cleanup_system_memory(self):
        """Clean up system memory."""
        # Force Python garbage collection
        gc.collect()
        
        # Clear caches if available
        try:
            import sys
            if hasattr(sys, 'intern'):
                # Clear string intern cache (Python implementation detail)
                pass
        except:
            pass
        
        logger.info("System memory cleanup completed")
    
    def start_monitoring(self, interval_seconds: int = 30, callback: Optional[Callable] = None):
        """Start memory monitoring in background thread."""
        if self._monitoring_active:
            logger.warning("Memory monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds, callback),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Memory monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self, interval_seconds: int, callback: Optional[Callable]):
        """Memory monitoring loop."""
        while self._monitoring_active:
            try:
                memory_info = self.get_memory_info()
                status = memory_info['status']
                
                if status in ['WARNING', 'CRITICAL']:
                    logger.warning(f"Memory usage {status}: {memory_info['percent_used']}%")
                    
                    if status == 'CRITICAL':
                        logger.critical("Critical memory usage - triggering cleanup")
                        self.cleanup_system_memory()
                    
                    if callback:
                        callback(memory_info)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(interval_seconds)

class TemporaryFileManager:
    """Temporary file cleanup and management."""
    
    def __init__(self, temp_dirs: List[str] = None, max_age_hours: int = 24):
        self.temp_dirs = temp_dirs or ['data/temp', 'logs']
        self.max_age_hours = max_age_hours
    
    def cleanup_temp_files(self, force: bool = False) -> Dict[str, Any]:
        """Clean up temporary files."""
        cleanup_stats = {
            'files_removed': 0,
            'bytes_freed': 0,
            'directories_processed': 0,
            'errors': []
        }
        
        cutoff_time = datetime.now() - timedelta(hours=self.max_age_hours)
        
        for temp_dir in self.temp_dirs:
            try:
                if not os.path.exists(temp_dir):
                    continue
                
                cleanup_stats['directories_processed'] += 1
                
                for root, dirs, files in os.walk(temp_dir):
                    for filename in files:
                        filepath = os.path.join(root, filename)
                        
                        try:
                            # Check file age
                            file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                            
                            if force or file_mtime < cutoff_time:
                                file_size = os.path.getsize(filepath)
                                os.remove(filepath)
                                cleanup_stats['files_removed'] += 1
                                cleanup_stats['bytes_freed'] += file_size
                                
                        except Exception as e:
                            error_msg = f"Failed to remove {filepath}: {e}"
                            logger.warning(error_msg)
                            cleanup_stats['errors'].append(error_msg)
                
            except Exception as e:
                error_msg = f"Failed to process directory {temp_dir}: {e}"
                logger.error(error_msg)
                cleanup_stats['errors'].append(error_msg)
        
        cleanup_stats['mb_freed'] = round(cleanup_stats['bytes_freed'] / (1024**2), 2)
        
        logger.info(f"Temp file cleanup: {cleanup_stats['files_removed']} files, {cleanup_stats['mb_freed']}MB freed")
        
        return cleanup_stats

class MemoryManager:
    """Unified memory manager combining GPU, system, and cache management."""
    
    def __init__(self, config: Optional[Any] = None):
        self.gpu_manager = GPUMemoryManager()
        self.model_cache = ModelCache(
            max_cache_size=getattr(config, 'max_cached_models', 3) if config else 3
        )
        self.system_manager = SystemMemoryManager(
            warning_threshold_percent=getattr(config, 'memory_warning_threshold', 85.0) if config else 85.0
        )
        self.temp_manager = TemporaryFileManager(
            max_age_hours=getattr(config, 'temp_file_retention_hours', 24) if config else 24
        )
        
        # Start system monitoring
        self.system_manager.start_monitoring(
            interval_seconds=60,
            callback=self._memory_warning_callback
        )
        
    def _memory_warning_callback(self, memory_info: Dict[str, Any]):
        """Callback for memory warnings."""
        if memory_info['status'] == 'CRITICAL':
            logger.warning("Critical memory usage - triggering comprehensive cleanup")
            self.emergency_cleanup()
    
    def get_comprehensive_memory_info(self) -> Dict[str, Any]:
        """Get comprehensive memory information."""
        return {
            'timestamp': datetime.now().isoformat(),
            'gpu': self.gpu_manager.get_memory_info(),
            'system': self.system_manager.get_memory_info(),
            'model_cache': self.model_cache.get_cache_info()
        }
    
    def routine_cleanup(self):
        """Perform routine memory cleanup."""
        logger.info("Starting routine memory cleanup...")
        
        # Clean up temporary files
        self.temp_manager.cleanup_temp_files()
        
        # System memory cleanup
        self.system_manager.cleanup_system_memory()
        
        # GPU memory cleanup (gentle)
        self.gpu_manager.cleanup_gpu_memory(aggressive=False)
        
        logger.info("Routine cleanup completed")
    
    def emergency_cleanup(self):
        """Perform aggressive memory cleanup in emergency situations."""
        logger.warning("Starting emergency memory cleanup...")
        
        # Clear model cache
        self.model_cache.clear_cache()
        
        # Aggressive GPU cleanup
        self.gpu_manager.cleanup_gpu_memory(aggressive=True)
        
        # System cleanup
        self.system_manager.cleanup_system_memory()
        
        # Force temp file cleanup
        self.temp_manager.cleanup_temp_files(force=True)
        
        logger.warning("Emergency cleanup completed")
    
    def optimize_for_inference(self):
        """Optimize memory settings for inference."""
        logger.info("Optimizing memory for inference...")
        
        # GPU optimization
        self.gpu_manager.optimize_memory_usage(max_memory_fraction=0.9)
        
        # Pre-cleanup
        self.routine_cleanup()
        
        logger.info("Memory optimization for inference completed")
    
    def shutdown(self):
        """Shutdown memory manager and cleanup resources."""
        logger.info("Shutting down memory manager...")
        
        # Stop monitoring
        self.system_manager.stop_monitoring()
        
        # Final cleanup
        self.routine_cleanup()
        
        logger.info("Memory manager shutdown completed")


# Global memory manager instance
_memory_manager = None

def get_memory_manager(config: Optional[Any] = None) -> MemoryManager:
    """Get global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(config)
    return _memory_manager

def cleanup_memory(aggressive: bool = False):
    """Quick memory cleanup function."""
    manager = get_memory_manager()
    if aggressive:
        manager.emergency_cleanup()
    else:
        manager.routine_cleanup()

if __name__ == "__main__":
    # Test memory management
    manager = MemoryManager()
    
    print("=== Memory Info ===")
    info = manager.get_comprehensive_memory_info()
    
    import json
    print(json.dumps(info, indent=2))
    
    print("\n=== Running Cleanup ===")
    manager.routine_cleanup()
    
    print("\n=== Post-cleanup Memory Info ===")
    info_after = manager.get_comprehensive_memory_info()
    print(json.dumps(info_after, indent=2))