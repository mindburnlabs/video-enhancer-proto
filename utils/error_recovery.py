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
import time
import logging
import threading
import gc
import os
import shutil
import tempfile
from typing import (
    Callable, Dict, Any, Optional, List, Union, TypeVar, Generic,
    Awaitable, Type, Tuple
)
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import functools

from utils.error_handler import (
    VideoEnhancementError, ErrorCode, error_handler
)

logger = logging.getLogger(__name__)

T = TypeVar('T')

class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    ABORT = "abort"
    CLEANUP = "cleanup"

class FailureMode(Enum):
    """Types of failures that can be recovered from"""
    TEMPORARY = "temporary"  # Network, memory pressure, etc.
    RESOURCE = "resource"    # Out of memory, disk space, etc.
    MODEL = "model"         # Model loading, inference failures
    INPUT = "input"         # Invalid input, corruption
    SYSTEM = "system"       # System-level failures
    UNKNOWN = "unknown"     # Unclassified failures

@dataclass
class RecoveryConfig:
    """Configuration for error recovery behavior"""
    max_retries: int = 3
    retry_delays: List[float] = field(default_factory=lambda: [1.0, 2.0, 5.0])
    exponential_backoff: bool = True
    max_retry_delay: float = 60.0
    
    # Cleanup settings
    auto_cleanup: bool = True
    temp_cleanup_patterns: List[str] = field(default_factory=lambda: [
        "*.tmp", "*.temp", "temp_*", "cache_*", "*.partial"
    ])
    
    # Memory management
    force_gc_on_memory_error: bool = True
    clear_cuda_cache: bool = True
    
    # Fallback strategies
    enable_cpu_fallback: bool = True
    enable_model_fallback: bool = True
    enable_quality_degradation: bool = True

@dataclass  
class RecoveryAttempt:
    """Record of a recovery attempt"""
    timestamp: float
    strategy: RecoveryStrategy
    attempt_number: int
    error: Exception
    success: bool = False
    duration_ms: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)

class RecoveryManager:
    """Manages error recovery strategies and cleanup operations"""
    
    def __init__(self, config: RecoveryConfig = None):
        self.config = config or RecoveryConfig()
        self.active_operations: Dict[str, List[RecoveryAttempt]] = {}
        self.cleanup_registered: List[Callable[[], None]] = []
        self.temp_directories: List[Path] = []
        self.temp_files: List[Path] = []
        self._lock = threading.Lock()
    
    def register_cleanup(self, cleanup_func: Callable[[], None]):
        """Register a cleanup function to be called during error recovery"""
        self.cleanup_registered.append(cleanup_func)
    
    def register_temp_resource(self, path: Union[str, Path], is_directory: bool = False):
        """Register a temporary resource for automatic cleanup"""
        path = Path(path)
        if is_directory:
            self.temp_directories.append(path)
        else:
            self.temp_files.append(path)
    
    def classify_error(self, error: Exception) -> FailureMode:
        """Classify an error to determine recovery strategy"""
        error_msg = str(error).lower()
        
        if isinstance(error, MemoryError) or "out of memory" in error_msg:
            return FailureMode.RESOURCE
        elif isinstance(error, (ConnectionError, TimeoutError)) or "timeout" in error_msg:
            return FailureMode.TEMPORARY  
        elif isinstance(error, (FileNotFoundError, PermissionError, OSError)):
            return FailureMode.SYSTEM
        elif "model" in error_msg and ("load" in error_msg or "initialization" in error_msg):
            return FailureMode.MODEL
        elif "invalid" in error_msg or "corrupt" in error_msg:
            return FailureMode.INPUT
        else:
            return FailureMode.UNKNOWN
    
    def get_recovery_strategies(self, failure_mode: FailureMode) -> List[RecoveryStrategy]:
        """Get appropriate recovery strategies for a failure mode"""
        strategies = {
            FailureMode.TEMPORARY: [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK],
            FailureMode.RESOURCE: [RecoveryStrategy.CLEANUP, RecoveryStrategy.DEGRADE, RecoveryStrategy.FALLBACK],
            FailureMode.MODEL: [RecoveryStrategy.FALLBACK, RecoveryStrategy.RETRY],
            FailureMode.INPUT: [RecoveryStrategy.DEGRADE, RecoveryStrategy.ABORT],
            FailureMode.SYSTEM: [RecoveryStrategy.RETRY, RecoveryStrategy.ABORT],
            FailureMode.UNKNOWN: [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK]
        }
        return strategies.get(failure_mode, [RecoveryStrategy.ABORT])
    
    def calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay before retry attempt"""
        if not self.config.exponential_backoff:
            return self.config.retry_delays[min(attempt - 1, len(self.config.retry_delays) - 1)]
        
        base_delay = self.config.retry_delays[0] if self.config.retry_delays else 1.0
        delay = base_delay * (2 ** (attempt - 1))
        return min(delay, self.config.max_retry_delay)
    
    def cleanup_resources(self, operation_id: str = None):
        """Perform cleanup operations"""
        logger.info(f"🧹 Starting resource cleanup (operation: {operation_id})")
        
        try:
            # Run registered cleanup functions
            for cleanup_func in self.cleanup_registered:
                try:
                    cleanup_func()
                except Exception as e:
                    logger.warning(f"Cleanup function failed: {e}")
            
            # Clean up temporary files
            for temp_file in list(self.temp_files):
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                        logger.debug(f"Cleaned up temp file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
            
            # Clean up temporary directories
            for temp_dir in list(self.temp_directories):
                try:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                        logger.debug(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")
            
            # Clean up cache files using patterns
            self._cleanup_by_patterns()
            
            # Memory cleanup
            if self.config.force_gc_on_memory_error:
                gc.collect()
                
                # Clear CUDA cache if available
                if self.config.clear_cuda_cache:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            logger.debug("Cleared CUDA cache")
                    except ImportError:
                        pass
            
            logger.info("✅ Resource cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _cleanup_by_patterns(self):
        """Clean up files matching configured patterns"""
        temp_dir = Path(tempfile.gettempdir())
        
        for pattern in self.config.temp_cleanup_patterns:
            try:
                for file_path in temp_dir.glob(pattern):
                    if file_path.is_file():
                        file_path.unlink()
                        logger.debug(f"Cleaned up pattern file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup pattern {pattern}: {e}")

def with_recovery(
    config: RecoveryConfig = None,
    operation_id: str = None,
    cleanup_on_failure: bool = True
):
    """Decorator for automatic error recovery"""
    
    if config is None:
        config = RecoveryConfig()
    
    recovery_manager = RecoveryManager(config)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            op_id = operation_id or f"{func.__name__}_{int(time.time())}"
            
            with recovery_manager._lock:
                recovery_manager.active_operations[op_id] = []
            
            last_error = None
            
            try:
                # Try the operation with recovery strategies
                for attempt in range(config.max_retries + 1):
                    try:
                        start_time = time.perf_counter()
                        result = func(*args, **kwargs)
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        
                        # Record success
                        if attempt > 0:
                            logger.info(f"✅ Operation {op_id} succeeded on attempt {attempt + 1}")
                        
                        return result
                        
                    except Exception as error:
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        last_error = error
                        
                        # Record the attempt
                        attempt_record = RecoveryAttempt(
                            timestamp=time.time(),
                            strategy=RecoveryStrategy.RETRY,
                            attempt_number=attempt + 1,
                            error=error,
                            success=False,
                            duration_ms=duration_ms
                        )
                        
                        with recovery_manager._lock:
                            recovery_manager.active_operations[op_id].append(attempt_record)
                        
                        # Classify the error
                        failure_mode = recovery_manager.classify_error(error)
                        logger.warning(f"⚠️ Operation {op_id} failed (attempt {attempt + 1}): {error}")
                        
                        # Skip retry if we've exhausted attempts
                        if attempt >= config.max_retries:
                            break
                        
                        # Apply recovery strategies
                        strategies = recovery_manager.get_recovery_strategies(failure_mode)
                        
                        if RecoveryStrategy.CLEANUP in strategies:
                            recovery_manager.cleanup_resources(op_id)
                        
                        if RecoveryStrategy.RETRY in strategies:
                            delay = recovery_manager.calculate_retry_delay(attempt + 1)
                            logger.info(f"🔄 Retrying operation {op_id} in {delay:.1f}s...")
                            time.sleep(delay)
                        else:
                            # No retry strategy, break the loop
                            break
                
                # All attempts failed
                if cleanup_on_failure:
                    recovery_manager.cleanup_resources(op_id)
                
                # Convert to VideoEnhancementError if needed
                if not isinstance(last_error, VideoEnhancementError):
                    enhanced_error = error_handler.handle_error(
                        error=last_error,
                        component="recovery",
                        operation=op_id,
                        user_message=f"Operation failed after {config.max_retries + 1} attempts",
                        suggestions=[
                            "Try again with different settings",
                            "Check system resources (memory, disk space)",
                            "Contact support if the problem persists"
                        ]
                    )
                    raise enhanced_error
                else:
                    raise last_error
                    
            finally:
                # Cleanup operation record
                with recovery_manager._lock:
                    recovery_manager.active_operations.pop(op_id, None)
        
        return wrapper
    return decorator

class CircuitBreaker:
    """Circuit breaker pattern for failing operations"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with self._lock:
                if self.state == "open":
                    if time.time() - self.last_failure_time > self.recovery_timeout:
                        self.state = "half-open"
                        logger.info("🔄 Circuit breaker entering half-open state")
                    else:
                        raise VideoEnhancementError(
                            message="Circuit breaker is open",
                            error_code=ErrorCode.API_SERVICE_UNAVAILABLE
                        )
            
            try:
                result = func(*args, **kwargs)
                
                # Success - reset circuit breaker
                with self._lock:
                    if self.state == "half-open":
                        self.state = "closed"
                        self.failure_count = 0
                        logger.info("✅ Circuit breaker closed - service recovered")
                
                return result
                
            except Exception as e:
                with self._lock:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if self.failure_count >= self.failure_threshold:
                        self.state = "open"
                        logger.warning(f"🔥 Circuit breaker opened after {self.failure_count} failures")
                
                raise e
        
        return wrapper

def create_fallback_chain(*functions: Callable[..., T]) -> Callable[..., T]:
    """Create a fallback chain where each function is tried in sequence"""
    
    def fallback_wrapper(*args, **kwargs) -> T:
        last_error = None
        
        for i, func in enumerate(functions):
            try:
                logger.debug(f"🔄 Trying fallback option {i + 1}: {func.__name__}")
                result = func(*args, **kwargs)
                
                if i > 0:
                    logger.info(f"✅ Fallback succeeded with option {i + 1}: {func.__name__}")
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ Fallback option {i + 1} failed: {func.__name__}: {e}")
        
        # All fallbacks failed
        raise VideoEnhancementError(
            message="All fallback options exhausted",
            error_code=ErrorCode.SYSTEM_RESOURCE_EXHAUSTED,
            original_error=last_error
        )
    
    return fallback_wrapper

# Async versions of recovery decorators
def with_async_recovery(
    config: RecoveryConfig = None,
    operation_id: str = None,
    cleanup_on_failure: bool = True
):
    """Async version of recovery decorator"""
    
    if config is None:
        config = RecoveryConfig()
    
    recovery_manager = RecoveryManager(config)
    
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            op_id = operation_id or f"{func.__name__}_{int(time.time())}"
            
            with recovery_manager._lock:
                recovery_manager.active_operations[op_id] = []
            
            last_error = None
            
            try:
                for attempt in range(config.max_retries + 1):
                    try:
                        start_time = time.perf_counter()
                        result = await func(*args, **kwargs)
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        
                        if attempt > 0:
                            logger.info(f"✅ Async operation {op_id} succeeded on attempt {attempt + 1}")
                        
                        return result
                        
                    except Exception as error:
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        last_error = error
                        
                        failure_mode = recovery_manager.classify_error(error)
                        logger.warning(f"⚠️ Async operation {op_id} failed (attempt {attempt + 1}): {error}")
                        
                        if attempt >= config.max_retries:
                            break
                        
                        strategies = recovery_manager.get_recovery_strategies(failure_mode)
                        
                        if RecoveryStrategy.CLEANUP in strategies:
                            recovery_manager.cleanup_resources(op_id)
                        
                        if RecoveryStrategy.RETRY in strategies:
                            delay = recovery_manager.calculate_retry_delay(attempt + 1)
                            logger.info(f"🔄 Retrying async operation {op_id} in {delay:.1f}s...")
                            await asyncio.sleep(delay)
                        else:
                            break
                
                # All attempts failed
                if cleanup_on_failure:
                    recovery_manager.cleanup_resources(op_id)
                
                if not isinstance(last_error, VideoEnhancementError):
                    enhanced_error = error_handler.handle_error(
                        error=last_error,
                        component="async_recovery",
                        operation=op_id,
                        user_message=f"Async operation failed after {config.max_retries + 1} attempts"
                    )
                    raise enhanced_error
                else:
                    raise last_error
                    
            finally:
                with recovery_manager._lock:
                    recovery_manager.active_operations.pop(op_id, None)
        
        return wrapper
    return decorator

# Global recovery manager instance
global_recovery_manager = RecoveryManager()