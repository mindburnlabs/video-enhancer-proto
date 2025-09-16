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

import logging
import sys
import os
import json
import traceback
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager
import functools

# Try to import error handling (may not be available during bootstrap)
try:
    from utils.error_handler import VideoEnhancementError, ErrorCode
except ImportError:
    VideoEnhancementError = None
    ErrorCode = None

class EnhancedDebugFormatter(logging.Formatter):
    """Enhanced formatter with color coding and detailed context"""
    
    # Color codes for terminal output
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green  
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, use_colors: bool = None):
        super().__init__()
        # Auto-detect color support
        self.use_colors = (
            use_colors if use_colors is not None 
            else (hasattr(sys.stderr, 'isatty') and sys.stderr.isatty())
        )
        
    def format(self, record: logging.LogRecord) -> str:
        # Create timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Get color for level
        color = self.COLORS.get(record.levelname, '') if self.use_colors else ''
        reset = self.COLORS['RESET'] if self.use_colors else ''
        
        # Build base message
        base_msg = f"{timestamp} {color}{record.levelname:8}{reset} [{record.name}] {record.getMessage()}"
        
        # Add context information
        context_parts = []
        
        # Add location info for debugging
        if record.levelno <= logging.DEBUG:
            location = f"{Path(record.pathname).name}:{record.lineno} in {record.funcName}()"
            context_parts.append(f"📍 {location}")
        
        # Add component and operation if available
        if hasattr(record, 'component'):
            context_parts.append(f"🔧 {record.component}")
        if hasattr(record, 'operation'):
            context_parts.append(f"⚙️ {record.operation}")
        
        # Add performance info if available
        if hasattr(record, 'duration_ms'):
            context_parts.append(f"⏱️ {record.duration_ms:.1f}ms")
        if hasattr(record, 'memory_mb'):
            context_parts.append(f"🧠 {record.memory_mb:.1f}MB")
        
        # Add job/request ID if available
        if hasattr(record, 'job_id'):
            context_parts.append(f"📋 job:{record.job_id[:8]}")
        if hasattr(record, 'request_id'):
            context_parts.append(f"🌐 req:{record.request_id[:8]}")
        
        # Add error code if available
        if hasattr(record, 'error_code'):
            context_parts.append(f"❌ {record.error_code}")
        
        # Format final message
        if context_parts:
            context_str = " | ".join(context_parts)
            final_msg = f"{base_msg}\n    {context_str}"
        else:
            final_msg = base_msg
        
        # Add exception info if present
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            final_msg += f"\n{exc_text}"
        
        return final_msg

class ContextualDebugLogger:
    """Logger with rich context and debugging features"""
    
    def __init__(self, name: str, component: str = None):
        self.logger = logging.getLogger(name)
        self.component = component or name
        self.context: Dict[str, Any] = {}
        self._call_stack: List[Dict[str, Any]] = []
        
    def set_context(self, **kwargs):
        """Set context that will be included in all log messages"""
        self.context.update(kwargs)
        
    def clear_context(self):
        """Clear all context"""
        self.context.clear()
        
    def add_call_context(self, function_name: str, **kwargs):
        """Add function call context (for debugging call chains)"""
        call_info = {
            'function': function_name,
            'timestamp': time.time(),
            'thread': threading.current_thread().name,
            **kwargs
        }
        self._call_stack.append(call_info)
        
        # Keep stack size reasonable
        if len(self._call_stack) > 50:
            self._call_stack.pop(0)
    
    def _create_extra(self, **kwargs) -> Dict[str, Any]:
        """Create extra dictionary for logging"""
        extra = {
            'component': self.component,
            **self.context,
            **kwargs
        }
        return extra
    
    def debug(self, message: str, **kwargs):
        """Enhanced debug logging"""
        extra = self._create_extra(**kwargs)
        self.logger.debug(message, extra=extra)
        
    def info(self, message: str, **kwargs):
        """Enhanced info logging"""  
        extra = self._create_extra(**kwargs)
        self.logger.info(message, extra=extra)
        
    def warning(self, message: str, **kwargs):
        """Enhanced warning logging"""
        extra = self._create_extra(**kwargs)
        self.logger.warning(message, extra=extra)
        
    def error(self, message: str, error: Exception = None, **kwargs):
        """Enhanced error logging with exception handling"""
        extra = self._create_extra(**kwargs)
        
        if error:
            extra.update({
                'error_type': type(error).__name__,
                'error_message': str(error)
            })
            
            # Add VideoEnhancementError specific context
            if VideoEnhancementError and isinstance(error, VideoEnhancementError):
                extra.update({
                    'error_code': error.error_code.value,
                    'user_message': error.context.user_message,
                    'retry_possible': error.context.retry_possible,
                    'fallback_available': error.context.fallback_available,
                    'suggestions': error.context.suggestions
                })
        
        self.logger.error(message, extra=extra, exc_info=error)
        
    def critical(self, message: str, error: Exception = None, **kwargs):
        """Enhanced critical logging"""
        extra = self._create_extra(**kwargs)
        
        if error:
            extra.update({
                'error_type': type(error).__name__,
                'error_message': str(error)
            })
        
        self.logger.critical(message, extra=extra, exc_info=error)
    
    def trace_call(self, message: str, **kwargs):
        """Log function call trace for debugging"""
        extra = self._create_extra(call_trace=True, **kwargs)
        self.debug(f"🔍 TRACE: {message}", **extra)
    
    def performance(self, operation: str, duration_ms: float, **kwargs):
        """Log performance information"""
        extra = self._create_extra(
            operation=operation,
            duration_ms=duration_ms,
            **kwargs
        )
        
        if duration_ms > 5000:  # Log slow operations as warnings
            self.warning(f"⚠️ Slow operation: {operation} took {duration_ms:.1f}ms", **extra)
        elif duration_ms > 1000:
            self.info(f"⏱️ {operation} completed in {duration_ms:.1f}ms", **extra)
        else:
            self.debug(f"✅ {operation} completed in {duration_ms:.1f}ms", **extra)
    
    def get_call_stack_summary(self) -> List[Dict[str, Any]]:
        """Get summary of recent function calls for debugging"""
        return list(self._call_stack)

class TimedOperation:
    """Context manager for timing operations and automatic logging"""
    
    def __init__(self, logger: ContextualDebugLogger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.trace_call(f"Starting {self.operation}", **self.context)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        
        if exc_type is not None:
            self.logger.error(
                f"❌ {self.operation} failed after {duration_ms:.1f}ms",
                error=exc_val,
                operation=self.operation,
                duration_ms=duration_ms,
                **self.context
            )
        else:
            self.logger.performance(
                self.operation,
                duration_ms,
                **self.context
            )

def log_function_calls(logger: ContextualDebugLogger, include_args: bool = False):
    """Decorator to automatically log function calls"""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            # Create context
            context = {'function': func_name}
            if include_args:
                context.update({
                    'args': str(args)[:200],  # Limit length
                    'kwargs': {k: str(v)[:100] for k, v in kwargs.items()}
                })
            
            with TimedOperation(logger, f"call {func_name}", **context):
                return func(*args, **kwargs)
                
        return wrapper
    return decorator

def setup_debug_logging(
    log_level: str = "DEBUG",
    enable_colors: bool = None,
    log_file: Optional[str] = None
) -> ContextualDebugLogger:
    """Setup enhanced debug logging"""
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create console handler with enhanced formatting
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = EnhancedDebugFormatter(use_colors=enable_colors)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # Always debug level for files
        file_formatter = EnhancedDebugFormatter(use_colors=False)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Create and return main debug logger
    main_logger = ContextualDebugLogger("debug", "system")
    main_logger.info("🚀 Enhanced debug logging initialized", log_level=log_level)
    
    return main_logger

# Convenience function for getting component-specific debug loggers
def get_debug_logger(component: str) -> ContextualDebugLogger:
    """Get a component-specific debug logger"""
    return ContextualDebugLogger(f"debug.{component}", component)

# Global debug logger instance (will be None until setup_debug_logging is called)
debug_logger: Optional[ContextualDebugLogger] = None