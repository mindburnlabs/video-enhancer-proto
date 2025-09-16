"""
Comprehensive Logging Configuration for SOTA Video Enhancer
Structured logging with performance tracking, error handling, and production monitoring.
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
import sys
import logging
import logging.handlers
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
        }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        if hasattr(record, 'processing_time'):
            log_data['processing_time'] = record.processing_time
        if hasattr(record, 'model_name'):
            log_data['model_name'] = record.model_name
        if hasattr(record, 'video_path'):
            log_data['video_path'] = record.video_path
        if hasattr(record, 'gpu_memory'):
            log_data['gpu_memory'] = record.gpu_memory
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_data, ensure_ascii=False)

class PerformanceLogger:
    """Specialized logger for performance metrics and monitoring."""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = logging.getLogger(logger_name)
        
    def log_processing_start(self, request_id: str, video_path: str, model_name: str, user_id: str = None):
        """Log start of video processing."""
        self.logger.info(
            "Video processing started",
            extra={
                'request_id': request_id,
                'video_path': video_path,
                'model_name': model_name,
                'user_id': user_id,
                'event_type': 'processing_start'
            }
        )
    
    def log_processing_end(self, request_id: str, success: bool, processing_time: float, 
                          model_name: str, gpu_memory: float = None, quality_score: float = None):
        """Log end of video processing."""
        self.logger.info(
            f"Video processing {'completed' if success else 'failed'}",
            extra={
                'request_id': request_id,
                'success': success,
                'processing_time': processing_time,
                'model_name': model_name,
                'gpu_memory': gpu_memory,
                'quality_score': quality_score,
                'event_type': 'processing_end'
            }
        )
    
    def log_model_load(self, model_name: str, load_time: float, gpu_memory: float = None):
        """Log model loading performance."""
        self.logger.info(
            f"Model {model_name} loaded",
            extra={
                'model_name': model_name,
                'load_time': load_time,
                'gpu_memory': gpu_memory,
                'event_type': 'model_load'
            }
        )
    
    def log_gpu_memory(self, allocated: float, cached: float, total: float):
        """Log GPU memory usage."""
        self.logger.debug(
            "GPU memory status",
            extra={
                'gpu_memory_allocated': allocated,
                'gpu_memory_cached': cached,
                'gpu_memory_total': total,
                'gpu_memory_utilization': (cached / total) * 100,
                'event_type': 'gpu_memory'
            }
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log errors with context."""
        extra = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'event_type': 'error'
        }
        if context:
            extra.update(context)
            
        self.logger.error(
            f"Error occurred: {error}",
            extra=extra,
            exc_info=True
        )

class RequestLogger:
    """Logger for HTTP requests and API calls."""
    
    def __init__(self, logger_name: str = "requests"):
        self.logger = logging.getLogger(logger_name)
    
    def log_request(self, method: str, path: str, user_id: str = None, 
                   file_size: int = None, request_id: str = None):
        """Log incoming request."""
        self.logger.info(
            f"{method} {path}",
            extra={
                'method': method,
                'path': path,
                'user_id': user_id,
                'file_size': file_size,
                'request_id': request_id,
                'event_type': 'request'
            }
        )
    
    def log_response(self, status_code: int, response_time: float, 
                    request_id: str = None, error: str = None):
        """Log response."""
        self.logger.info(
            f"Response {status_code}",
            extra={
                'status_code': status_code,
                'response_time': response_time,
                'request_id': request_id,
                'error': error,
                'event_type': 'response'
            }
        )

def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_console: bool = True,
    enable_file: bool = True,
    enable_structured: bool = False,
    max_file_size_mb: int = 100,
    backup_count: int = 5
):
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_structured: Use structured JSON logging
        max_file_size_mb: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    
    # Create logs directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Setup formatters
    if enable_structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Use simpler format for console
        if not enable_structured:
            console_formatter = logging.Formatter(
                fmt='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
        else:
            console_handler.setFormatter(formatter)
        
        root_logger.addHandler(console_handler)
    
    # File handlers
    if enable_file:
        # Main application log
        app_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(log_dir, 'app.log'),
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        app_handler.setLevel(logging.DEBUG)
        app_handler.setFormatter(formatter)
        root_logger.addHandler(app_handler)
        
        # Error log (errors only)
        error_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(log_dir, 'error.log'),
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
        
        # Performance log (performance events only)
        performance_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(log_dir, 'performance.log'),
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        performance_handler.setLevel(logging.INFO)
        performance_handler.setFormatter(formatter)
        
        # Only add performance events to performance log
        performance_logger = logging.getLogger('performance')
        performance_logger.addHandler(performance_handler)
        performance_logger.propagate = False  # Don't propagate to root logger
        
        # Request log (HTTP requests only)
        request_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(log_dir, 'requests.log'),
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        request_handler.setLevel(logging.INFO)
        request_handler.setFormatter(formatter)
        
        request_logger = logging.getLogger('requests')
        request_logger.addHandler(request_handler)
        request_logger.propagate = False
    
    # Configure specific loggers
    configure_logger_levels()
    
    logging.info(f"Logging configured: level={log_level}, console={enable_console}, file={enable_file}, structured={enable_structured}")

def configure_logger_levels():
    """Configure specific logger levels to reduce noise."""
    
    # Reduce noise from common libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('diffusers').setLevel(logging.WARNING)
    logging.getLogger('gradio').setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('flask').setLevel(logging.WARNING)
    
    # Set appropriate levels for our modules
    logging.getLogger('models').setLevel(logging.INFO)
    logging.getLogger('agents').setLevel(logging.INFO)
    logging.getLogger('config').setLevel(logging.INFO)
    
    # Debug level for specific components during development
    if os.getenv('DEBUG_MODELS'):
        logging.getLogger('models').setLevel(logging.DEBUG)
    if os.getenv('DEBUG_AGENTS'):
        logging.getLogger('agents').setLevel(logging.DEBUG)

def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance with optional name."""
    return logging.getLogger(name)

def get_performance_logger() -> PerformanceLogger:
    """Get performance logger instance."""
    return PerformanceLogger()

def get_request_logger() -> RequestLogger:
    """Get request logger instance."""
    return RequestLogger()

def log_system_info():
    """Log system information at startup."""
    import platform
    import psutil
    
    logger = get_logger('system')
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python Version: {platform.python_version()}")
    logger.info(f"CPU Count: {psutil.cpu_count()}")
    logger.info(f"Memory Total: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # GPU information
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA Available: True")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f} GB)")
        else:
            logger.info("CUDA Available: False")
    except ImportError:
        logger.info("PyTorch not available")
    
    logger.info("=== End System Information ===")

def setup_production_logging(config):
    """Setup logging based on production configuration."""
    from .production_config import ProductionConfig
    
    if isinstance(config, ProductionConfig):
        setup_logging(
            log_level=config.monitoring.log_level,
            log_dir="logs",
            enable_console=True,
            enable_file=True,
            enable_structured=config.environment == 'production',
            max_file_size_mb=100,
            backup_count=5
        )
        
        # Log system info on startup
        log_system_info()
        
        return get_performance_logger(), get_request_logger()
    else:
        raise ValueError("Invalid configuration object")


# Context managers for request tracking
class RequestContext:
    """Context manager for tracking request lifecycle."""
    
    def __init__(self, request_id: str, user_id: str = None):
        self.request_id = request_id
        self.user_id = user_id
        self.start_time = None
        self.performance_logger = get_performance_logger()
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.performance_logger.log_error(
                exc_val,
                context={
                    'request_id': self.request_id,
                    'user_id': self.user_id,
                    'duration': (datetime.utcnow() - self.start_time).total_seconds()
                }
            )
        return False  # Don't suppress exceptions

if __name__ == "__main__":
    # Test logging setup
    setup_logging(enable_structured=True)
    
    logger = get_logger("test")
    perf_logger = get_performance_logger()
    
    logger.info("Test message")
    logger.warning("Test warning")
    logger.error("Test error")
    
    perf_logger.log_processing_start("test-123", "test.mp4", "vsrm", "user-456")
    perf_logger.log_processing_end("test-123", True, 15.5, "vsrm", 8.2, 0.95)