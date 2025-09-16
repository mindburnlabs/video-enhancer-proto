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
import traceback
import sys
from enum import Enum
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from datetime import datetime
import functools

logger = logging.getLogger(__name__)

class ErrorCode(Enum):
    """Standardized error codes for the video enhancement system"""
    
    # System errors (1000-1099)
    SYSTEM_UNKNOWN = "SYS_1000"
    SYSTEM_MEMORY_ERROR = "SYS_1001"
    SYSTEM_STORAGE_ERROR = "SYS_1002"
    SYSTEM_PERMISSION_ERROR = "SYS_1003"
    SYSTEM_TIMEOUT = "SYS_1004"
    SYSTEM_RESOURCE_EXHAUSTED = "SYS_1005"
    
    # Input validation errors (1100-1199)
    INPUT_INVALID_FORMAT = "INP_1100"
    INPUT_FILE_NOT_FOUND = "INP_1101"
    INPUT_FILE_TOO_LARGE = "INP_1102"
    INPUT_FILE_CORRUPTED = "INP_1103"
    INPUT_UNSUPPORTED_FORMAT = "INP_1104"
    INPUT_INVALID_RESOLUTION = "INP_1105"
    INPUT_INVALID_FRAMERATE = "INP_1106"
    INPUT_MISSING_REQUIRED = "INP_1107"
    
    # Model errors (1200-1299)
    MODEL_NOT_FOUND = "MDL_1200"
    MODEL_LOAD_ERROR = "MDL_1201"
    MODEL_INFERENCE_ERROR = "MDL_1202"
    MODEL_MEMORY_ERROR = "MDL_1203"
    MODEL_COMPATIBILITY_ERROR = "MDL_1204"
    MODEL_WEIGHTS_MISSING = "MDL_1205"
    MODEL_CUDA_ERROR = "MDL_1206"
    MODEL_TIMEOUT = "MDL_1207"
    
    # Processing errors (1300-1399)
    PROC_ENHANCEMENT_FAILED = "PROC_1300"
    PROC_DEGRADATION_ANALYSIS_FAILED = "PROC_1301"
    PROC_FACE_RESTORATION_FAILED = "PROC_1302"
    PROC_INTERPOLATION_FAILED = "PROC_1303"
    PROC_OUTPUT_GENERATION_FAILED = "PROC_1304"
    PROC_QUALITY_CHECK_FAILED = "PROC_1305"
    PROC_CANCELLED = "PROC_1306"
    
    # Agent/coordination errors (1400-1499)
    AGENT_INITIALIZATION_FAILED = "AGT_1400"
    AGENT_COMMUNICATION_FAILED = "AGT_1401"
    AGENT_ROUTING_FAILED = "AGT_1402"
    AGENT_TASK_FAILED = "AGT_1403"
    AGENT_COORDINATION_TIMEOUT = "AGT_1404"
    
    # API errors (1500-1599)
    API_INVALID_REQUEST = "API_1500"
    API_AUTHENTICATION_FAILED = "API_1501"
    API_RATE_LIMITED = "API_1502"
    API_SERVICE_UNAVAILABLE = "API_1503"
    API_INTERNAL_ERROR = "API_1504"
    
    # Security errors (1600-1699)
    SECURITY_ACCESS_DENIED = "SEC_1600"
    SECURITY_AUTHENTICATION_FAILED = "SEC_1601"
    SECURITY_RATE_LIMIT = "SEC_1602"
    SECURITY_MALICIOUS_FILE = "SEC_1603"
    SECURITY_VALIDATION_FAILED = "SEC_1604"
    SECURITY_ENCRYPTION_ERROR = "SEC_1605"

@dataclass
class ErrorContext:
    """Additional context information for errors"""
    component: str
    operation: str
    user_message: str
    technical_details: str
    suggestions: List[str]
    retry_possible: bool = False
    fallback_available: bool = False
    estimated_fix_time: Optional[str] = None

class VideoEnhancementError(Exception):
    """Base exception for video enhancement system"""
    
    def __init__(
        self, 
        message: str,
        error_code: ErrorCode,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or ErrorContext(
            component="unknown",
            operation="unknown", 
            user_message=message,
            technical_details="",
            suggestions=[]
        )
        self.original_error = original_error
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses"""
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "user_message": self.context.user_message,
            "component": self.context.component,
            "operation": self.context.operation,
            "suggestions": self.context.suggestions,
            "retry_possible": self.context.retry_possible,
            "fallback_available": self.context.fallback_available,
            "timestamp": self.timestamp.isoformat(),
            "technical_details": self.context.technical_details if logger.isEnabledFor(logging.DEBUG) else None
        }

class SystemError(VideoEnhancementError):
    """System-level errors (memory, storage, permissions)"""
    pass

class InputValidationError(VideoEnhancementError):
    """Input validation and format errors"""
    pass

class ModelError(VideoEnhancementError):
    """Model loading, inference, and compatibility errors"""
    pass

class ProcessingError(VideoEnhancementError):
    """Video processing and enhancement errors"""
    pass

class AgentError(VideoEnhancementError):
    """Agent coordination and communication errors"""
    pass

class APIError(VideoEnhancementError):
    """API and service errors"""
    pass

class ValidationError(VideoEnhancementError):
    """General validation errors"""
    pass

class SecurityError(VideoEnhancementError):
    """Security-related errors"""
    pass

class ErrorHandler:
    """Centralized error handling and reporting"""
    
    def __init__(self):
        self._error_stats: Dict[str, int] = {}
        self._recent_errors: List[VideoEnhancementError] = []
        self._max_recent_errors = 100
    
    def handle_error(
        self,
        error: Exception,
        component: str,
        operation: str,
        user_message: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        retry_possible: bool = False,
        fallback_available: bool = False
    ) -> VideoEnhancementError:
        """Convert any exception to a VideoEnhancementError with context"""
        
        # Determine error type and code
        error_code, error_class = self._classify_error(error)
        
        # Create context
        context = ErrorContext(
            component=component,
            operation=operation,
            user_message=user_message or self._generate_user_message(error, error_code),
            technical_details=self._get_technical_details(error),
            suggestions=suggestions or self._generate_suggestions(error, error_code),
            retry_possible=retry_possible,
            fallback_available=fallback_available
        )
        
        # Create appropriate error instance
        if isinstance(error, VideoEnhancementError):
            enhanced_error = error
            enhanced_error.context = context
        else:
            enhanced_error = error_class(
                message=str(error),
                error_code=error_code,
                context=context,
                original_error=error
            )
        
        # Track error statistics
        self._track_error(enhanced_error)
        
        # Log the error
        self._log_error(enhanced_error)
        
        return enhanced_error
    
    def _classify_error(self, error: Exception) -> tuple[ErrorCode, type]:
        """Classify exception into error code and class"""
        
        error_msg = str(error).lower()
        
        # Memory errors
        if isinstance(error, (MemoryError, RuntimeError)) and 'memory' in error_msg:
            return ErrorCode.SYSTEM_MEMORY_ERROR, SystemError
        elif 'cuda out of memory' in error_msg or 'out of memory' in error_msg:
            return ErrorCode.MODEL_MEMORY_ERROR, ModelError
        
        # File/IO errors
        elif isinstance(error, FileNotFoundError):
            return ErrorCode.INPUT_FILE_NOT_FOUND, InputValidationError
        elif isinstance(error, PermissionError):
            return ErrorCode.SYSTEM_PERMISSION_ERROR, SystemError
        elif isinstance(error, OSError) and 'disk' in error_msg:
            return ErrorCode.SYSTEM_STORAGE_ERROR, SystemError
        
        # Model errors
        elif 'model' in error_msg and ('load' in error_msg or 'initialization' in error_msg):
            return ErrorCode.MODEL_LOAD_ERROR, ModelError
        elif 'cuda' in error_msg or 'gpu' in error_msg:
            return ErrorCode.MODEL_CUDA_ERROR, ModelError
        elif isinstance(error, TimeoutError) and 'model' in error_msg:
            return ErrorCode.MODEL_TIMEOUT, ModelError
        
        # Processing errors
        elif 'enhancement' in error_msg or 'processing' in error_msg:
            return ErrorCode.PROC_ENHANCEMENT_FAILED, ProcessingError
        elif 'degradation' in error_msg:
            return ErrorCode.PROC_DEGRADATION_ANALYSIS_FAILED, ProcessingError
        
        # Validation errors
        elif 'invalid' in error_msg and 'format' in error_msg:
            return ErrorCode.INPUT_INVALID_FORMAT, InputValidationError
        elif 'unsupported' in error_msg:
            return ErrorCode.INPUT_UNSUPPORTED_FORMAT, InputValidationError
        
        # Default to system error
        return ErrorCode.SYSTEM_UNKNOWN, SystemError
    
    def _generate_user_message(self, error: Exception, error_code: ErrorCode) -> str:
        """Generate user-friendly error message"""
        
        user_messages = {
            ErrorCode.SYSTEM_MEMORY_ERROR: "The system ran out of memory. Please try with a smaller video or restart the application.",
            ErrorCode.MODEL_MEMORY_ERROR: "The model requires more GPU memory than available. Try reducing the video resolution or using CPU mode.",
            ErrorCode.INPUT_FILE_NOT_FOUND: "The video file could not be found. Please check the file path and try again.",
            ErrorCode.INPUT_INVALID_FORMAT: "The video format is not supported. Please use MP4, AVI, or MOV formats.",
            ErrorCode.MODEL_LOAD_ERROR: "Failed to load the enhancement model. Please check your internet connection and try again.",
            ErrorCode.PROC_ENHANCEMENT_FAILED: "Video enhancement failed. The video may be corrupted or in an unsupported format.",
            ErrorCode.API_SERVICE_UNAVAILABLE: "The service is temporarily unavailable. Please try again in a few minutes.",
        }
        
        return user_messages.get(error_code, f"An unexpected error occurred: {str(error)}")
    
    def _generate_suggestions(self, error: Exception, error_code: ErrorCode) -> List[str]:
        """Generate suggestions for resolving the error"""
        
        suggestions_map = {
            ErrorCode.SYSTEM_MEMORY_ERROR: [
                "Try processing a smaller video file",
                "Close other applications to free up memory",
                "Restart the application",
                "Use a machine with more RAM"
            ],
            ErrorCode.MODEL_MEMORY_ERROR: [
                "Reduce video resolution using the settings",
                "Switch to CPU mode if GPU memory is limited",
                "Process shorter video segments",
                "Close other GPU-intensive applications"
            ],
            ErrorCode.INPUT_FILE_NOT_FOUND: [
                "Check that the file path is correct",
                "Ensure the file hasn't been moved or deleted",
                "Try uploading the file again"
            ],
            ErrorCode.INPUT_INVALID_FORMAT: [
                "Convert the video to MP4 format",
                "Use a different video file",
                "Check if the file is corrupted"
            ],
            ErrorCode.MODEL_LOAD_ERROR: [
                "Check your internet connection",
                "Restart the application",
                "Clear the model cache and retry",
                "Contact support if the problem persists"
            ]
        }
        
        return suggestions_map.get(error_code, ["Contact support for assistance"])
    
    def _get_technical_details(self, error: Exception) -> str:
        """Get technical error details for debugging"""
        details = f"Exception type: {type(error).__name__}\n"
        details += f"Error message: {str(error)}\n"
        
        if hasattr(error, '__cause__') and error.__cause__:
            details += f"Caused by: {str(error.__cause__)}\n"
        
        # Include traceback for debugging
        details += f"Traceback:\n{''.join(traceback.format_tb(error.__traceback__))}"
        
        return details
    
    def _track_error(self, error: VideoEnhancementError):
        """Track error statistics"""
        error_key = f"{error.context.component}:{error.error_code.value}"
        self._error_stats[error_key] = self._error_stats.get(error_key, 0) + 1
        
        # Keep recent errors for analysis
        self._recent_errors.append(error)
        if len(self._recent_errors) > self._max_recent_errors:
            self._recent_errors.pop(0)
    
    def _log_error(self, error: VideoEnhancementError):
        """Log error with appropriate level"""
        log_data = {
            "error_code": error.error_code.value,
            "component": error.context.component,
            "operation": error.context.operation,
            "message": error.message,
            "retry_possible": error.context.retry_possible,
            "fallback_available": error.context.fallback_available
        }
        
        if error.error_code.value.startswith('SYS_'):
            logger.error(f"System error: {log_data}", exc_info=error.original_error)
        elif error.error_code.value.startswith('MDL_'):
            logger.warning(f"Model error: {log_data}", exc_info=error.original_error)
        else:
            logger.info(f"Processing error: {log_data}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        return {
            "total_errors": sum(self._error_stats.values()),
            "error_breakdown": dict(self._error_stats),
            "recent_errors_count": len(self._recent_errors),
            "top_errors": sorted(self._error_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def clear_stats(self):
        """Clear error statistics"""
        self._error_stats.clear()
        self._recent_errors.clear()

# Global error handler instance
error_handler = ErrorHandler()

def handle_exceptions(
    component: str,
    operation: str,
    user_message: Optional[str] = None,
    suggestions: Optional[List[str]] = None,
    retry_possible: bool = False,
    fallback_available: bool = False,
    reraise: bool = True
):
    """Decorator for automatic exception handling"""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except VideoEnhancementError:
                # Re-raise our custom errors as-is
                raise
            except Exception as e:
                # Convert other exceptions
                enhanced_error = error_handler.handle_error(
                    error=e,
                    component=component,
                    operation=operation,
                    user_message=user_message,
                    suggestions=suggestions,
                    retry_possible=retry_possible,
                    fallback_available=fallback_available
                )
                
                if reraise:
                    raise enhanced_error
                else:
                    logger.error(f"Suppressed error in {component}.{operation}: {enhanced_error}")
                    return None
        
        return wrapper
    return decorator

def create_error_response(error: VideoEnhancementError, status_code: int = 500) -> Dict[str, Any]:
    """Create standardized error response for APIs"""
    response = error.to_dict()
    response["status_code"] = status_code
    response["success"] = False
    return response