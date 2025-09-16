"""
Security Integration Layer

This module provides a unified interface for all security measures in the video enhancement application,
integrating file security, data protection, authentication, and monitoring capabilities.

MIT License - Copyright (c) 2024 Video Enhancement Project
"""

import os
import logging
import secrets
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
import ipaddress
import re

from utils.file_security import FileSecurityManager, SecurityThreat
from utils.data_protection import (
    DataProtectionManager, DataCategory, data_protection_manager
)
from utils.error_handler import (
    error_handler, ValidationError, SecurityError, ErrorCode
)

logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    # API Security
    api_key_required: bool = True
    rate_limit_enabled: bool = True
    max_requests_per_minute: int = 60
    max_file_size_mb: int = 500
    
    # File Security
    file_validation_enabled: bool = True
    virus_scanning_enabled: bool = True
    content_analysis_enabled: bool = True
    allowed_extensions: List[str] = field(default_factory=lambda: ['.mp4', '.avi', '.mov', '.mkv'])
    
    # Data Protection
    encryption_enabled: bool = True
    data_retention_hours: int = 24
    anonymization_enabled: bool = True
    audit_logging_enabled: bool = True
    
    # Network Security
    allowed_ips: List[str] = field(default_factory=list)
    blocked_ips: List[str] = field(default_factory=list)
    require_https: bool = True
    
    # Session Security
    session_timeout_minutes: int = 60
    max_concurrent_sessions: int = 5

@dataclass
class SecurityContext:
    """Security context for requests"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    authenticated: bool = False
    permissions: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class RateLimiter:
    """Simple rate limiting implementation"""
    
    def __init__(self, max_requests: int = 60, window_minutes: int = 1):
        self.max_requests = max_requests
        self.window_seconds = window_minutes * 60
        self.requests = {}  # ip -> list of timestamps
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is within rate limits"""
        now = time.time()
        
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Clean old requests outside the window
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier] 
            if now - req_time < self.window_seconds
        ]
        
        # Check if under limit
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(now)
            return True
        
        return False
    
    def get_remaining(self, identifier: str) -> int:
        """Get remaining requests for identifier"""
        now = time.time()
        
        if identifier not in self.requests:
            return self.max_requests
        
        # Count recent requests
        recent_requests = sum(
            1 for req_time in self.requests[identifier]
            if now - req_time < self.window_seconds
        )
        
        return max(0, self.max_requests - recent_requests)

class SecurityManager:
    """Unified security management system"""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.file_security = FileSecurityManager()
        self.data_protection = data_protection_manager
        self.rate_limiter = RateLimiter(
            max_requests=self.config.max_requests_per_minute,
            window_minutes=1
        )
        
        # Security state
        self.active_sessions = {}  # session_id -> SecurityContext
        self.blocked_ips = set(self.config.blocked_ips)
        self.security_events = []  # Recent security events
        
        # Generate API key if not exists
        self.api_key = self._get_or_create_api_key()
        
        logger.info("Security manager initialized with comprehensive protection")
    
    def _get_or_create_api_key(self) -> str:
        """Get or create API key for authentication"""
        api_key_file = Path("api_key.txt")
        
        if api_key_file.exists():
            try:
                with open(api_key_file, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                logger.warning(f"Could not read API key: {e}")
        
        # Generate new API key
        api_key = f"vep_{secrets.token_urlsafe(32)}"
        
        try:
            with open(api_key_file, 'w') as f:
                f.write(api_key)
            os.chmod(api_key_file, 0o600)
            logger.info("Generated new API key")
        except Exception as e:
            logger.error(f"Failed to save API key: {e}")
        
        return api_key
    
    def create_security_context(
        self, 
        request_data: Dict[str, Any]
    ) -> SecurityContext:
        """Create security context from request data"""
        
        context = SecurityContext(
            user_id=request_data.get('user_id'),
            session_id=request_data.get('session_id'),
            ip_address=request_data.get('ip_address'),
            user_agent=request_data.get('user_agent'),
            metadata=request_data.get('metadata', {})
        )
        
        # Calculate risk score
        context.risk_score = self._calculate_risk_score(context)
        
        return context
    
    def _calculate_risk_score(self, context: SecurityContext) -> float:
        """Calculate security risk score (0-1, higher = more risky)"""
        risk_score = 0.0
        
        # IP-based risk
        if context.ip_address:
            if context.ip_address in self.blocked_ips:
                risk_score += 0.8
            elif not self._is_allowed_ip(context.ip_address):
                risk_score += 0.3
        else:
            risk_score += 0.2  # No IP provided
        
        # Session-based risk
        if not context.session_id:
            risk_score += 0.1
        
        # User agent analysis
        if context.user_agent:
            if self._is_suspicious_user_agent(context.user_agent):
                risk_score += 0.2
        else:
            risk_score += 0.1
        
        return min(1.0, risk_score)
    
    def _is_allowed_ip(self, ip_address: str) -> bool:
        """Check if IP address is allowed"""
        if not self.config.allowed_ips:
            return True  # No restrictions
        
        try:
            client_ip = ipaddress.ip_address(ip_address)
            for allowed_range in self.config.allowed_ips:
                if client_ip in ipaddress.ip_network(allowed_range, strict=False):
                    return True
        except Exception as e:
            logger.warning(f"Invalid IP address format: {ip_address} - {e}")
            return False
        
        return False
    
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check for suspicious user agent patterns"""
        suspicious_patterns = [
            r'(?i)(bot|crawler|spider|scraper)',
            r'(?i)(curl|wget|python|requests)',
            r'(?i)(test|exploit|hack)',
            r'^$',  # Empty user agent
        ]
        
        return any(re.search(pattern, user_agent) for pattern in suspicious_patterns)
    
    def authenticate_request(self, context: SecurityContext, api_key: str = None) -> bool:
        """Authenticate API request"""
        
        if not self.config.api_key_required:
            context.authenticated = True
            return True
        
        if api_key == self.api_key:
            context.authenticated = True
            logger.debug(f"Authenticated request from {context.ip_address}")
            return True
        
        logger.warning(f"Authentication failed for {context.ip_address}")
        self._log_security_event("authentication_failed", context)
        return False
    
    def check_rate_limits(self, context: SecurityContext) -> bool:
        """Check if request is within rate limits"""
        
        if not self.config.rate_limit_enabled:
            return True
        
        identifier = context.ip_address or "unknown"
        
        if not self.rate_limiter.is_allowed(identifier):
            logger.warning(f"Rate limit exceeded for {identifier}")
            self._log_security_event("rate_limit_exceeded", context)
            return False
        
        return True
    
    def validate_and_secure_file(
        self, 
        file_path: Path, 
        context: SecurityContext,
        original_filename: str = ""
    ) -> Tuple[bool, Optional[str], Optional[List[SecurityThreat]]]:
        """Comprehensive file validation and security processing"""
        
        try:
            # Step 1: Basic file validation
            if not self.config.file_validation_enabled:
                return True, None, []
            
            # Check file size
            if file_path.stat().st_size > self.config.max_file_size_mb * 1024 * 1024:
                logger.warning(f"File too large: {file_path}")
                return False, "File size exceeds maximum allowed", []
            
            # Check file extension
            if file_path.suffix.lower() not in self.config.allowed_extensions:
                logger.warning(f"Disallowed file extension: {file_path.suffix}")
                return False, f"File type {file_path.suffix} not allowed", []
            
            # Step 2: Security scanning
            threats = self.file_security.scan_file(file_path)
            
            # Filter out low-risk threats for production use
            high_risk_threats = [
                threat for threat in threats 
                if threat.risk_level in ['HIGH', 'CRITICAL']
            ]
            
            if high_risk_threats:
                threat_desc = ", ".join([t.threat_type for t in high_risk_threats])
                logger.error(f"Security threats detected in {file_path}: {threat_desc}")
                self._log_security_event("malicious_file_detected", context, {
                    "threats": [t.to_dict() for t in high_risk_threats],
                    "filename": original_filename
                })
                return False, f"Security threats detected: {threat_desc}", high_risk_threats
            
            # Step 3: Apply data protection
            if self.config.encryption_enabled:
                # Determine data category based on context
                data_category = self._determine_data_category(context, original_filename)
                
                record_id = self.data_protection.protect_file(
                    file_path,
                    category=data_category,
                    original_filename=original_filename,
                    user_consent=context.metadata.get('user_consent', False),
                    metadata=self._create_file_metadata(context, original_filename)
                )
                
                logger.info(f"Applied data protection: {record_id}")
                return True, record_id, threats
            
            return True, None, threats
            
        except Exception as e:
            logger.error(f"File security validation failed: {e}")
            return False, f"Security validation error: {str(e)}", []
    
    def _determine_data_category(self, context: SecurityContext, filename: str) -> DataCategory:
        """Determine appropriate data category for file"""
        
        # Check for personal identifiers in filename or metadata
        personal_indicators = ['personal', 'private', 'confidential', 'user']
        
        if any(indicator in filename.lower() for indicator in personal_indicators):
            return DataCategory.PERSONAL
        
        # Check context for personal data markers
        if context.user_id or context.metadata.get('contains_personal_data'):
            return DataCategory.PERSONAL
        
        # Default to anonymous for video processing
        return DataCategory.ANONYMOUS
    
    def _create_file_metadata(self, context: SecurityContext, filename: str) -> Dict[str, Any]:
        """Create metadata for file protection"""
        
        metadata = {
            'upload_time': datetime.now().isoformat(),
            'original_filename': filename,
            'file_extension': Path(filename).suffix if filename else '',
            'user_ip_hash': hashlib.sha256(
                (context.ip_address or 'unknown').encode()
            ).hexdigest()[:16] if context.ip_address else None,
            'session_id_hash': hashlib.sha256(
                (context.session_id or 'unknown').encode()
            ).hexdigest()[:16] if context.session_id else None,
            'risk_score': context.risk_score
        }
        
        # Add anonymized context metadata
        if context.metadata:
            metadata.update(
                self.data_protection.anonymize_metadata(context.metadata)
            )
        
        return metadata
    
    def access_protected_file(
        self, 
        record_id: str, 
        context: SecurityContext
    ) -> Optional[Path]:
        """Securely access a protected file"""
        
        try:
            # Audit the access attempt
            self.data_protection.audit_data_access(
                record_id, 
                "file_access", 
                {
                    'user_id': context.user_id,
                    'ip_address': context.ip_address,
                    'session_id': context.session_id
                }
            )
            
            return self.data_protection.access_protected_file(record_id)
            
        except Exception as e:
            logger.error(f"Failed to access protected file {record_id}: {e}")
            self._log_security_event("file_access_error", context, {
                "record_id": record_id,
                "error": str(e)
            })
            return None
    
    def _log_security_event(
        self, 
        event_type: str, 
        context: SecurityContext, 
        additional_data: Dict[str, Any] = None
    ):
        """Log security events for monitoring and analysis"""
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'ip_address': context.ip_address,
            'user_agent': context.user_agent,
            'user_id': context.user_id,
            'session_id': context.session_id,
            'risk_score': context.risk_score
        }
        
        if additional_data:
            event.update(additional_data)
        
        # Keep only recent events in memory
        self.security_events.append(event)
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-500:]
        
        # In production, send to SIEM or security monitoring system
        if self.config.audit_logging_enabled:
            logger.warning(f"SECURITY_EVENT: {event}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status and metrics"""
        
        now = datetime.now()
        recent_events = [
            event for event in self.security_events
            if (now - datetime.fromisoformat(event['timestamp'])).seconds < 3600
        ]
        
        return {
            'active_sessions': len(self.active_sessions),
            'blocked_ips': len(self.blocked_ips),
            'recent_events': len(recent_events),
            'data_protection_summary': self.data_protection.get_protection_summary(),
            'rate_limiter_status': {
                'max_requests_per_minute': self.config.max_requests_per_minute,
                'window_seconds': self.rate_limiter.window_seconds
            },
            'security_config': {
                'api_key_required': self.config.api_key_required,
                'file_validation_enabled': self.config.file_validation_enabled,
                'encryption_enabled': self.config.encryption_enabled,
                'audit_logging_enabled': self.config.audit_logging_enabled
            }
        }
    
    def cleanup_expired_data(self):
        """Clean up expired security data"""
        
        # Clean up expired data protection records
        self.data_protection.cleanup_expired_data()
        
        # Clean up old security events
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.security_events = [
            event for event in self.security_events
            if datetime.fromisoformat(event['timestamp']) > cutoff_time
        ]
        
        # Clean up expired sessions
        expired_sessions = []
        for session_id, context in self.active_sessions.items():
            if (datetime.now() - context.timestamp).seconds > self.config.session_timeout_minutes * 60:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

# Decorator for securing API endpoints
def secure_endpoint(
    security_manager: SecurityManager,
    require_auth: bool = True,
    check_rate_limit: bool = True
):
    """Decorator to secure API endpoints"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract security context from request
            # This would typically come from Flask/FastAPI request object
            request_data = kwargs.get('security_context', {})
            context = security_manager.create_security_context(request_data)
            
            # Check IP blocking
            if context.ip_address in security_manager.blocked_ips:
                raise SecurityError(
                    message="Access denied from blocked IP",
                    error_code=ErrorCode.SECURITY_ACCESS_DENIED
                )
            
            # Rate limiting
            if check_rate_limit and not security_manager.check_rate_limits(context):
                raise SecurityError(
                    message="Rate limit exceeded",
                    error_code=ErrorCode.SECURITY_RATE_LIMIT
                )
            
            # Authentication
            if require_auth:
                api_key = request_data.get('api_key')
                if not security_manager.authenticate_request(context, api_key):
                    raise SecurityError(
                        message="Authentication required",
                        error_code=ErrorCode.SECURITY_AUTHENTICATION_FAILED
                    )
            
            # Add context to kwargs for the endpoint function
            kwargs['security_context'] = context
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Global security manager instance
security_manager = SecurityManager()