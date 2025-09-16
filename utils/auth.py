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
import time
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import threading
from collections import defaultdict

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from utils.error_handler import (
    APIError, ErrorCode, error_handler
)

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User role enumeration"""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    SERVICE = "service"

class ResourceAction(Enum):
    """Resource action types"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

@dataclass
class APIKey:
    """API Key information"""
    key_id: str
    key_hash: str
    name: str
    role: UserRole
    created_at: datetime
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    rate_limit: int = 100  # requests per hour
    allowed_ips: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RateLimitInfo:
    """Rate limiting information"""
    key: str
    requests_count: int
    window_start: datetime
    last_request: datetime

class AuthenticationManager:
    """Manages API authentication and authorization"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "auth_config.json"
        self.api_keys: Dict[str, APIKey] = {}
        self.rate_limits: Dict[str, RateLimitInfo] = {}
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self._lock = threading.Lock()
        
        # Load configuration
        self._load_auth_config()
        
        # Rate limiting settings
        self.rate_limit_window = timedelta(hours=1)
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
    
    def _load_auth_config(self):
        """Load authentication configuration from file"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Load API keys
                for key_data in config.get('api_keys', []):
                    api_key = APIKey(
                        key_id=key_data['key_id'],
                        key_hash=key_data['key_hash'],
                        name=key_data['name'],
                        role=UserRole(key_data['role']),
                        created_at=datetime.fromisoformat(key_data['created_at']),
                        last_used=datetime.fromisoformat(key_data['last_used']) if key_data.get('last_used') else None,
                        expires_at=datetime.fromisoformat(key_data['expires_at']) if key_data.get('expires_at') else None,
                        is_active=key_data.get('is_active', True),
                        rate_limit=key_data.get('rate_limit', 100),
                        allowed_ips=key_data.get('allowed_ips', []),
                        metadata=key_data.get('metadata', {})
                    )
                    self.api_keys[key_data['key_id']] = api_key
                
                logger.info(f"Loaded {len(self.api_keys)} API keys from config")
            else:
                # Create default admin key if no config exists
                self._create_default_admin_key()
                
        except Exception as e:
            logger.error(f"Failed to load auth config: {e}")
            self._create_default_admin_key()
    
    def _create_default_admin_key(self):
        """Create a default admin API key"""
        admin_key = self.create_api_key(
            name="Default Admin",
            role=UserRole.ADMIN,
            rate_limit=1000
        )
        logger.warning(f"Created default admin API key: {admin_key}")
        logger.warning("Please change this key in production!")
    
    def _save_auth_config(self):
        """Save authentication configuration to file"""
        try:
            config = {
                'api_keys': []
            }
            
            for key_id, api_key in self.api_keys.items():
                key_data = {
                    'key_id': api_key.key_id,
                    'key_hash': api_key.key_hash,
                    'name': api_key.name,
                    'role': api_key.role.value,
                    'created_at': api_key.created_at.isoformat(),
                    'last_used': api_key.last_used.isoformat() if api_key.last_used else None,
                    'expires_at': api_key.expires_at.isoformat() if api_key.expires_at else None,
                    'is_active': api_key.is_active,
                    'rate_limit': api_key.rate_limit,
                    'allowed_ips': api_key.allowed_ips,
                    'metadata': api_key.metadata
                }
                config['api_keys'].append(key_data)
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save auth config: {e}")
    
    def create_api_key(
        self,
        name: str,
        role: UserRole,
        expires_in_days: Optional[int] = None,
        rate_limit: int = 100,
        allowed_ips: List[str] = None
    ) -> str:
        """Create a new API key"""
        
        # Generate key components
        key_id = secrets.token_urlsafe(16)
        raw_key = secrets.token_urlsafe(32)
        full_key = f"vep_{key_id}_{raw_key}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # Set expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        # Create API key object
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            role=role,
            created_at=datetime.now(),
            expires_at=expires_at,
            rate_limit=rate_limit,
            allowed_ips=allowed_ips or []
        )
        
        with self._lock:
            self.api_keys[key_id] = api_key
            self._save_auth_config()
        
        logger.info(f"Created API key '{name}' with role {role.value}")
        return full_key
    
    def validate_api_key(self, api_key: str, client_ip: str) -> Optional[APIKey]:
        """Validate an API key and return key info if valid"""
        
        try:
            # Parse API key format: vep_{key_id}_{raw_key}
            if not api_key.startswith("vep_"):
                return None
            
            parts = api_key[4:].split("_", 1)
            if len(parts) != 2:
                return None
            
            key_id, raw_key = parts
            
            with self._lock:
                # Check if key exists
                if key_id not in self.api_keys:
                    self._record_failed_attempt(client_ip)
                    return None
                
                api_key_obj = self.api_keys[key_id]
                
                # Check if key is active
                if not api_key_obj.is_active:
                    return None
                
                # Check expiration
                if api_key_obj.expires_at and datetime.now() > api_key_obj.expires_at:
                    return None
                
                # Verify key hash
                provided_hash = hashlib.sha256(raw_key.encode()).hexdigest()
                if provided_hash != api_key_obj.key_hash:
                    self._record_failed_attempt(client_ip)
                    return None
                
                # Check IP restrictions
                if api_key_obj.allowed_ips and client_ip not in api_key_obj.allowed_ips:
                    logger.warning(f"API key access denied from IP {client_ip}")
                    return None
                
                # Check rate limiting
                if not self._check_rate_limit(key_id, api_key_obj.rate_limit):
                    raise APIError(
                        message="Rate limit exceeded",
                        error_code=ErrorCode.API_RATE_LIMITED
                    )
                
                # Update last used timestamp
                api_key_obj.last_used = datetime.now()
                self._save_auth_config()
                
                return api_key_obj
                
        except APIError:
            raise  # Re-raise rate limit errors
        except Exception as e:
            logger.error(f"API key validation error: {e}")
            self._record_failed_attempt(client_ip)
            return None
    
    def _check_rate_limit(self, key_id: str, limit: int) -> bool:
        """Check if the API key is within rate limits"""
        
        now = datetime.now()
        
        if key_id not in self.rate_limits:
            self.rate_limits[key_id] = RateLimitInfo(
                key=key_id,
                requests_count=1,
                window_start=now,
                last_request=now
            )
            return True
        
        rate_info = self.rate_limits[key_id]
        
        # Check if we're in a new window
        if now - rate_info.window_start >= self.rate_limit_window:
            # Reset counter for new window
            rate_info.requests_count = 1
            rate_info.window_start = now
            rate_info.last_request = now
            return True
        
        # Check if limit exceeded
        if rate_info.requests_count >= limit:
            return False
        
        # Increment counter
        rate_info.requests_count += 1
        rate_info.last_request = now
        return True
    
    def _record_failed_attempt(self, client_ip: str):
        """Record a failed authentication attempt"""
        
        now = datetime.now()
        
        # Clean old attempts (older than lockout duration)
        cutoff = now - self.lockout_duration
        self.failed_attempts[client_ip] = [
            attempt for attempt in self.failed_attempts[client_ip] 
            if attempt > cutoff
        ]
        
        # Add current attempt
        self.failed_attempts[client_ip].append(now)
        
        if len(self.failed_attempts[client_ip]) >= self.max_failed_attempts:
            logger.warning(f"IP {client_ip} has exceeded maximum failed attempts")
    
    def is_ip_locked_out(self, client_ip: str) -> bool:
        """Check if an IP is locked out due to failed attempts"""
        
        if client_ip not in self.failed_attempts:
            return False
        
        now = datetime.now()
        cutoff = now - self.lockout_duration
        
        # Clean old attempts
        recent_attempts = [
            attempt for attempt in self.failed_attempts[client_ip] 
            if attempt > cutoff
        ]
        
        self.failed_attempts[client_ip] = recent_attempts
        return len(recent_attempts) >= self.max_failed_attempts
    
    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key"""
        
        with self._lock:
            if key_id in self.api_keys:
                self.api_keys[key_id].is_active = False
                self._save_auth_config()
                logger.info(f"Revoked API key {key_id}")
                return True
            return False
    
    def list_api_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (without sensitive data)"""
        
        keys_info = []
        for key_id, api_key in self.api_keys.items():
            keys_info.append({
                'key_id': key_id,
                'name': api_key.name,
                'role': api_key.role.value,
                'created_at': api_key.created_at.isoformat(),
                'last_used': api_key.last_used.isoformat() if api_key.last_used else None,
                'expires_at': api_key.expires_at.isoformat() if api_key.expires_at else None,
                'is_active': api_key.is_active,
                'rate_limit': api_key.rate_limit
            })
        
        return keys_info
    
    def get_rate_limit_info(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get rate limit information for an API key"""
        
        if key_id not in self.rate_limits:
            return None
        
        rate_info = self.rate_limits[key_id]
        api_key = self.api_keys.get(key_id)
        
        if not api_key:
            return None
        
        now = datetime.now()
        window_remaining = self.rate_limit_window - (now - rate_info.window_start)
        
        return {
            'requests_made': rate_info.requests_count,
            'rate_limit': api_key.rate_limit,
            'requests_remaining': max(0, api_key.rate_limit - rate_info.requests_count),
            'window_remaining_seconds': max(0, int(window_remaining.total_seconds())),
            'last_request': rate_info.last_request.isoformat()
        }

# Global authentication manager
auth_manager = AuthenticationManager()

# FastAPI security scheme
security = HTTPBearer(auto_error=False)

def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    # Check for X-Forwarded-For header (proxy)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    # Check for X-Real-IP header (nginx)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fall back to direct connection IP
    return request.client.host

def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> APIKey:
    """FastAPI dependency to get current authenticated user"""
    
    client_ip = get_client_ip(request)
    
    # Check if IP is locked out
    if auth_manager.is_ip_locked_out(client_ip):
        raise APIError(
            message="IP address is temporarily locked due to too many failed attempts",
            error_code=ErrorCode.API_AUTHENTICATION_FAILED
        )
    
    # Check for API key
    if not credentials:
        raise APIError(
            message="Authentication required",
            error_code=ErrorCode.API_AUTHENTICATION_FAILED
        )
    
    # Validate API key
    api_key_obj = auth_manager.validate_api_key(credentials.credentials, client_ip)
    
    if not api_key_obj:
        raise APIError(
            message="Invalid or expired API key",
            error_code=ErrorCode.API_AUTHENTICATION_FAILED
        )
    
    return api_key_obj

def require_role(required_role: UserRole):
    """Decorator to require specific user role"""
    
    def role_dependency(current_user: APIKey = Depends(get_current_user)):
        # Admin can access everything
        if current_user.role == UserRole.ADMIN:
            return current_user
        
        # Check if user has required role
        role_hierarchy = {
            UserRole.READONLY: 0,
            UserRole.USER: 1,
            UserRole.SERVICE: 2,
            UserRole.ADMIN: 3
        }
        
        if role_hierarchy.get(current_user.role, 0) < role_hierarchy.get(required_role, 0):
            raise APIError(
                message=f"Insufficient permissions. Required role: {required_role.value}",
                error_code=ErrorCode.API_AUTHENTICATION_FAILED
            )
        
        return current_user
    
    return role_dependency

def check_resource_permission(
    resource_type: str, 
    action: ResourceAction, 
    current_user: APIKey = Depends(get_current_user)
) -> APIKey:
    """Check if user has permission for specific resource action"""
    
    # Admin has all permissions
    if current_user.role == UserRole.ADMIN:
        return current_user
    
    # Define permissions matrix
    permissions = {
        UserRole.READONLY: {
            "videos": [ResourceAction.READ],
            "jobs": [ResourceAction.READ],
            "models": [ResourceAction.READ]
        },
        UserRole.USER: {
            "videos": [ResourceAction.READ, ResourceAction.WRITE],
            "jobs": [ResourceAction.READ, ResourceAction.WRITE],
            "models": [ResourceAction.READ]
        },
        UserRole.SERVICE: {
            "videos": [ResourceAction.READ, ResourceAction.WRITE, ResourceAction.DELETE],
            "jobs": [ResourceAction.READ, ResourceAction.WRITE, ResourceAction.DELETE],
            "models": [ResourceAction.READ, ResourceAction.WRITE]
        }
    }
    
    user_permissions = permissions.get(current_user.role, {})
    allowed_actions = user_permissions.get(resource_type, [])
    
    if action not in allowed_actions:
        raise APIError(
            message=f"Insufficient permissions for {action.value} on {resource_type}",
            error_code=ErrorCode.API_AUTHENTICATION_FAILED
        )
    
    return current_user

# Convenience functions for common permission checks
def require_admin():
    return require_role(UserRole.ADMIN)

def require_user():
    return require_role(UserRole.USER)

def require_service():
    return require_role(UserRole.SERVICE)