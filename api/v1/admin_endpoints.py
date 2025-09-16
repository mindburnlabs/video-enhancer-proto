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

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from utils.auth import (
    auth_manager, get_current_user, require_admin, 
    APIKey, UserRole, get_client_ip
)
from utils.error_handler import error_handler, ErrorCode

logger = logging.getLogger(__name__)

# Create router for admin endpoints
router = APIRouter(
    prefix="/api/v1/admin",
    tags=["Admin"],
    responses={401: {"description": "Authentication required"}},
    dependencies=[Depends(require_admin())]  # All endpoints require admin role
)

class CreateAPIKeyRequest(BaseModel):
    """Request model for creating API keys"""
    name: str = Field(..., min_length=1, max_length=100, description="Human-readable name for the API key")
    role: UserRole = Field(..., description="User role for the API key")
    expires_in_days: Optional[int] = Field(None, ge=1, le=365, description="Expiration in days (optional)")
    rate_limit: int = Field(100, ge=1, le=10000, description="Request rate limit per hour")
    allowed_ips: List[str] = Field(default_factory=list, description="Allowed IP addresses (empty = all IPs)")

class APIKeyResponse(BaseModel):
    """Response model for API key information"""
    key_id: str
    name: str
    role: str
    created_at: str
    last_used: Optional[str]
    expires_at: Optional[str]
    is_active: bool
    rate_limit: int

class APIKeyCreatedResponse(BaseModel):
    """Response for newly created API key"""
    api_key: str = Field(..., description="The API key - store securely, it won't be shown again")
    key_info: APIKeyResponse

class SecurityStatsResponse(BaseModel):
    """Security statistics response"""
    total_api_keys: int
    active_api_keys: int
    failed_attempts_last_hour: int
    locked_ips: int
    top_error_codes: List[Dict[str, Any]]

@router.post("/api-keys", response_model=APIKeyCreatedResponse,
             summary="Create API Key",
             description="Create a new API key with specified permissions")
async def create_api_key(
    request: CreateAPIKeyRequest,
    current_user: APIKey = Depends(require_admin())
) -> APIKeyCreatedResponse:
    """Create a new API key"""
    
    try:
        # Create the API key
        new_key = auth_manager.create_api_key(
            name=request.name,
            role=request.role,
            expires_in_days=request.expires_in_days,
            rate_limit=request.rate_limit,
            allowed_ips=request.allowed_ips
        )
        
        # Get key info for response
        key_id = new_key.split('_')[1]  # Extract key_id from full key
        api_key_obj = auth_manager.api_keys[key_id]
        
        key_info = APIKeyResponse(
            key_id=api_key_obj.key_id,
            name=api_key_obj.name,
            role=api_key_obj.role.value,
            created_at=api_key_obj.created_at.isoformat(),
            last_used=api_key_obj.last_used.isoformat() if api_key_obj.last_used else None,
            expires_at=api_key_obj.expires_at.isoformat() if api_key_obj.expires_at else None,
            is_active=api_key_obj.is_active,
            rate_limit=api_key_obj.rate_limit
        )
        
        logger.info(f"Admin {current_user.name} created API key: {request.name}")
        
        return APIKeyCreatedResponse(
            api_key=new_key,
            key_info=key_info
        )
        
    except Exception as e:
        logger.error(f"Failed to create API key: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create API key: {str(e)}"
        )

@router.get("/api-keys", response_model=List[APIKeyResponse],
            summary="List API Keys",
            description="List all API keys (excluding the actual keys)")
async def list_api_keys(
    current_user: APIKey = Depends(require_admin())
) -> List[APIKeyResponse]:
    """List all API keys"""
    
    try:
        keys_info = auth_manager.list_api_keys()
        
        return [
            APIKeyResponse(
                key_id=key_info['key_id'],
                name=key_info['name'],
                role=key_info['role'],
                created_at=key_info['created_at'],
                last_used=key_info['last_used'],
                expires_at=key_info['expires_at'],
                is_active=key_info['is_active'],
                rate_limit=key_info['rate_limit']
            )
            for key_info in keys_info
        ]
        
    except Exception as e:
        logger.error(f"Failed to list API keys: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve API keys"
        )

@router.delete("/api-keys/{key_id}",
               summary="Revoke API Key",
               description="Revoke an API key by marking it as inactive")
async def revoke_api_key(
    key_id: str,
    current_user: APIKey = Depends(require_admin())
) -> Dict[str, str]:
    """Revoke an API key"""
    
    try:
        success = auth_manager.revoke_api_key(key_id)
        
        if success:
            logger.info(f"Admin {current_user.name} revoked API key: {key_id}")
            return {"message": f"API key {key_id} has been revoked"}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"API key {key_id} not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to revoke API key {key_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to revoke API key"
        )

@router.get("/api-keys/{key_id}/rate-limit",
            summary="Get Rate Limit Info",
            description="Get current rate limit status for an API key")
async def get_rate_limit_info(
    key_id: str,
    current_user: APIKey = Depends(require_admin())
) -> Dict[str, Any]:
    """Get rate limit information for an API key"""
    
    try:
        rate_info = auth_manager.get_rate_limit_info(key_id)
        
        if rate_info:
            return rate_info
        else:
            raise HTTPException(
                status_code=404,
                detail=f"No rate limit information found for key {key_id}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get rate limit info for {key_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve rate limit information"
        )

@router.get("/security-stats", response_model=SecurityStatsResponse,
            summary="Security Statistics",
            description="Get system security statistics and metrics")
async def get_security_stats(
    current_user: APIKey = Depends(require_admin())
) -> SecurityStatsResponse:
    """Get security statistics"""
    
    try:
        # Count API keys
        total_keys = len(auth_manager.api_keys)
        active_keys = sum(1 for key in auth_manager.api_keys.values() if key.is_active)
        
        # Count locked IPs
        locked_ips = sum(1 for ip in auth_manager.failed_attempts.keys() 
                        if auth_manager.is_ip_locked_out(ip))
        
        # Count recent failed attempts (last hour)
        now = datetime.now()
        recent_failures = 0
        for attempts in auth_manager.failed_attempts.values():
            recent_failures += sum(1 for attempt in attempts 
                                 if (now - attempt).total_seconds() < 3600)
        
        # Get error statistics from error handler
        error_stats = error_handler.get_error_stats()
        top_errors = error_stats.get('top_errors', [])[:5]  # Top 5 errors
        
        return SecurityStatsResponse(
            total_api_keys=total_keys,
            active_api_keys=active_keys,
            failed_attempts_last_hour=recent_failures,
            locked_ips=locked_ips,
            top_error_codes=[
                {"error_code": error[0], "count": error[1]} 
                for error in top_errors
            ]
        )
        
    except Exception as e:
        logger.error(f"Failed to get security stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve security statistics"
        )

@router.post("/clear-failed-attempts",
             summary="Clear Failed Attempts",
             description="Clear failed login attempts for a specific IP or all IPs")
async def clear_failed_attempts(
    ip_address: Optional[str] = None,
    current_user: APIKey = Depends(require_admin())
) -> Dict[str, str]:
    """Clear failed authentication attempts"""
    
    try:
        if ip_address:
            # Clear for specific IP
            if ip_address in auth_manager.failed_attempts:
                auth_manager.failed_attempts[ip_address].clear()
                message = f"Cleared failed attempts for IP {ip_address}"
            else:
                message = f"No failed attempts found for IP {ip_address}"
        else:
            # Clear all failed attempts
            auth_manager.failed_attempts.clear()
            message = "Cleared all failed authentication attempts"
        
        logger.info(f"Admin {current_user.name}: {message}")
        return {"message": message}
        
    except Exception as e:
        logger.error(f"Failed to clear failed attempts: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to clear failed attempts"
        )

@router.get("/audit-log",
            summary="Security Audit Log", 
            description="Get recent security-related events and API key usage")
async def get_audit_log(
    limit: int = 100,
    current_user: APIKey = Depends(require_admin())
) -> Dict[str, Any]:
    """Get security audit information"""
    
    try:
        # This is a simplified audit log
        # In production, you would implement proper audit logging
        
        audit_events = []
        
        # Recent API key usage
        for key_id, api_key in auth_manager.api_keys.items():
            if api_key.last_used:
                audit_events.append({
                    "timestamp": api_key.last_used.isoformat(),
                    "event_type": "api_key_used",
                    "key_id": key_id,
                    "key_name": api_key.name,
                    "role": api_key.role.value
                })
        
        # Failed attempts by IP
        for ip, attempts in auth_manager.failed_attempts.items():
            for attempt in attempts[-5:]:  # Last 5 attempts per IP
                audit_events.append({
                    "timestamp": attempt.isoformat(),
                    "event_type": "authentication_failed",
                    "ip_address": ip,
                    "locked": auth_manager.is_ip_locked_out(ip)
                })
        
        # Sort by timestamp (most recent first)
        audit_events.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return {
            "audit_events": audit_events[:limit],
            "total_events": len(audit_events),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get audit log: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve audit log"
        )

@router.get("/system-health",
            summary="System Health Check",
            description="Check system health and security status")
async def system_health_check(
    current_user: APIKey = Depends(require_admin())
) -> Dict[str, Any]:
    """Perform system health check"""
    
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {
                "authentication_system": "operational",
                "error_handling": "operational", 
                "rate_limiting": "operational",
                "file_security": "operational"
            },
            "warnings": [],
            "errors": []
        }
        
        # Check for system issues
        
        # Check for excessive failed attempts
        total_failures = sum(len(attempts) for attempts in auth_manager.failed_attempts.values())
        if total_failures > 100:  # Arbitrary threshold
            health_status["warnings"].append(f"High number of failed authentication attempts: {total_failures}")
        
        # Check for locked IPs
        locked_count = sum(1 for ip in auth_manager.failed_attempts.keys() 
                          if auth_manager.is_ip_locked_out(ip))
        if locked_count > 10:  # Arbitrary threshold
            health_status["warnings"].append(f"Many IPs are locked out: {locked_count}")
        
        # Check error rate
        error_stats = error_handler.get_error_stats()
        if error_stats.get('total_errors', 0) > 1000:  # Arbitrary threshold
            health_status["warnings"].append(f"High error rate: {error_stats['total_errors']} total errors")
        
        # Overall status
        if health_status["errors"]:
            health_status["status"] = "unhealthy"
        elif health_status["warnings"]:
            health_status["status"] = "warning"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }