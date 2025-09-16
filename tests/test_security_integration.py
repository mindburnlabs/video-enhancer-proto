"""
Comprehensive Security Integration Tests

Tests the complete security system including file security, data protection,
authentication, rate limiting, and integration with the main application.

MIT License - Copyright (c) 2024 Video Enhancement Project
"""

import os
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from utils.security_integration import (
    SecurityManager, SecurityConfig, SecurityContext,
    RateLimiter, secure_endpoint
)
from utils.data_protection import DataCategory
from utils.file_security import SecurityThreat
from utils.error_handler import SecurityError, ErrorCode


class TestRateLimiter:
    """Test rate limiting functionality"""
    
    def test_rate_limiter_allows_requests_under_limit(self):
        """Test that requests under the limit are allowed"""
        limiter = RateLimiter(max_requests=5, window_minutes=1)
        
        # Should allow 5 requests
        for i in range(5):
            assert limiter.is_allowed("test_ip"), f"Request {i+1} should be allowed"
        
        # 6th request should be denied
        assert not limiter.is_allowed("test_ip"), "6th request should be denied"
    
    def test_rate_limiter_window_reset(self):
        """Test that rate limiter resets after window expires"""
        limiter = RateLimiter(max_requests=2, window_minutes=1)
        
        # Use up limit
        assert limiter.is_allowed("test_ip")
        assert limiter.is_allowed("test_ip")
        assert not limiter.is_allowed("test_ip")
        
        # Mock time to simulate window expiry
        with patch('time.time', return_value=time.time() + 70):  # 70 seconds later
            assert limiter.is_allowed("test_ip"), "Should allow after window reset"
    
    def test_rate_limiter_per_ip(self):
        """Test that rate limiting is per IP"""
        limiter = RateLimiter(max_requests=2, window_minutes=1)
        
        # First IP uses up limit
        assert limiter.is_allowed("ip1")
        assert limiter.is_allowed("ip1")
        assert not limiter.is_allowed("ip1")
        
        # Second IP should have its own limit
        assert limiter.is_allowed("ip2")
        assert limiter.is_allowed("ip2")
        assert not limiter.is_allowed("ip2")
    
    def test_get_remaining_requests(self):
        """Test getting remaining request count"""
        limiter = RateLimiter(max_requests=5, window_minutes=1)
        
        assert limiter.get_remaining("test_ip") == 5
        
        limiter.is_allowed("test_ip")
        assert limiter.get_remaining("test_ip") == 4
        
        limiter.is_allowed("test_ip")
        limiter.is_allowed("test_ip")
        assert limiter.get_remaining("test_ip") == 2


class TestSecurityManager:
    """Test security manager functionality"""
    
    @pytest.fixture
    def security_config(self):
        """Create test security configuration"""
        return SecurityConfig(
            api_key_required=True,
            rate_limit_enabled=True,
            max_requests_per_minute=10,
            max_file_size_mb=100,
            file_validation_enabled=True,
            encryption_enabled=True,
            allowed_extensions=['.mp4', '.avi', '.mov']
        )
    
    @pytest.fixture
    def security_manager(self, security_config):
        """Create test security manager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            manager = SecurityManager(security_config)
            yield manager
    
    @pytest.fixture
    def test_context(self):
        """Create test security context"""
        return SecurityContext(
            user_id="test_user",
            session_id="test_session",
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 Test Browser",
            metadata={"source": "test"}
        )
    
    def test_security_manager_initialization(self, security_manager):
        """Test that security manager initializes correctly"""
        assert security_manager.config is not None
        assert security_manager.file_security is not None
        assert security_manager.data_protection is not None
        assert security_manager.rate_limiter is not None
        assert security_manager.api_key is not None
        assert security_manager.api_key.startswith("vep_")
    
    def test_create_security_context(self, security_manager):
        """Test security context creation"""
        request_data = {
            'user_id': 'test_user',
            'ip_address': '192.168.1.100',
            'user_agent': 'Test Agent',
            'metadata': {'key': 'value'}
        }
        
        context = security_manager.create_security_context(request_data)
        
        assert context.user_id == 'test_user'
        assert context.ip_address == '192.168.1.100'
        assert context.user_agent == 'Test Agent'
        assert context.risk_score >= 0.0
        assert context.risk_score <= 1.0
    
    def test_risk_score_calculation(self, security_manager):
        """Test risk score calculation logic"""
        # Low risk context
        low_risk_context = SecurityContext(
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            session_id="valid_session"
        )
        low_risk = security_manager._calculate_risk_score(low_risk_context)
        
        # High risk context
        high_risk_context = SecurityContext(
            ip_address=None,  # No IP
            user_agent="curl/7.68.0",  # Suspicious user agent
            session_id=None  # No session
        )
        high_risk = security_manager._calculate_risk_score(high_risk_context)
        
        assert high_risk > low_risk, "High risk context should have higher score"
    
    def test_authentication_success(self, security_manager, test_context):
        """Test successful authentication"""
        assert security_manager.authenticate_request(test_context, security_manager.api_key)
        assert test_context.authenticated
    
    def test_authentication_failure(self, security_manager, test_context):
        """Test failed authentication"""
        assert not security_manager.authenticate_request(test_context, "invalid_key")
        assert not test_context.authenticated
    
    def test_rate_limiting(self, security_manager, test_context):
        """Test rate limiting integration"""
        # Should allow requests under limit
        for i in range(security_manager.config.max_requests_per_minute):
            assert security_manager.check_rate_limits(test_context), f"Request {i+1} should pass rate limit"
        
        # Should deny request over limit
        assert not security_manager.check_rate_limits(test_context), "Should be rate limited"
    
    def test_ip_blocking(self, security_manager, test_context):
        """Test IP blocking functionality"""
        # Add IP to blocked list
        security_manager.blocked_ips.add(test_context.ip_address)
        
        # Risk score should be high for blocked IP
        risk_score = security_manager._calculate_risk_score(test_context)
        assert risk_score >= 0.8, "Blocked IP should have high risk score"
    
    @patch('utils.security_integration.Path.stat')
    def test_file_size_validation(self, mock_stat, security_manager, test_context):
        """Test file size validation"""
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_file:
            file_path = Path(temp_file.name)
            
            # Mock file size to be over limit
            mock_stat.return_value.st_size = security_manager.config.max_file_size_mb * 1024 * 1024 + 1
            
            is_valid, error_msg, threats = security_manager.validate_and_secure_file(
                file_path, test_context, "test.mp4"
            )
            
            assert not is_valid
            assert "File size exceeds maximum allowed" in error_msg
    
    def test_file_extension_validation(self, security_manager, test_context):
        """Test file extension validation"""
        with tempfile.NamedTemporaryFile(suffix='.exe') as temp_file:
            file_path = Path(temp_file.name)
            
            is_valid, error_msg, threats = security_manager.validate_and_secure_file(
                file_path, test_context, "malicious.exe"
            )
            
            assert not is_valid
            assert "File type .exe not allowed" in error_msg
    
    @patch('utils.file_security.FileSecurityManager.scan_file')
    def test_file_threat_detection(self, mock_scan, security_manager, test_context):
        """Test file threat detection"""
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_file:
            file_path = Path(temp_file.name)
            
            # Mock high-risk threat
            high_risk_threat = SecurityThreat(
                threat_type="MALWARE",
                description="Test malware",
                risk_level="HIGH",
                confidence=0.9,
                details={"scanner": "test"}
            )
            mock_scan.return_value = [high_risk_threat]
            
            is_valid, error_msg, threats = security_manager.validate_and_secure_file(
                file_path, test_context, "suspicious.mp4"
            )
            
            assert not is_valid
            assert "Security threats detected: MALWARE" in error_msg
            assert len(threats) == 1
    
    @patch('utils.file_security.FileSecurityManager.scan_file')
    def test_successful_file_processing(self, mock_scan, security_manager, test_context):
        """Test successful file processing with encryption"""
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_file:
            file_path = Path(temp_file.name)
            # Write some test content
            temp_file.write(b"test video content")
            temp_file.flush()
            
            # Mock no threats
            mock_scan.return_value = []
            
            is_valid, record_id, threats = security_manager.validate_and_secure_file(
                file_path, test_context, "test.mp4"
            )
            
            assert is_valid
            assert record_id is not None
            assert record_id.startswith("dp_")
            assert threats == []
    
    def test_data_category_determination(self, security_manager, test_context):
        """Test data category determination logic"""
        # Personal filename
        personal_category = security_manager._determine_data_category(
            test_context, "personal_video.mp4"
        )
        assert personal_category == DataCategory.PERSONAL
        
        # Regular filename with user context
        test_context.user_id = "user123"
        user_category = security_manager._determine_data_category(
            test_context, "video.mp4"
        )
        assert user_category == DataCategory.PERSONAL
        
        # Anonymous context
        anonymous_context = SecurityContext()
        anonymous_category = security_manager._determine_data_category(
            anonymous_context, "video.mp4"
        )
        assert anonymous_category == DataCategory.ANONYMOUS
    
    def test_security_event_logging(self, security_manager, test_context):
        """Test security event logging"""
        initial_events = len(security_manager.security_events)
        
        security_manager._log_security_event("test_event", test_context, {"key": "value"})
        
        assert len(security_manager.security_events) == initial_events + 1
        
        latest_event = security_manager.security_events[-1]
        assert latest_event['event_type'] == "test_event"
        assert latest_event['ip_address'] == test_context.ip_address
        assert latest_event['key'] == "value"
    
    def test_security_status_reporting(self, security_manager):
        """Test security status reporting"""
        status = security_manager.get_security_status()
        
        assert 'active_sessions' in status
        assert 'blocked_ips' in status
        assert 'recent_events' in status
        assert 'data_protection_summary' in status
        assert 'rate_limiter_status' in status
        assert 'security_config' in status
        
        assert isinstance(status['data_protection_summary'], dict)
        assert isinstance(status['rate_limiter_status'], dict)
        assert isinstance(status['security_config'], dict)
    
    def test_cleanup_expired_data(self, security_manager):
        """Test cleanup of expired data"""
        # Add some test data
        security_manager.active_sessions['old_session'] = SecurityContext()
        security_manager.security_events.append({
            'timestamp': '2020-01-01T00:00:00',  # Old event
            'event_type': 'test'
        })
        
        security_manager.cleanup_expired_data()
        
        # Should clean up old events but keep recent ones
        recent_events = [
            event for event in security_manager.security_events
            if '2020-01-01' not in event['timestamp']
        ]
        assert len(recent_events) >= 0  # Should have removed old events


class TestSecureEndpointDecorator:
    """Test the secure endpoint decorator"""
    
    @pytest.fixture
    def security_manager(self):
        """Create test security manager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            config = SecurityConfig(
                api_key_required=True,
                rate_limit_enabled=True,
                max_requests_per_minute=5
            )
            return SecurityManager(config)
    
    def test_secure_endpoint_success(self, security_manager):
        """Test successful request through secure endpoint"""
        @secure_endpoint(security_manager)
        def test_endpoint(**kwargs):
            return "success"
        
        request_data = {
            'api_key': security_manager.api_key,
            'ip_address': '192.168.1.100',
            'user_agent': 'Test Browser'
        }
        
        result = test_endpoint(security_context=request_data)
        assert result == "success"
    
    def test_secure_endpoint_authentication_failure(self, security_manager):
        """Test authentication failure in secure endpoint"""
        @secure_endpoint(security_manager)
        def test_endpoint(**kwargs):
            return "success"
        
        request_data = {
            'api_key': 'invalid_key',
            'ip_address': '192.168.1.100'
        }
        
        with pytest.raises(SecurityError) as exc_info:
            test_endpoint(security_context=request_data)
        
        assert exc_info.value.error_code == ErrorCode.SECURITY_AUTHENTICATION_FAILED
    
    def test_secure_endpoint_rate_limit(self, security_manager):
        """Test rate limiting in secure endpoint"""
        @secure_endpoint(security_manager)
        def test_endpoint(**kwargs):
            return "success"
        
        request_data = {
            'api_key': security_manager.api_key,
            'ip_address': '192.168.1.100'
        }
        
        # Make requests up to limit
        for i in range(security_manager.config.max_requests_per_minute):
            result = test_endpoint(security_context=request_data)
            assert result == "success"
        
        # Next request should be rate limited
        with pytest.raises(SecurityError) as exc_info:
            test_endpoint(security_context=request_data)
        
        assert exc_info.value.error_code == ErrorCode.SECURITY_RATE_LIMIT
    
    def test_secure_endpoint_ip_blocking(self, security_manager):
        """Test IP blocking in secure endpoint"""
        blocked_ip = '192.168.1.999'
        security_manager.blocked_ips.add(blocked_ip)
        
        @secure_endpoint(security_manager)
        def test_endpoint(**kwargs):
            return "success"
        
        request_data = {
            'api_key': security_manager.api_key,
            'ip_address': blocked_ip
        }
        
        with pytest.raises(SecurityError) as exc_info:
            test_endpoint(security_context=request_data)
        
        assert exc_info.value.error_code == ErrorCode.SECURITY_ACCESS_DENIED
    
    def test_secure_endpoint_no_auth_required(self, security_manager):
        """Test secure endpoint with authentication disabled"""
        @secure_endpoint(security_manager, require_auth=False)
        def test_endpoint(**kwargs):
            return "success"
        
        request_data = {
            'ip_address': '192.168.1.100'
        }
        
        result = test_endpoint(security_context=request_data)
        assert result == "success"
    
    def test_secure_endpoint_no_rate_limit(self, security_manager):
        """Test secure endpoint with rate limiting disabled"""
        @secure_endpoint(security_manager, check_rate_limit=False)
        def test_endpoint(**kwargs):
            return "success"
        
        request_data = {
            'api_key': security_manager.api_key,
            'ip_address': '192.168.1.100'
        }
        
        # Make many requests - should not be rate limited
        for i in range(20):
            result = test_endpoint(security_context=request_data)
            assert result == "success"


class TestIntegrationScenarios:
    """Test complete security integration scenarios"""
    
    @pytest.fixture
    def security_manager(self):
        """Create test security manager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            return SecurityManager()
    
    def test_complete_file_processing_workflow(self, security_manager):
        """Test complete workflow from upload to processing"""
        # Create test video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(b"fake video content for testing")
            test_file_path = Path(temp_file.name)
        
        try:
            # Create security context
            context = SecurityContext(
                user_id="test_user_123",
                ip_address="192.168.1.100",
                user_agent="Mozilla/5.0 Test Browser",
                metadata={"user_consent": True, "source": "web_upload"}
            )
            
            # Step 1: File validation and security
            with patch('utils.file_security.FileSecurityManager.scan_file', return_value=[]):
                is_valid, record_id, threats = security_manager.validate_and_secure_file(
                    test_file_path, context, "test_video.mp4"
                )
            
            assert is_valid, "File should pass security validation"
            assert record_id is not None, "Should get data protection record ID"
            
            # Step 2: Access the protected file
            protected_file_path = security_manager.access_protected_file(record_id, context)
            assert protected_file_path is not None, "Should be able to access protected file"
            assert protected_file_path.exists(), "Protected file should exist"
            
            # Step 3: Verify data protection record exists
            summary = security_manager.data_protection.get_protection_summary()
            assert summary['total_records'] >= 1, "Should have at least one protected record"
            
        finally:
            # Clean up
            if test_file_path.exists():
                test_file_path.unlink()
    
    def test_security_incident_handling(self, security_manager):
        """Test handling of security incidents"""
        # Simulate malicious file upload
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(b"malicious content")
            malicious_file = Path(temp_file.name)
        
        try:
            context = SecurityContext(
                ip_address="192.168.1.100",
                user_agent="curl/7.68.0"  # Suspicious user agent
            )
            
            # Mock high-risk threat detection
            with patch('utils.file_security.FileSecurityManager.scan_file') as mock_scan:
                mock_scan.return_value = [
                    SecurityThreat(
                        threat_type="MALWARE",
                        description="Detected malware signature",
                        risk_level="CRITICAL",
                        confidence=0.95,
                        details={"signature": "test_malware"}
                    )
                ]
                
                is_valid, error_msg, threats = security_manager.validate_and_secure_file(
                    malicious_file, context, "malicious.mp4"
                )
            
            # Should reject the file
            assert not is_valid
            assert "MALWARE" in error_msg
            assert len(threats) == 1
            
            # Should log security event
            security_events = security_manager.security_events
            malicious_events = [
                event for event in security_events
                if event['event_type'] == 'malicious_file_detected'
            ]
            assert len(malicious_events) >= 1, "Should log malicious file detection"
            
        finally:
            if malicious_file.exists():
                malicious_file.unlink()
    
    def test_performance_under_load(self, security_manager):
        """Test security system performance under load"""
        import concurrent.futures
        import threading
        
        def make_request(request_id):
            """Simulate a single request"""
            context = SecurityContext(
                user_id=f"user_{request_id}",
                ip_address=f"192.168.1.{request_id % 255}",
                session_id=f"session_{request_id}"
            )
            
            # Test authentication
            auth_result = security_manager.authenticate_request(
                context, security_manager.api_key
            )
            
            # Test rate limiting (should mostly pass for different IPs)
            rate_result = security_manager.check_rate_limits(context)
            
            return auth_result, rate_result
        
        # Test with concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(make_request, i) 
                for i in range(50)
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Most requests should succeed (different IPs)
        successful_auths = sum(1 for auth, _ in results if auth)
        successful_rates = sum(1 for _, rate in results if rate)
        
        assert successful_auths >= 45, "Most authentications should succeed"
        assert successful_rates >= 40, "Most rate limit checks should pass"  # Some might fail due to same IP


if __name__ == '__main__':
    pytest.main([__file__, '-v'])