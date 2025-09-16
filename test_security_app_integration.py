#!/usr/bin/env python3
"""
Test Security Integration

Simple test to verify that the security system is properly integrated
with the main video enhancement application.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_security_imports():
    """Test that all security modules can be imported"""
    try:
        from utils.security_integration import SecurityManager, SecurityConfig
        from utils.data_protection import DataProtectionManager, DataCategory
        from utils.file_security import FileSecurityManager
        logger.info("✅ All security modules imported successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Security import failed: {e}")
        return False

def test_security_manager_initialization():
    """Test that security manager initializes correctly"""
    try:
        from utils.security_integration import SecurityManager, SecurityConfig
        
        config = SecurityConfig(
            api_key_required=False,  # Disable for testing
            file_validation_enabled=True,
            encryption_enabled=True
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            manager = SecurityManager(config)
            
            assert manager.config is not None
            assert manager.file_security is not None
            assert manager.data_protection is not None
            assert manager.rate_limiter is not None
            
            logger.info("✅ Security manager initialized successfully")
            return True
            
    except Exception as e:
        logger.error(f"❌ Security manager initialization failed: {e}")
        return False

def test_file_security_validation():
    """Test file security validation"""
    try:
        from utils.security_integration import SecurityManager, SecurityConfig, SecurityContext
        
        config = SecurityConfig(
            file_validation_enabled=True,
            encryption_enabled=True,
            allowed_extensions=['.mp4', '.avi']
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            manager = SecurityManager(config)
            
            # Create a test file
            test_file = Path(temp_dir) / "test_video.mp4"
            test_file.write_bytes(b"fake video content for testing")
            
            # Create security context
            context = SecurityContext(
                ip_address="127.0.0.1",
                user_agent="Test-Client",
                metadata={"source": "test", "user_consent": True}
            )
            
            # Test file validation
            is_valid, result, threats = manager.validate_and_secure_file(
                test_file, context, "test_video.mp4"
            )
            
            if is_valid:
                logger.info("✅ File security validation passed")
                logger.info(f"   Protected record ID: {result}")
                if threats:
                    logger.info(f"   Security threats detected (low risk): {len(threats)}")
                return True
            else:
                logger.error(f"❌ File security validation failed: {result}")
                return False
                
    except Exception as e:
        logger.error(f"❌ File security validation test failed: {e}")
        return False

def test_rate_limiting():
    """Test rate limiting functionality"""
    try:
        from utils.security_integration import SecurityManager, SecurityConfig, SecurityContext
        
        config = SecurityConfig(
            rate_limit_enabled=True,
            max_requests_per_minute=3  # Low limit for testing
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            manager = SecurityManager(config)
            
            context = SecurityContext(
                ip_address="127.0.0.1",
                user_agent="Test-Client"
            )
            
            # Should allow first few requests
            for i in range(3):
                assert manager.check_rate_limits(context), f"Request {i+1} should be allowed"
            
            # Should deny the 4th request
            assert not manager.check_rate_limits(context), "4th request should be rate limited"
            
            logger.info("✅ Rate limiting test passed")
            return True
            
    except Exception as e:
        logger.error(f"❌ Rate limiting test failed: {e}")
        return False

def test_data_encryption():
    """Test data encryption functionality"""
    try:
        from utils.data_protection import DataProtectionManager, DataCategory
        
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            
            # Create test file
            test_file = Path("test_data.txt")
            test_content = b"This is test data that should be encrypted"
            test_file.write_bytes(test_content)
            
            # Initialize data protection manager
            dp_manager = DataProtectionManager(data_dir="test_protected_data")
            
            # Protect the file with longer retention
            record_id = dp_manager.protect_file(
                test_file,
                DataCategory.ANONYMOUS,  # Use ANONYMOUS instead of TEMPORARY
                "test_data.txt",
                user_consent=True
            )
            
            assert record_id is not None, "Should get a record ID"
            
            # Access the protected file
            from utils.security_integration import SecurityContext
            context = SecurityContext()
            decrypted_path = dp_manager.access_protected_file(record_id)
            
            assert decrypted_path is not None, "Should be able to access protected file"
            assert decrypted_path.exists(), "Decrypted file should exist"
            
            # Verify content
            decrypted_content = decrypted_path.read_bytes()
            assert decrypted_content == test_content, "Decrypted content should match original"
            
            logger.info("✅ Data encryption test passed")
            return True
            
    except Exception as e:
        logger.error(f"❌ Data encryption test failed: {e}")
        return False

def test_app_integration():
    """Test integration with main app (import only)"""
    try:
        # This will test if the app can import with security integration
        os.environ['SECURITY_API_KEY_REQUIRED'] = 'false'  # Disable for testing
        os.environ['SECURITY_RATE_LIMIT_ENABLED'] = 'true'
        
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            
            # Create config directory and required files
            config_dir = Path('config')
            config_dir.mkdir(exist_ok=True)
            
            # Create a minimal logging config
            logging_config_path = config_dir / 'logging_config.py'
            logging_config_path.write_text("""
def setup_production_logging(log_level='INFO'):
    import logging
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    return logging.getLogger()

def get_performance_logger():
    import logging
    return logging.getLogger('performance')
""")
            
            # Try to import the main app components
            sys.path.insert(0, str(project_root))
            import app
        
        # Check that security manager was initialized
        assert hasattr(app, 'app_security_manager'), "App should have security manager"
        assert hasattr(app, 'security_config'), "App should have security config"
        
        # Test security manager
        manager = app.app_security_manager
        assert manager is not None, "Security manager should be initialized"
        
        # Get security status
        status = manager.get_security_status()
        assert isinstance(status, dict), "Security status should be a dict"
        assert 'security_config' in status, "Status should include security config"
        
        logger.info("✅ App integration test passed")
        logger.info(f"   Security features active: {len(status.get('security_config', {}))}")
        return True
        
    except Exception as e:
        logger.error(f"❌ App integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🔒 Starting Security Integration Tests...")
    
    tests = [
        ("Security Imports", test_security_imports),
        ("Security Manager Initialization", test_security_manager_initialization),
        ("File Security Validation", test_file_security_validation),
        ("Rate Limiting", test_rate_limiting),
        ("Data Encryption", test_data_encryption),
        ("App Integration", test_app_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                failed += 1
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name} CRASHED: {e}")
    
    logger.info(f"\n📊 Test Results:")
    logger.info(f"   Passed: {passed}")
    logger.info(f"   Failed: {failed}")
    logger.info(f"   Total:  {passed + failed}")
    
    if failed == 0:
        logger.info("🎉 All security integration tests passed!")
        return True
    else:
        logger.error(f"💥 {failed} tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)