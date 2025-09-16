#!/usr/bin/env python3
"""
Simple Security Integration Test

Focused test to verify core security functionality works correctly.
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

def test_security_integration():
    """Test core security integration functionality"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            
            # Test 1: Import security modules
            from utils.security_integration import SecurityManager, SecurityConfig, SecurityContext
            from utils.data_protection import DataProtectionManager, DataCategory
            from utils.file_security import FileSecurityManager, SecurityThreat
            logger.info("✅ Security modules imported successfully")
            
            # Test 2: Initialize security manager
            config = SecurityConfig(
                api_key_required=False,
                file_validation_enabled=True,
                encryption_enabled=True,
                allowed_extensions=['.mp4', '.txt']
            )
            
            manager = SecurityManager(config)
            logger.info("✅ Security manager initialized")
            
            # Test 3: Create and validate test file
            test_file = Path("test_file.txt")
            test_content = b"This is a test file for security validation"
            test_file.write_bytes(test_content)
            
            context = SecurityContext(
                ip_address="127.0.0.1",
                user_agent="Test-Client",
                metadata={"source": "test", "user_consent": True}
            )
            
            # Test file security validation
            is_valid, result, threats = manager.validate_and_secure_file(
                test_file, context, "test_file.txt"
            )
            
            if is_valid and result:
                logger.info(f"✅ File security validation passed: {result}")
                
                # Test 4: Access protected file
                protected_path = manager.access_protected_file(result, context)
                if protected_path and protected_path.exists():
                    # Verify content
                    decrypted_content = protected_path.read_bytes()
                    if decrypted_content == test_content:
                        logger.info("✅ File encryption/decryption successful")
                        return True
                    else:
                        logger.error("❌ File content mismatch after decryption")
                else:
                    logger.error("❌ Could not access protected file")
            else:
                logger.error(f"❌ File security validation failed: {result}")
                
        return False
        
    except Exception as e:
        logger.error(f"❌ Security integration test failed: {e}")
        return False

def main():
    """Run the security integration test"""
    logger.info("🔒 Running Simple Security Integration Test...")
    
    success = test_security_integration()
    
    if success:
        logger.info("🎉 Security integration test PASSED!")
        return True
    else:
        logger.error("💥 Security integration test FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)