"""
Torchvision compatibility fix for Real-ESRGAN.

This module provides compatibility for torchvision.transforms.functional_tensor
which was deprecated in torchvision >= 0.13.0 and moved to torchvision.transforms.functional.
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


import sys
import warnings
import logging

logger = logging.getLogger(__name__)

def apply_torchvision_compatibility_fix():
    """
    Apply compatibility fix for torchvision.transforms.functional_tensor.
    
    This function creates a compatibility layer that redirects imports from
    the deprecated functional_tensor module to the new functional module.
    """
    try:
        import torchvision
        torchvision_version = torchvision.__version__
        
        # Parse version
        version_parts = torchvision_version.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        
        logger.info(f"🔧 TorchVision compatibility check: version {torchvision_version}")
        
        # Check if we need the compatibility fix
        if major > 0 or (major == 0 and minor >= 13):
            logger.info("🔧 Applying TorchVision functional_tensor compatibility fix...")
            
            try:
                # Try to import functional_tensor to see if it exists
                import torchvision.transforms.functional_tensor
                logger.info("✅ functional_tensor module exists, no fix needed")
                return True
                
            except ImportError:
                # functional_tensor doesn't exist, create compatibility layer
                logger.info("🔧 Creating functional_tensor compatibility layer...")
                
                # Import the new functional module
                import torchvision.transforms.functional as F
                
                # Create a mock functional_tensor module
                class MockFunctionalTensorModule:
                    """Mock module that redirects functional_tensor calls to functional."""
                    
                    def __getattr__(self, name):
                        """Redirect attribute access to the functional module."""
                        if hasattr(F, name):
                            attr = getattr(F, name)
                            
                            # Issue a deprecation warning for first use
                            if not hasattr(self, '_warned_functions'):
                                self._warned_functions = set()
                            
                            if name not in self._warned_functions:
                                warnings.warn(
                                    f"Using deprecated torchvision.transforms.functional_tensor.{name}. "
                                    f"Please use torchvision.transforms.functional.{name} instead.",
                                    DeprecationWarning,
                                    stacklevel=3
                                )
                                self._warned_functions.add(name)
                            
                            return attr
                        else:
                            raise AttributeError(f"module 'torchvision.transforms.functional_tensor' has no attribute '{name}'")
                
                # Create and inject the mock module
                mock_module = MockFunctionalTensorModule()
                sys.modules['torchvision.transforms.functional_tensor'] = mock_module
                
                # Also add it to the torchvision.transforms module
                import torchvision.transforms as transforms
                transforms.functional_tensor = mock_module
                
                logger.info("✅ TorchVision functional_tensor compatibility fix applied successfully")
                return True
                
        else:
            logger.info(f"✅ TorchVision {torchvision_version} - no compatibility fix needed")
            return True
            
    except ImportError as e:
        logger.warning(f"⚠️ TorchVision not available: {e}")
        return False
        
    except Exception as e:
        logger.error(f"❌ Failed to apply TorchVision compatibility fix: {e}")
        return False

def patch_realesrgan_import():
    """
    Patch Real-ESRGAN import to handle torchvision compatibility.
    
    This function monkey-patches the Real-ESRGAN module to handle the
    functional_tensor import issue gracefully.
    """
    try:
        # First apply the general compatibility fix
        apply_torchvision_compatibility_fix()
        
        # Then try to patch any specific Real-ESRGAN issues
        try:
            import realesrgan
            logger.info("✅ Real-ESRGAN import successful after compatibility fix")
            return True
            
        except ImportError as e:
            if "functional_tensor" in str(e):
                logger.warning(f"⚠️ Real-ESRGAN still has functional_tensor issues: {e}")
                # Additional specific patches can be added here if needed
                return False
            else:
                # Different import error, re-raise
                raise e
                
    except Exception as e:
        logger.error(f"❌ Failed to patch Real-ESRGAN import: {e}")
        return False

# Apply the fix when this module is imported
if __name__ != "__main__":
    apply_torchvision_compatibility_fix()