#!/usr/bin/env python3
"""
Face Restoration Backend Configuration
Manages NCNN/Python backend configuration and conditional loading logic.
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
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Union
import subprocess
import importlib
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class BackendConfig:
    """Configuration for a face restoration backend"""
    name: str
    enabled: bool
    priority: int
    device: str
    model_path: Optional[str] = None
    binary_path: Optional[str] = None
    threads: int = 4
    memory_limit_mb: Optional[int] = None
    additional_params: Dict = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


@dataclass
class FaceRestorationSettings:
    """Global face restoration settings"""
    auto_select_backend: bool = True
    fallback_enabled: bool = True
    quality_threshold: float = 0.3
    min_face_size: int = 64
    max_batch_size: int = 8
    enable_face_alignment: bool = True
    blend_ratio: float = 0.8
    edge_feather: int = 10


class FaceRestorationBackendManager:
    """Manages face restoration backends with intelligent selection and fallback"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.backends: Dict[str, BackendConfig] = {}
        self.settings = FaceRestorationSettings()
        self.available_backends: List[str] = []
        self.active_backend: Optional[str] = None
        
        # Initialize configuration
        self._initialize_config()
        self._detect_available_backends()
        self._select_optimal_backend()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        project_root = Path(__file__).parent.parent
        config_dir = project_root / "data" / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        return str(config_dir / "face_restoration_backends.json")
    
    def _initialize_config(self):
        """Initialize configuration from file or create defaults"""
        try:
            if Path(self.config_path).exists():
                self._load_config()
            else:
                self._create_default_config()
                self._save_config()
        except Exception as e:
            logger.warning(f"Failed to initialize config: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default backend configurations"""
        # NCNN Backend (Fast, GPU-accelerated via Vulkan)
        self.backends['ncnn'] = BackendConfig(
            name='ncnn',
            enabled=True,
            priority=1,  # Highest priority (preferred)
            device='vulkan',
            binary_path=self._get_ncnn_binary_path(),
            threads=4,
            memory_limit_mb=2048,
            additional_params={
                'model_name': 'gfpgan',
                'scale': 1,
                'tile_size': 400,
                'use_vulkan': True
            }
        )
        
        # Python Backend (Flexible, wide hardware support)
        self.backends['python'] = BackendConfig(
            name='python',
            enabled=True,
            priority=2,  # Second priority
            device='cuda',  # Will fallback to CPU if CUDA unavailable
            model_path=self._get_gfpgan_model_path(),
            memory_limit_mb=4096,
            additional_params={
                'model_name': 'GFPGANv1.4',
                'upscale': 1,
                'arch': 'clean',
                'channel_multiplier': 2,
                'bg_upsampler': None
            }
        )
        
        # CodeFormer Backend (Alternative Python backend)
        self.backends['codeformer'] = BackendConfig(
            name='codeformer',
            enabled=True,
            priority=3,  # Third priority
            device='cuda',
            model_path=self._get_codeformer_model_path(),
            memory_limit_mb=3072,
            additional_params={
                'model_name': 'CodeFormer',
                'upscale': 1,
                'fidelity_weight': 0.5,
                'bg_upsampler': None
            }
        )
        
        # CPU-only Backend (Fallback)
        self.backends['cpu_fallback'] = BackendConfig(
            name='cpu_fallback',
            enabled=True,
            priority=10,  # Lowest priority (last resort)
            device='cpu',
            threads=2,  # Limit threads on CPU
            memory_limit_mb=1024,
            additional_params={
                'model_name': 'GFPGAN_CPU',
                'tile_size': 200,  # Smaller tiles for CPU
                'precision': 'fp32'
            }
        )
    
    def _get_ncnn_binary_path(self) -> Optional[str]:
        """Get NCNN binary path if available"""
        project_root = Path(__file__).parent.parent
        binary_path = project_root / "data" / "models" / "gfpgan_ncnn" / "gfpgan-ncnn-vulkan"
        
        if binary_path.exists():
            return str(binary_path)
        
        # Check system PATH for installed NCNN
        try:
            result = subprocess.run(['which', 'gfpgan-ncnn-vulkan'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return None
    
    def _get_gfpgan_model_path(self) -> Optional[str]:
        """Get GFPGAN model path"""
        # Check project models directory
        project_root = Path(__file__).parent.parent
        model_path = project_root / "data" / "models" / "face_restoration" / "GFPGANv1.4.pth"
        
        if model_path.exists():
            return str(model_path)
        
        # Check user cache directory
        cache_path = Path.home() / ".cache" / "gfpgan" / "GFPGANv1.4.pth"
        if cache_path.exists():
            return str(cache_path)
        
        return None
    
    def _get_codeformer_model_path(self) -> Optional[str]:
        """Get CodeFormer model path"""
        project_root = Path(__file__).parent.parent
        model_path = project_root / "data" / "models" / "face_restoration" / "codeformer.pth"
        
        if model_path.exists():
            return str(model_path)
        
        cache_path = Path.home() / ".cache" / "codeformer" / "codeformer.pth"
        if cache_path.exists():
            return str(cache_path)
        
        return None
    
    def _detect_available_backends(self):
        """Detect which backends are actually available"""
        self.available_backends = []
        
        for backend_name, config in self.backends.items():
            if not config.enabled:
                continue
                
            is_available = False
            
            if backend_name == 'ncnn':
                is_available = self._check_ncnn_availability(config)
            elif backend_name in ['python', 'codeformer', 'cpu_fallback']:
                is_available = self._check_python_availability(config)
            
            if is_available:
                self.available_backends.append(backend_name)
                logger.info(f"✅ {backend_name} backend available")
            else:
                logger.warning(f"❌ {backend_name} backend not available")
        
        # Sort by priority
        self.available_backends.sort(key=lambda x: self.backends[x].priority)
    
    def _check_ncnn_availability(self, config: BackendConfig) -> bool:
        """Check if NCNN backend is available"""
        if not config.binary_path or not Path(config.binary_path).exists():
            return False
        
        # Check Vulkan support if required
        if config.device == 'vulkan':
            try:
                result = subprocess.run(['vulkaninfo'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode != 0:
                    logger.warning("Vulkan runtime not available for NCNN")
                    return False
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("vulkaninfo not found, Vulkan support uncertain")
        
        # Test binary execution
        try:
            result = subprocess.run([config.binary_path, '--help'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0 or 'gfpgan' in result.stderr.lower()
        except Exception:
            return False
    
    def _check_python_availability(self, config: BackendConfig) -> bool:
        """Check if Python backend is available"""
        if config.name == 'python':
            try:
                import gfpgan
                import basicsr
                return True
            except ImportError:
                return False
        
        elif config.name == 'codeformer':
            try:
                import codeformer
                return True
            except ImportError:
                try:
                    # CodeFormer might be available through other packages
                    import basicsr
                    return config.model_path is not None
                except ImportError:
                    return False
        
        elif config.name == 'cpu_fallback':
            try:
                import cv2
                import numpy as np
                return True  # Basic CPU processing always available
            except ImportError:
                return False
        
        return False
    
    def _select_optimal_backend(self):
        """Select the optimal backend based on availability and performance"""
        if not self.available_backends:
            logger.error("No face restoration backends available")
            return
        
        if self.settings.auto_select_backend:
            # Environment-based selection
            env_backend = os.getenv('FACE_RESTORATION_BACKEND')
            if env_backend and env_backend in self.available_backends:
                self.active_backend = env_backend
                logger.info(f"Using environment-specified backend: {env_backend}")
                return
            
            # Performance-based selection
            self.active_backend = self._select_performance_optimal_backend()
        else:
            # Use highest priority available backend
            self.active_backend = self.available_backends[0]
        
        logger.info(f"Selected active backend: {self.active_backend}")
    
    def _select_performance_optimal_backend(self) -> str:
        """Select backend based on performance characteristics"""
        # System capability detection
        system_info = self._detect_system_capabilities()
        
        # Backend selection logic
        if system_info['has_vulkan'] and 'ncnn' in self.available_backends:
            return 'ncnn'  # Best performance with Vulkan
        
        if system_info['has_cuda'] and 'python' in self.available_backends:
            return 'python'  # Good performance with CUDA
        
        if 'cpu_fallback' in self.available_backends:
            return 'cpu_fallback'  # Always works, slower
        
        # Return first available as fallback
        return self.available_backends[0] if self.available_backends else None
    
    def _detect_system_capabilities(self) -> Dict[str, bool]:
        """Detect system capabilities for backend selection"""
        capabilities = {
            'has_cuda': False,
            'has_vulkan': False,
            'has_opencl': False,
            'memory_gb': 0
        }
        
        # CUDA detection
        try:
            import torch
            capabilities['has_cuda'] = torch.cuda.is_available()
        except ImportError:
            try:
                result = subprocess.run(['nvidia-smi'], 
                                      capture_output=True, text=True, timeout=5)
                capabilities['has_cuda'] = result.returncode == 0
            except:
                pass
        
        # Vulkan detection
        try:
            result = subprocess.run(['vulkaninfo'], 
                                  capture_output=True, text=True, timeout=5)
            capabilities['has_vulkan'] = result.returncode == 0
        except:
            pass
        
        # Memory detection
        try:
            import psutil
            capabilities['memory_gb'] = psutil.virtual_memory().total // (1024**3)
        except ImportError:
            capabilities['memory_gb'] = 8  # Reasonable default
        
        return capabilities
    
    def get_active_backend_config(self) -> Optional[BackendConfig]:
        """Get configuration for the active backend"""
        if not self.active_backend:
            return None
        return self.backends.get(self.active_backend)
    
    def get_fallback_backends(self) -> List[str]:
        """Get list of fallback backends in priority order"""
        if not self.active_backend:
            return self.available_backends
        
        # Return available backends except the active one
        return [b for b in self.available_backends if b != self.active_backend]
    
    def set_active_backend(self, backend_name: str) -> bool:
        """Manually set active backend"""
        if backend_name not in self.available_backends:
            logger.error(f"Backend {backend_name} not available")
            return False
        
        self.active_backend = backend_name
        logger.info(f"Switched to backend: {backend_name}")
        return True
    
    def reload_backends(self):
        """Reload and re-detect available backends"""
        self._detect_available_backends()
        self._select_optimal_backend()
    
    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            
            # Load backends
            if 'backends' in data:
                for name, config_dict in data['backends'].items():
                    self.backends[name] = BackendConfig(**config_dict)
            
            # Load settings
            if 'settings' in data:
                for key, value in data['settings'].items():
                    if hasattr(self.settings, key):
                        setattr(self.settings, key, value)
            
            logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _save_config(self):
        """Save configuration to JSON file"""
        try:
            data = {
                'backends': {name: asdict(config) for name, config in self.backends.items()},
                'settings': asdict(self.settings)
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get_status(self) -> Dict:
        """Get comprehensive backend status"""
        return {
            'active_backend': self.active_backend,
            'available_backends': self.available_backends,
            'total_backends': len(self.backends),
            'config_path': self.config_path,
            'settings': asdict(self.settings),
            'backend_details': {
                name: {
                    'available': name in self.available_backends,
                    'priority': config.priority,
                    'device': config.device,
                    'enabled': config.enabled
                }
                for name, config in self.backends.items()
            }
        }
    
    def create_backend_instance(self, backend_name: Optional[str] = None):
        """Create an instance of the specified backend"""
        backend_name = backend_name or self.active_backend
        
        if not backend_name or backend_name not in self.available_backends:
            raise ValueError(f"Backend {backend_name} not available")
        
        config = self.backends[backend_name]
        
        if backend_name == 'ncnn':
            return self._create_ncnn_instance(config)
        elif backend_name in ['python', 'codeformer', 'cpu_fallback']:
            return self._create_python_instance(config)
        else:
            raise ValueError(f"Unknown backend: {backend_name}")
    
    def _create_ncnn_instance(self, config: BackendConfig):
        """Create NCNN backend instance"""
        from models.enhancement.backends.ncnn_backend import NCNNFaceRestoration
        
        return NCNNFaceRestoration(
            binary_path=config.binary_path,
            device=config.device,
            threads=config.threads,
            **config.additional_params
        )
    
    def _create_python_instance(self, config: BackendConfig):
        """Create Python backend instance"""
        if config.name == 'python':
            from models.enhancement.backends.gfpgan_backend import GFPGANBackend
            return GFPGANBackend(
                model_path=config.model_path,
                device=config.device,
                **config.additional_params
            )
        elif config.name == 'codeformer':
            from models.enhancement.backends.codeformer_backend import CodeFormerBackend
            return CodeFormerBackend(
                model_path=config.model_path,
                device=config.device,
                **config.additional_params
            )
        elif config.name == 'cpu_fallback':
            from models.enhancement.backends.cpu_backend import CPUFaceRestoration
            return CPUFaceRestoration(
                device='cpu',
                threads=config.threads,
                **config.additional_params
            )


# Global backend manager instance
_backend_manager = None


def get_backend_manager() -> FaceRestorationBackendManager:
    """Get or create the global backend manager instance"""
    global _backend_manager
    if _backend_manager is None:
        _backend_manager = FaceRestorationBackendManager()
    return _backend_manager


def get_active_backend():
    """Get the active face restoration backend instance"""
    manager = get_backend_manager()
    return manager.create_backend_instance()


def set_backend(backend_name: str) -> bool:
    """Set the active face restoration backend"""
    manager = get_backend_manager()
    return manager.set_active_backend(backend_name)


def get_backend_status() -> Dict:
    """Get face restoration backend status"""
    manager = get_backend_manager()
    return manager.get_status()


if __name__ == "__main__":
    # Test the backend manager
    logging.basicConfig(level=logging.INFO)
    
    manager = FaceRestorationBackendManager()
    status = manager.get_status()
    
    print("🔧 Face Restoration Backend Status:")
    print(f"   Active: {status['active_backend']}")
    print(f"   Available: {status['available_backends']}")
    print(f"   Total backends configured: {status['total_backends']}")
    
    for backend, details in status['backend_details'].items():
        status_icon = "✅" if details['available'] else "❌"
        print(f"   {status_icon} {backend}: {details['device']} (priority: {details['priority']})")
    
    if status['active_backend']:
        try:
            backend_instance = manager.create_backend_instance()
            print(f"✅ Successfully created {status['active_backend']} backend instance")
        except Exception as e:
            print(f"❌ Failed to create backend instance: {e}")