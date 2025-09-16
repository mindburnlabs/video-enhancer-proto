"""
Production Configuration Management for SOTA Video Enhancer
Environment-based settings for development, staging, and production deployments.
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
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model-specific configuration settings."""
    device: str = "cuda"
    precision: str = "fp16"  # fp16, fp32, int8
    max_memory_gb: float = 16.0
    tile_size: int = 512
    batch_size: int = 1
    enable_compilation: bool = False
    cache_models: bool = True
    model_cache_dir: str = "models/cache"

@dataclass
class PerformanceConfig:
    """Performance and resource optimization settings."""
    max_concurrent_requests: int = 3
    request_timeout_seconds: int = 300
    enable_model_warming: bool = True
    enable_memory_optimization: bool = True
    enable_tensorrt: bool = False
    enable_xformers: bool = True
    use_gpu_memory_fraction: float = 0.9

@dataclass
class SecurityConfig:
    """Security and access control settings."""
    enable_auth: bool = False
    api_key: Optional[str] = None
    rate_limit_requests_per_minute: int = 30
    max_file_size_mb: int = 500
    allowed_file_extensions: List[str] = field(default_factory=lambda: ['.mp4', '.avi', '.mov', '.mkv'])
    enable_cors: bool = True
    trusted_origins: List[str] = field(default_factory=list)

@dataclass
class MonitoringConfig:
    """Monitoring and observability settings."""
    enable_metrics: bool = True
    metrics_port: int = 7861
    enable_health_checks: bool = True
    log_level: str = "INFO"
    enable_request_logging: bool = True
    enable_performance_tracking: bool = True
    metrics_retention_hours: int = 24

@dataclass
class StorageConfig:
    """Storage and file management settings."""
    temp_dir: str = "data/temp"
    output_dir: str = "data/output"
    models_dir: str = "models"
    cleanup_temp_files: bool = True
    temp_file_retention_hours: int = 2
    max_temp_storage_gb: float = 50.0

class ProductionConfig:
    """Main configuration class with environment-based settings."""
    
    ENVIRONMENTS = ["development", "staging", "production"]
    
    def __init__(self, environment: Optional[str] = None):
        self.environment = environment or self._detect_environment()
        self._validate_environment()
        
        # Load base configuration
        self.model = ModelConfig()
        self.performance = PerformanceConfig()
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
        self.storage = StorageConfig()
        
        # Feature toggles and license gating
        self.license_mode: str = os.getenv('LICENSE_MODE', 'permissive_only')  # permissive_only | allow_all
        self.enable_face_expert: bool = os.getenv('ENABLE_FACE_EXPERT', '0') in ['1', 'true', 'True']
        self.enable_hfr: bool = os.getenv('ENABLE_HFR', '0') in ['1', 'true', 'True']
        self.seedvr2_3b_dir: Optional[str] = os.getenv('SEEDVR2_3B_DIR')
        self.seedvr2_7b_dir: Optional[str] = os.getenv('SEEDVR2_7B_DIR')
        
        # Apply environment-specific overrides
        self._apply_environment_config()
        
        # Apply environment variable overrides
        self._apply_env_var_overrides()
        
        # Validate final configuration
        self._validate_config()
        
        logger.info(f"🔧 Configuration loaded for environment: {self.environment}")
        logger.info(f"License mode: {self.license_mode}, Face expert: {self.enable_face_expert}, HFR: {self.enable_hfr}")
    
    def _detect_environment(self) -> str:
        """Detect environment from various sources."""
        # Check environment variable
        env = os.getenv('ENVIRONMENT', os.getenv('ENV', 'development')).lower()
        
        # Check Hugging Face Spaces
        if os.getenv('SPACE_ID'):
            return 'production'
        
        # Check Docker environment
        if os.path.exists('/.dockerenv'):
            return 'production'
        
        # Check for common production indicators
        if any(os.getenv(var) for var in ['KUBERNETES_SERVICE_HOST', 'AWS_REGION', 'HEROKU_APP_NAME']):
            return 'production'
        
        return env
    
    def _validate_environment(self):
        """Validate environment setting."""
        if self.environment not in self.ENVIRONMENTS:
            logger.warning(f"Unknown environment '{self.environment}', defaulting to 'development'")
            self.environment = 'development'
    
    def _apply_environment_config(self):
        """Apply environment-specific configuration overrides."""
        if self.environment == 'development':
            self._apply_development_config()
        elif self.environment == 'staging':
            self._apply_staging_config()
        elif self.environment == 'production':
            self._apply_production_config()
    
    def _apply_development_config(self):
        """Development environment configuration."""
        logger.info("📝 Applying development configuration")
        
        # More permissive settings for development
        self.model.precision = "fp32"
        self.model.tile_size = 256
        self.model.cache_models = False
        
        self.performance.max_concurrent_requests = 1
        self.performance.enable_model_warming = False
        self.performance.enable_memory_optimization = False
        
        self.security.rate_limit_requests_per_minute = 100
        self.security.max_file_size_mb = 100
        
        self.monitoring.log_level = "DEBUG"
        self.monitoring.enable_performance_tracking = True
        
        self.storage.cleanup_temp_files = False
        self.storage.temp_file_retention_hours = 24
    
    def _apply_staging_config(self):
        """Staging environment configuration."""
        logger.info("🧪 Applying staging configuration")
        
        # Production-like but more relaxed
        self.model.precision = "fp16"
        self.model.tile_size = 512
        
        self.performance.max_concurrent_requests = 2
        self.performance.enable_model_warming = True
        self.performance.enable_memory_optimization = True
        
        self.security.rate_limit_requests_per_minute = 60
        self.security.max_file_size_mb = 300
        
        self.monitoring.log_level = "INFO"
        self.monitoring.metrics_retention_hours = 12
    
    def _apply_production_config(self):
        """Production environment configuration."""
        logger.info("🚀 Applying production configuration")
        
        # Optimized for performance and reliability
        self.model.precision = "fp16"
        self.model.tile_size = 512
        self.model.enable_compilation = True
        self.model.cache_models = True
        
        self.performance.max_concurrent_requests = 3
        self.performance.enable_model_warming = True
        self.performance.enable_memory_optimization = True
        self.performance.enable_xformers = True
        
        self.security.enable_auth = os.getenv('ENABLE_AUTH', 'false').lower() == 'true'
        self.security.rate_limit_requests_per_minute = 30
        self.security.max_file_size_mb = 500
        
        self.monitoring.log_level = "INFO"
        self.monitoring.enable_metrics = True
        self.monitoring.enable_health_checks = True
        
        self.storage.cleanup_temp_files = True
        self.storage.temp_file_retention_hours = 2
        self.storage.max_temp_storage_gb = 20.0
    
    def _apply_env_var_overrides(self):
        """Apply environment variable overrides."""
        logger.info("🔧 Applying environment variable overrides")
        
        # Model configuration
        if os.getenv('MODEL_DEVICE'):
            self.model.device = os.getenv('MODEL_DEVICE')
        if os.getenv('MODEL_PRECISION'):
            self.model.precision = os.getenv('MODEL_PRECISION')
        if os.getenv('MAX_MEMORY_GB'):
            self.model.max_memory_gb = float(os.getenv('MAX_MEMORY_GB'))
        if os.getenv('TILE_SIZE'):
            self.model.tile_size = int(os.getenv('TILE_SIZE'))
        
        # Performance configuration
        if os.getenv('MAX_CONCURRENT_REQUESTS'):
            self.performance.max_concurrent_requests = int(os.getenv('MAX_CONCURRENT_REQUESTS'))
        if os.getenv('REQUEST_TIMEOUT'):
            self.performance.request_timeout_seconds = int(os.getenv('REQUEST_TIMEOUT'))
        if os.getenv('ENABLE_XFORMERS'):
            self.performance.enable_xformers = os.getenv('ENABLE_XFORMERS').lower() == 'true'
        
        # Security configuration
        if os.getenv('API_KEY'):
            self.security.api_key = os.getenv('API_KEY')
            self.security.enable_auth = True
        if os.getenv('RATE_LIMIT_RPM'):
            self.security.rate_limit_requests_per_minute = int(os.getenv('RATE_LIMIT_RPM'))
        if os.getenv('MAX_FILE_SIZE_MB'):
            self.security.max_file_size_mb = int(os.getenv('MAX_FILE_SIZE_MB'))
        
        # Monitoring configuration
        if os.getenv('LOG_LEVEL'):
            self.monitoring.log_level = os.getenv('LOG_LEVEL').upper()
        if os.getenv('METRICS_PORT'):
            self.monitoring.metrics_port = int(os.getenv('METRICS_PORT'))
        
        # Storage configuration
        if os.getenv('TEMP_DIR'):
            self.storage.temp_dir = os.getenv('TEMP_DIR')
        if os.getenv('MODELS_DIR'):
            self.storage.models_dir = os.getenv('MODELS_DIR')
        
        # Feature toggles and model paths
        if os.getenv('LICENSE_MODE'):
            self.license_mode = os.getenv('LICENSE_MODE')
        if os.getenv('ENABLE_FACE_EXPERT'):
            self.enable_face_expert = os.getenv('ENABLE_FACE_EXPERT') in ['1', 'true', 'True']
        if os.getenv('ENABLE_HFR'):
            self.enable_hfr = os.getenv('ENABLE_HFR') in ['1', 'true', 'True']
        if os.getenv('SEEDVR2_3B_DIR'):
            self.seedvr2_3b_dir = os.getenv('SEEDVR2_3B_DIR')
        if os.getenv('SEEDVR2_7B_DIR'):
            self.seedvr2_7b_dir = os.getenv('SEEDVR2_7B_DIR')
    
    def _validate_config(self):
        """Validate final configuration values."""
        errors = []
        
        # Validate model config
        if self.model.device not in ['cpu', 'cuda', 'auto']:
            errors.append(f"Invalid device: {self.model.device}")
        if self.model.precision not in ['fp16', 'fp32', 'int8']:
            errors.append(f"Invalid precision: {self.model.precision}")
        if self.model.max_memory_gb <= 0:
            errors.append("max_memory_gb must be positive")
        
        # Validate performance config
        if self.performance.max_concurrent_requests <= 0:
            errors.append("max_concurrent_requests must be positive")
        if not 0 < self.performance.use_gpu_memory_fraction <= 1:
            errors.append("use_gpu_memory_fraction must be between 0 and 1")
        
        # Validate security config
        if self.security.max_file_size_mb <= 0:
            errors.append("max_file_size_mb must be positive")
        if self.security.rate_limit_requests_per_minute <= 0:
            errors.append("rate_limit_requests_per_minute must be positive")
        
        # Validate monitoring config
        if self.monitoring.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            errors.append(f"Invalid log_level: {self.monitoring.log_level}")
        
        if errors:
            error_msg = "Configuration validation errors: " + "; ".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("✅ Configuration validation passed")
    
    def create_directories(self):
        """Create necessary directories based on configuration."""
        directories = [
            self.storage.temp_dir,
            self.storage.output_dir,
            self.storage.models_dir,
            self.model.model_cache_dir,
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    def get_model_path(self, model_name: str) -> str:
        """Get full path for a model file."""
        return os.path.join(self.storage.models_dir, "checkpoints", f"{model_name}.pth")
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available and should be used."""
        if self.model.device == 'cpu':
            return False
        elif self.model.device == 'cuda':
            try:
                import torch
                return torch.cuda.is_available()
            except ImportError:
                return False
        else:  # auto
            try:
                import torch
                return torch.cuda.is_available()
            except ImportError:
                return False
    
    def get_effective_device(self) -> str:
        """Get the effective device to use."""
        if self.model.device == 'auto':
            return 'cuda' if self.is_gpu_available() else 'cpu'
        return self.model.device
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'environment': self.environment,
            'model': {
                'device': self.model.device,
                'precision': self.model.precision,
                'max_memory_gb': self.model.max_memory_gb,
                'tile_size': self.model.tile_size,
                'batch_size': self.model.batch_size,
                'enable_compilation': self.model.enable_compilation,
                'cache_models': self.model.cache_models
            },
            'performance': {
                'max_concurrent_requests': self.performance.max_concurrent_requests,
                'request_timeout_seconds': self.performance.request_timeout_seconds,
                'enable_model_warming': self.performance.enable_model_warming,
                'enable_memory_optimization': self.performance.enable_memory_optimization,
                'enable_xformers': self.performance.enable_xformers
            },
            'security': {
                'enable_auth': self.security.enable_auth,
                'rate_limit_requests_per_minute': self.security.rate_limit_requests_per_minute,
                'max_file_size_mb': self.security.max_file_size_mb,
                'allowed_file_extensions': self.security.allowed_file_extensions
            },
            'monitoring': {
                'enable_metrics': self.monitoring.enable_metrics,
                'metrics_port': self.monitoring.metrics_port,
                'log_level': self.monitoring.log_level,
                'enable_performance_tracking': self.monitoring.enable_performance_tracking
            },
            'storage': {
                'temp_dir': self.storage.temp_dir,
                'output_dir': self.storage.output_dir,
                'models_dir': self.storage.models_dir,
                'cleanup_temp_files': self.storage.cleanup_temp_files
            }
        }
    
    def save_config(self, filepath: str):
        """Save configuration to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to: {filepath}")


# Global configuration instance
_config = None

def get_config(environment: Optional[str] = None) -> ProductionConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = ProductionConfig(environment)
    return _config

def reload_config(environment: Optional[str] = None):
    """Reload configuration (useful for testing)."""
    global _config
    _config = ProductionConfig(environment)
    return _config


if __name__ == "__main__":
    # Test configuration loading
    config = ProductionConfig()
    print(f"Environment: {config.environment}")
    print(f"Device: {config.get_effective_device()}")
    print(f"GPU Available: {config.is_gpu_available()}")
    
    # Save example configuration
    config.save_config("config_example.json")