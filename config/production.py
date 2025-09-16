
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
from typing import List, Optional
from pathlib import Path

class ProductionConfig:
    """Production configuration with environment variables and secure defaults"""
    
    def __init__(self):
        # Server Configuration
        self.HOST = os.getenv('HOST', '0.0.0.0')
        self.PORT = int(os.getenv('PORT', 8080))
        self.WORKERS = int(os.getenv('WORKERS', 1))  # Single worker for GPU memory
        self.DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
        
        # Security Configuration
        self.JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', self._generate_secret_key())
        self.JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
        self.ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', 60))
        self.API_KEY_HEADER = os.getenv('API_KEY_HEADER', 'X-API-Key')
        
        # File Upload Limits
        self.MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 10 * 1024**3))  # 10GB
        self.ALLOWED_FORMATS = os.getenv('ALLOWED_FORMATS', '.mp4,.avi,.mov,.mkv,.webm,.m4v').split(',')
        self.UPLOAD_DIR = Path(os.getenv('UPLOAD_DIR', '/tmp/uploads'))
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Processing Configuration
        self.DEVICE = os.getenv('DEVICE', 'cuda' if self._is_cuda_available() else 'cpu')
        self.MAX_CONCURRENT_JOBS = int(os.getenv('MAX_CONCURRENT_JOBS', 4))
        self.JOB_TIMEOUT_MINUTES = int(os.getenv('JOB_TIMEOUT_MINUTES', 120))  # 2 hours
        self.WORKSPACE_PATH = Path(os.getenv('WORKSPACE_PATH', '/tmp/video_processor'))
        self.WORKSPACE_PATH.mkdir(parents=True, exist_ok=True)
        
        # Model Configuration
        self.MODEL_CACHE_DIR = Path(os.getenv('MODEL_CACHE_DIR', './models/.cache'))
        self.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.ENABLE_MODEL_OFFLOAD = os.getenv('ENABLE_MODEL_OFFLOAD', 'true').lower() == 'true'
        self.GPU_MEMORY_FRACTION = float(os.getenv('GPU_MEMORY_FRACTION', 0.9))
        
        # External Services
        self.VLLM_ENDPOINT = os.getenv('VLLM_ENDPOINT')  # Optional AI agent endpoint
        self.REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./video_enhancer.db')
        
        # Storage Configuration
        self.STORAGE_BACKEND = os.getenv('STORAGE_BACKEND', 'local')  # local, s3, gcs
        self.S3_BUCKET = os.getenv('S3_BUCKET')
        self.AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
        self.AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
        
        # Monitoring Configuration
        self.METRICS_PORT = int(os.getenv('METRICS_PORT', 9090))
        self.ENABLE_METRICS = os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
        self.SENTRY_DSN = os.getenv('SENTRY_DSN')  # Optional error tracking
        
        # Rate Limiting
        self.RATE_LIMIT_ENABLED = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
        self.RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.getenv('RATE_LIMIT_REQUESTS_PER_MINUTE', 60))
        self.RATE_LIMIT_BURST = int(os.getenv('RATE_LIMIT_BURST', 10))
        
        # Quality Tiers Pricing (cents per minute)
        self.PRICING = {
            'budget': {
                'price_per_minute': int(os.getenv('BUDGET_PRICE_PER_MINUTE', 50)),  # 50 cents
                'gpu_cost': int(os.getenv('BUDGET_GPU_COST', 15)),  # 15 cents
            },
            'standard': {
                'price_per_minute': int(os.getenv('STANDARD_PRICE_PER_MINUTE', 200)),  # $2.00
                'gpu_cost': int(os.getenv('STANDARD_GPU_COST', 60)),  # 60 cents
            },
            'premium': {
                'price_per_minute': int(os.getenv('PREMIUM_PRICE_PER_MINUTE', 500)),  # $5.00
                'gpu_cost': int(os.getenv('PREMIUM_GPU_COST', 150)),  # $1.50
            },
            'ultra': {
                'price_per_minute': int(os.getenv('ULTRA_PRICE_PER_MINUTE', 1000)),  # $10.00
                'gpu_cost': int(os.getenv('ULTRA_GPU_COST', 300)),  # $3.00
            }
        }
        
        # Webhook Configuration
        self.WEBHOOK_TIMEOUT = int(os.getenv('WEBHOOK_TIMEOUT', 10))
        self.WEBHOOK_RETRY_COUNT = int(os.getenv('WEBHOOK_RETRY_COUNT', 3))
        
        # Performance Tuning
        self.BATCH_SIZE = int(os.getenv('BATCH_SIZE', 4))
        self.MAX_FRAMES_PER_BATCH = int(os.getenv('MAX_FRAMES_PER_BATCH', 8))
        self.ENABLE_MIXED_PRECISION = os.getenv('ENABLE_MIXED_PRECISION', 'true').lower() == 'true'
        self.ENABLE_TORCH_COMPILE = os.getenv('ENABLE_TORCH_COMPILE', 'false').lower() == 'true'
        
        self._setup_logging()
        self._validate_config()
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key if not provided"""
        import secrets
        return secrets.token_urlsafe(32)
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _setup_logging(self):
        """Configure logging based on environment"""
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Reduce noise from external libraries
        logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        
        if self.SENTRY_DSN:
            try:
                import sentry_sdk
                from sentry_sdk.integrations.fastapi import FastApiIntegration
                from sentry_sdk.integrations.logging import LoggingIntegration
                
                sentry_sdk.init(
                    dsn=self.SENTRY_DSN,
                    integrations=[
                        FastApiIntegration(auto_enable=True),
                        LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)
                    ],
                    traces_sample_rate=0.1,
                    environment='production' if not self.DEBUG else 'development'
                )
                logging.info("Sentry error tracking initialized")
            except ImportError:
                logging.warning("Sentry SDK not available")
    
    def _validate_config(self):
        """Validate critical configuration"""
        logger = logging.getLogger(__name__)
        
        # Check GPU availability if CUDA device specified
        if self.DEVICE == 'cuda' and not self._is_cuda_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.DEVICE = 'cpu'
        
        # Validate file size limits
        if self.MAX_FILE_SIZE > 50 * 1024**3:  # 50GB
            logger.warning("Very large file size limit set: {:.1f}GB".format(self.MAX_FILE_SIZE / 1024**3))
        
        # Validate concurrent jobs vs GPU memory
        if self.DEVICE == 'cuda' and self.MAX_CONCURRENT_JOBS > 2:
            logger.warning("High concurrent job count may cause GPU OOM errors")
        
        # Ensure required directories exist
        for path in [self.WORKSPACE_PATH, self.MODEL_CACHE_DIR, self.UPLOAD_DIR]:
            if not path.exists():
                logger.warning(f"Creating missing directory: {path}")
                path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Configuration validated - Device: {self.DEVICE}, Max jobs: {self.MAX_CONCURRENT_JOBS}")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return not self.DEBUG and os.getenv('ENVIRONMENT', '').lower() == 'production'
    
    def get_model_config(self) -> dict:
        """Get model-specific configuration"""
        return {
            'device': self.DEVICE,
            'cache_dir': str(self.MODEL_CACHE_DIR),
            'enable_offload': self.ENABLE_MODEL_OFFLOAD,
            'memory_fraction': self.GPU_MEMORY_FRACTION,
            'batch_size': self.BATCH_SIZE,
            'max_frames_per_batch': self.MAX_FRAMES_PER_BATCH,
            'enable_mixed_precision': self.ENABLE_MIXED_PRECISION,
            'enable_torch_compile': self.ENABLE_TORCH_COMPILE,
        }
    
    def get_database_config(self) -> dict:
        """Get database configuration"""
        return {
            'url': self.DATABASE_URL,
            'pool_size': 5,
            'max_overflow': 10,
            'pool_timeout': 30,
            'pool_recycle': 3600
        }
    
    def get_storage_config(self) -> dict:
        """Get storage backend configuration"""
        config = {'backend': self.STORAGE_BACKEND}
        
        if self.STORAGE_BACKEND == 's3':
            config.update({
                'bucket': self.S3_BUCKET,
                'access_key_id': self.AWS_ACCESS_KEY_ID,
                'secret_access_key': self.AWS_SECRET_ACCESS_KEY,
                'region': self.AWS_REGION
            })
        elif self.STORAGE_BACKEND == 'local':
            config.update({
                'upload_dir': str(self.UPLOAD_DIR),
                'workspace_dir': str(self.WORKSPACE_PATH)
            })
        
        return config
    
    def __repr__(self) -> str:
        """String representation hiding sensitive data"""
        return f"ProductionConfig(device={self.DEVICE}, max_jobs={self.MAX_CONCURRENT_JOBS}, debug={self.DEBUG})"

# Development configuration
class DevelopmentConfig(ProductionConfig):
    """Development configuration with debug settings"""
    
    def __init__(self):
        super().__init__()
        self.DEBUG = True
        self.LOG_LEVEL = 'DEBUG'
        self.MAX_CONCURRENT_JOBS = 1
        self.RATE_LIMIT_ENABLED = False
        self.ENABLE_METRICS = False
        
        # Lower resource requirements for development
        self.MAX_FILE_SIZE = 1 * 1024**3  # 1GB
        self.JOB_TIMEOUT_MINUTES = 30  # 30 minutes
        
        self._setup_logging()

# Test configuration
class TestConfig(ProductionConfig):
    """Test configuration for unit/integration tests"""
    
    def __init__(self):
        super().__init__()
        self.DEBUG = True
        self.LOG_LEVEL = 'WARNING'  # Reduce test noise
        self.MAX_CONCURRENT_JOBS = 1
        self.RATE_LIMIT_ENABLED = False
        self.ENABLE_METRICS = False
        self.DEVICE = 'cpu'  # Always use CPU for tests
        
        # Use temporary directories for tests
        import tempfile
        self.WORKSPACE_PATH = Path(tempfile.mkdtemp())
        self.MODEL_CACHE_DIR = Path(tempfile.mkdtemp())
        self.UPLOAD_DIR = Path(tempfile.mkdtemp())
        
        self._setup_logging()

# Configuration factory
def get_config() -> ProductionConfig:
    """Get configuration based on environment"""
    env = os.getenv('ENVIRONMENT', 'production').lower()
    
    if env == 'development':
        return DevelopmentConfig()
    elif env == 'test':
        return TestConfig()
    else:
        return ProductionConfig()