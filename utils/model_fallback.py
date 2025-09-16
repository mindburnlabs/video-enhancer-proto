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

import logging
import time
import torch
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import gc

from utils.error_handler import (
    error_handler, ModelError, ErrorCode, handle_exceptions
)

logger = logging.getLogger(__name__)

class ModelPriority(Enum):
    """Model priority levels for fallback selection"""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    FALLBACK = "fallback"
    EMERGENCY = "emergency"

@dataclass
class ModelConfig:
    """Configuration for a model with fallback metadata"""
    name: str
    model_class: type
    priority: ModelPriority
    device_requirements: List[str]  # ['cuda', 'cpu'] etc.
    memory_requirement_gb: float
    load_timeout_seconds: int = 300
    retry_attempts: int = 3
    initialization_args: Optional[Dict[str, Any]] = None
    fallback_reason: Optional[str] = None

@dataclass
class ModelLoadResult:
    """Result of model loading attempt"""
    success: bool
    model: Optional[Any] = None
    config: Optional[ModelConfig] = None
    load_time: float = 0.0
    error: Optional[Exception] = None
    fallback_used: bool = False
    memory_used_mb: float = 0.0

class ModelFallbackManager:
    """Manages model loading with intelligent fallbacks"""
    
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.model_configs: Dict[str, List[ModelConfig]] = {}
        self.load_history: List[Dict[str, Any]] = []
        self.max_history = 100
    
    def register_model_hierarchy(
        self, 
        model_type: str, 
        configs: List[ModelConfig]
    ):
        """Register a hierarchy of models for a given type"""
        # Sort by priority
        priority_order = [
            ModelPriority.PRIMARY,
            ModelPriority.SECONDARY, 
            ModelPriority.FALLBACK,
            ModelPriority.EMERGENCY
        ]
        
        sorted_configs = sorted(
            configs,
            key=lambda x: priority_order.index(x.priority)
        )
        
        self.model_configs[model_type] = sorted_configs
        logger.info(f"Registered {len(sorted_configs)} model configs for type '{model_type}'")
    
    @handle_exceptions(
        component="model_fallback",
        operation="load_model",
        retry_possible=True,
        fallback_available=True
    )
    def load_model_with_fallbacks(
        self,
        model_type: str,
        device: str = "auto",
        force_reload: bool = False
    ) -> ModelLoadResult:
        """Load a model with automatic fallbacks"""
        
        logger.info(f"Loading model type '{model_type}' with fallbacks")
        
        # Check if model is already loaded
        if not force_reload and model_type in self.loaded_models:
            model = self.loaded_models[model_type]
            if model is not None:
                logger.info(f"Using cached model for type '{model_type}'")
                return ModelLoadResult(
                    success=True,
                    model=model,
                    config=self._get_loaded_config(model_type),
                    fallback_used=False
                )
        
        # Get model configurations for this type
        if model_type not in self.model_configs:
            raise ModelError(
                message=f"No model configurations registered for type '{model_type}'",
                error_code=ErrorCode.MODEL_NOT_FOUND
            )
        
        configs = self.model_configs[model_type]
        auto_device = self._determine_best_device() if device == "auto" else device
        
        # Try each configuration in priority order
        for i, config in enumerate(configs):
            logger.info(f"Attempting to load {config.name} (priority: {config.priority.value})")
            
            # Check device compatibility
            if auto_device not in config.device_requirements:
                logger.warning(f"Skipping {config.name}: device '{auto_device}' not supported")
                continue
            
            # Check memory requirements
            if not self._check_memory_availability(config, auto_device):
                logger.warning(f"Skipping {config.name}: insufficient memory")
                continue
            
            # Attempt to load the model
            result = self._load_single_model(config, auto_device)
            
            if result.success:
                self.loaded_models[model_type] = result.model
                self._record_load_history(model_type, config, result, fallback_used=(i > 0))
                
                if i > 0:
                    logger.warning(f"Loaded fallback model {config.name} for type '{model_type}'")
                    result.fallback_used = True
                else:
                    logger.info(f"Successfully loaded primary model {config.name}")
                
                return result
            else:
                logger.error(f"Failed to load {config.name}: {result.error}")
        
        # All models failed to load
        error_msg = f"All model loading attempts failed for type '{model_type}'"
        raise ModelError(
            message=error_msg,
            error_code=ErrorCode.MODEL_LOAD_ERROR,
            original_error=result.error if 'result' in locals() else None
        )
    
    def _load_single_model(
        self,
        config: ModelConfig,
        device: str
    ) -> ModelLoadResult:
        """Load a single model configuration"""
        
        start_time = time.time()
        
        try:
            # Clear GPU cache if using CUDA
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
                gc.collect()
            
            # Initialize model with timeout
            init_args = config.initialization_args or {}
            init_args['device'] = device
            
            logger.debug(f"Initializing {config.name} with args: {init_args}")
            
            # Use timeout for model loading
            model = self._load_with_timeout(
                config.model_class,
                init_args,
                config.load_timeout_seconds
            )
            
            load_time = time.time() - start_time
            memory_used = self._get_memory_usage(device)
            
            logger.info(f"Successfully loaded {config.name} in {load_time:.2f}s")
            
            return ModelLoadResult(
                success=True,
                model=model,
                config=config,
                load_time=load_time,
                memory_used_mb=memory_used
            )
            
        except TimeoutError as e:
            load_time = time.time() - start_time
            logger.error(f"Timeout loading {config.name} after {load_time:.2f}s")
            return ModelLoadResult(
                success=False,
                config=config,
                load_time=load_time,
                error=e
            )
            
        except Exception as e:
            load_time = time.time() - start_time
            logger.error(f"Error loading {config.name}: {str(e)}")
            return ModelLoadResult(
                success=False,
                config=config,
                load_time=load_time,
                error=e
            )
    
    def _load_with_timeout(
        self,
        model_class: type,
        init_args: Dict[str, Any],
        timeout_seconds: int
    ):
        """Load model with timeout (simplified implementation)"""
        # In a real implementation, you might use multiprocessing or threading
        # For now, we'll just load normally and trust the timeout handling in the calling code
        return model_class(**init_args)
    
    def _determine_best_device(self) -> str:
        """Determine the best device for model loading"""
        if torch.cuda.is_available():
            # Choose GPU with most free memory
            best_gpu = 0
            max_memory = 0
            
            for i in range(torch.cuda.device_count()):
                memory_free = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                if memory_free > max_memory:
                    max_memory = memory_free
                    best_gpu = i
            
            return f"cuda:{best_gpu}"
        else:
            return "cpu"
    
    def _check_memory_availability(
        self,
        config: ModelConfig,
        device: str
    ) -> bool:
        """Check if enough memory is available for the model"""
        required_bytes = config.memory_requirement_gb * 1024 * 1024 * 1024
        
        if device.startswith('cuda'):
            device_idx = int(device.split(':')[1]) if ':' in device else 0
            try:
                props = torch.cuda.get_device_properties(device_idx)
                available = props.total_memory - torch.cuda.memory_allocated(device_idx)
                return available > required_bytes * 1.2  # 20% safety margin
            except:
                return False
        else:
            # For CPU, we'll assume memory is available (simplified check)
            import psutil
            available = psutil.virtual_memory().available
            return available > required_bytes * 1.5  # 50% safety margin for CPU
    
    def _get_memory_usage(self, device: str) -> float:
        """Get current memory usage in MB"""
        if device.startswith('cuda'):
            device_idx = int(device.split(':')[1]) if ':' in device else 0
            return torch.cuda.memory_allocated(device_idx) / (1024 * 1024)
        else:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
    
    def _get_loaded_config(self, model_type: str) -> Optional[ModelConfig]:
        """Get the config for currently loaded model"""
        for entry in reversed(self.load_history):
            if entry['model_type'] == model_type and entry['success']:
                return entry['config']
        return None
    
    def _record_load_history(
        self,
        model_type: str,
        config: ModelConfig,
        result: ModelLoadResult,
        fallback_used: bool
    ):
        """Record model loading attempt in history"""
        entry = {
            'timestamp': time.time(),
            'model_type': model_type,
            'config': config,
            'success': result.success,
            'load_time': result.load_time,
            'memory_used_mb': result.memory_used_mb,
            'fallback_used': fallback_used,
            'error': str(result.error) if result.error else None
        }
        
        self.load_history.append(entry)
        
        # Keep only recent history
        if len(self.load_history) > self.max_history:
            self.load_history.pop(0)
    
    def get_model(self, model_type: str) -> Optional[Any]:
        """Get a loaded model by type"""
        return self.loaded_models.get(model_type)
    
    def unload_model(self, model_type: str):
        """Unload a model and free memory"""
        if model_type in self.loaded_models:
            model = self.loaded_models[model_type]
            
            # Try to move model to CPU first
            if hasattr(model, 'to'):
                try:
                    model.to('cpu')
                except:
                    pass
            
            # Remove from cache
            del self.loaded_models[model_type]
            del model
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gc.collect()
            logger.info(f"Unloaded model type '{model_type}'")
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get model loading statistics"""
        if not self.load_history:
            return {"total_attempts": 0}
        
        total_attempts = len(self.load_history)
        successful_loads = sum(1 for entry in self.load_history if entry['success'])
        fallback_uses = sum(1 for entry in self.load_history if entry.get('fallback_used', False))
        
        avg_load_time = sum(entry['load_time'] for entry in self.load_history) / total_attempts
        
        return {
            "total_attempts": total_attempts,
            "successful_loads": successful_loads,
            "success_rate": successful_loads / total_attempts * 100,
            "fallback_uses": fallback_uses,
            "fallback_rate": fallback_uses / total_attempts * 100,
            "avg_load_time_seconds": avg_load_time,
            "currently_loaded": list(self.loaded_models.keys())
        }

# Global instance
model_fallback_manager = ModelFallbackManager()

# Helper function for easy model loading
def load_model_with_fallbacks(
    model_type: str,
    device: str = "auto",
    force_reload: bool = False
) -> ModelLoadResult:
    """Convenient function to load models with fallbacks"""
    return model_fallback_manager.load_model_with_fallbacks(
        model_type=model_type,
        device=device,
        force_reload=force_reload
    )