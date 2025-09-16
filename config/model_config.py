"""
Model configuration and paths for 2025 SOTA video enhancer
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
import hashlib

class ModelRegistry:
    """Manages model registry, signatures, and validation."""

    def __init__(self, registry_path="config/model_registry.json"):
        self.registry_path = registry_path
        self.registry = self._load_registry()

    def _load_registry(self):
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {"models": {}}

    def _save_registry(self):
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=4)

    def add_model(self, model_id, path, signature):
        self.registry["models"][model_id] = {"path": path, "signature": signature}
        self._save_registry()

    def get_model_info(self, model_id):
        return self.registry["models"].get(model_id)

    def validate_model(self, model_id):
        model_info = self.get_model_info(model_id)
        if not model_info:
            return False, "Model not found in registry"

        if not os.path.exists(model_info["path"]):
            return False, "Model file not found"

        with open(model_info["path"], "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        if file_hash != model_info["signature"]:
            return False, "Invalid model signature"

        return True, "Model validated successfully"

class ModelConfig:
    """Configuration class for SOTA model paths and settings"""
    
        self.model_registry = ModelRegistry()
    BASE_MODEL_PATH = os.getenv('BASE_MODEL_PATH', '/app/models')
    CACHE_PATH = os.getenv('HUGGINGFACE_HUB_CACHE', '/app/cache')
    
    # ASR Models
    FASTER_WHISPER_MODEL_SIZE = os.getenv('FASTER_WHISPER_MODEL_SIZE', 'large-v3')
    FASTER_WHISPER_COMPUTE_TYPE = os.getenv('FASTER_WHISPER_COMPUTE_TYPE', 'float16')
    
    # 2025 SOTA Video Enhancement Models
    VSRM_CKPT = os.getenv('VSRM_CKPT', os.path.join(BASE_MODEL_PATH, 'vsrm/vsrm_mamba.pth'))
    SEEDVR2_CKPT = os.getenv('SEEDVR2_CKPT', os.path.join(BASE_MODEL_PATH, 'seedvr2/seedvr2_diffusion.pth'))
    DITVR_CKPT = os.getenv('DITVR_CKPT', os.path.join(BASE_MODEL_PATH, 'ditvr/ditvr_transformer.pth'))
    FAST_MAMBA_VSR_CKPT = os.getenv('FAST_MAMBA_VSR_CKPT', os.path.join(BASE_MODEL_PATH, 'fast_mamba_vsr/fast_mamba_vsr.pth'))
    
    # Fallback Models  
    RVRT_MODEL_PATH = os.getenv('RVRT_MODEL_PATH', os.path.join(BASE_MODEL_PATH, 'rvrt_model.pth'))
    
    # Frame Interpolation Models
    ENHANCED_RIFE_MODEL_PATH = os.getenv('ENHANCED_RIFE_MODEL_PATH', os.path.join(BASE_MODEL_PATH, 'enhanced_rife_v2.pth'))
    RIFE_REPO_PATH = os.getenv('RIFE_REPO_PATH', '/app/models/interpolation/RIFE')
    
    # Reframing Models
    YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'yolov8n.pt')  # Can be model name or path
    
    # Processing settings
    DEVICE = "cuda" if os.getenv('FORCE_CPU', '').lower() != 'true' else "cpu"
    
    # Pipeline defaults
    PIPELINE_DEFAULTS = {
        'allow_diffusion': os.getenv('ALLOW_DIFFUSION', 'true').lower() == 'true',
        'allow_zero_shot': os.getenv('ALLOW_ZERO_SHOT', 'true').lower() == 'true',
        'latency_class': os.getenv('LATENCY_CLASS', 'standard'),  # strict, standard, flexible
        'preferred_backbone': os.getenv('PREFERRED_BACKBONE', 'eamamba'),  # eamamba, mambairv2
    }
    
    # Face restoration backend
    FACE_RESTORATION_BACKEND = os.getenv('FACE_RESTORATION_BACKEND', 'ncnn')  # ncnn, python, off
    
    # Video processing settings
    DEFAULT_CLIP_LENGTH = int(os.getenv('VIDEO_CLIP_LENGTH', '7'))
    MAX_VIDEO_DURATION = int(os.getenv('MAX_VIDEO_DURATION', '300'))  # 5 minutes
    
    @classmethod
    def get_model_status(cls):
        status = {}
        for model_id, model_info in cls.model_registry.registry["models"].items():
            is_valid, reason = cls.model_registry.validate_model(model_id)
            status[model_id] = {"path": model_info["path"], "valid": is_valid, "reason": reason}
        return status
    
    @classmethod
    def validate_setup(cls):
        """Validate the current SOTA setup and return any issues"""
        issues = []
        
        # Check cache directory
        if not os.path.exists(cls.CACHE_PATH):
            try:
                os.makedirs(cls.CACHE_PATH, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create cache directory: {e}")
        
        # Validate SOTA model paths (warnings, not errors for now)
        sota_models = {
            'VSRM': cls.VSRM_CKPT,
            'SeedVR2': cls.SEEDVR2_CKPT,
            'DiTVR': cls.DITVR_CKPT,
            'Fast Mamba VSR': cls.FAST_MAMBA_VSR_CKPT,
        }
        
        missing_sota = []
        for name, path in sota_models.items():
            if not os.path.exists(path):
                missing_sota.append(f"{name} model not found at {path}")
        
        if missing_sota:
            issues.append(f"SOTA models not found (will use fallbacks): {', '.join(missing_sota)}")
        
        # Check fallback models
        if not os.path.exists(cls.RVRT_MODEL_PATH):
            issues.append(f"Fallback RVRT model not found at {cls.RVRT_MODEL_PATH}")
            
        if not os.path.exists(cls.RIFE_REPO_PATH):
            issues.append(f"RIFE repository not found at {cls.RIFE_REPO_PATH}")
            
        # Validate pipeline configuration
        if cls.PIPELINE_DEFAULTS['latency_class'] not in ['strict', 'standard', 'flexible']:
            issues.append(f"Invalid latency_class: {cls.PIPELINE_DEFAULTS['latency_class']}")
            
        if cls.FACE_RESTORATION_BACKEND not in ['ncnn', 'python', 'off']:
            issues.append(f"Invalid face restoration backend: {cls.FACE_RESTORATION_BACKEND}")
            
        return issues
