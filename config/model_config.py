"""
Model configuration and paths for 2025 SOTA video enhancer
"""
import os

class ModelConfig:
    """Configuration class for SOTA model paths and settings"""
    
    # Base paths
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
        """Check which SOTA models are available"""
        status = {
            # ASR Models
            'faster_whisper': True,  # Always available via pip
            
            # 2025 SOTA Models
            'vsrm': os.path.exists(cls.VSRM_CKPT),
            'seedvr2': os.path.exists(cls.SEEDVR2_CKPT),
            'ditvr': os.path.exists(cls.DITVR_CKPT),
            'fast_mamba_vsr': os.path.exists(cls.FAST_MAMBA_VSR_CKPT),
            
            # Fallback Models
            'rvrt': os.path.exists(cls.RVRT_MODEL_PATH),
            'enhanced_rife': os.path.exists(cls.ENHANCED_RIFE_MODEL_PATH),
            'reframing': True,  # YOLO models are downloaded automatically
            
            'device': cls.DEVICE,
            'pipeline_defaults': cls.PIPELINE_DEFAULTS,
            'face_restoration_backend': cls.FACE_RESTORATION_BACKEND,
            
            'paths': {
                'vsrm': cls.VSRM_CKPT,
                'seedvr2': cls.SEEDVR2_CKPT,
                'ditvr': cls.DITVR_CKPT,
                'fast_mamba_vsr': cls.FAST_MAMBA_VSR_CKPT,
                'rvrt': cls.RVRT_MODEL_PATH,
                'enhanced_rife': cls.ENHANCED_RIFE_MODEL_PATH,
                'yolo': cls.YOLO_MODEL_PATH,
                'cache': cls.CACHE_PATH
            }
        }
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
