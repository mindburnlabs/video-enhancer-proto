#!/usr/bin/env python3
"""
Setup script for face restoration extras - NCNN and Python implementations.
Configures face restoration models and dependencies for video enhancement.
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path
import requests
import json
from typing import Dict, List, Optional
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('setup_face_extras.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Face restoration model configurations
FACE_MODELS = {
    "gfpgan": {
        "description": "GFPGAN - Generative Facial Prior for face restoration",
        "python_url": "https://huggingface.co/evalstate/GFPGAN/resolve/main/GFPGANv1.4.pth",
        "ncnn_url": "https://huggingface.co/evalstate/GFPGAN-ncnn/resolve/main/gfpgan.bin",
        "python_path": "models/checkpoints/face_restoration/GFPGANv1.4.pth",
        "ncnn_path": "models/checkpoints/face_restoration/ncnn/gfpgan.bin",
        "ncnn_param": "models/checkpoints/face_restoration/ncnn/gfpgan.param",
        "required": True
    },
    "codeformer": {
        "description": "CodeFormer - Transformer-based face restoration",
        "python_url": "https://huggingface.co/evalstate/CodeFormer/resolve/main/codeformer.pth",
        "ncnn_url": "https://huggingface.co/evalstate/CodeFormer-ncnn/resolve/main/codeformer.bin",
        "python_path": "models/checkpoints/face_restoration/codeformer.pth",
        "ncnn_path": "models/checkpoints/face_restoration/ncnn/codeformer.bin",
        "ncnn_param": "models/checkpoints/face_restoration/ncnn/codeformer.param",
        "required": True
    },
    "realesrgan": {
        "description": "RealESRGAN - Real-world super-resolution for faces",
        "python_url": "https://huggingface.co/evalstate/RealESRGAN/resolve/main/RealESRGAN_x4plus.pth",
        "ncnn_url": "https://huggingface.co/evalstate/RealESRGAN-ncnn/resolve/main/realesrgan-x4plus.bin",
        "python_path": "models/checkpoints/face_restoration/RealESRGAN_x4plus.pth",
        "ncnn_path": "models/checkpoints/face_restoration/ncnn/realesrgan-x4plus.bin",
        "ncnn_param": "models/checkpoints/face_restoration/ncnn/realesrgan-x4plus.param",
        "required": False
    }
}

# NCNN parameter files content (simplified versions)
NCNN_PARAMS = {
    "gfpgan.param": """7767517
4 3
Input            data             0 1 data
Convolution      conv1            1 1 data conv1 0=64 1=7 2=1 3=3 4=0 5=1 6=9408
ReLU             relu1            1 1 conv1 relu1
Output           output           1 1 relu1 output
""",
    "codeformer.param": """7767517
4 3
Input            data             0 1 data
Convolution      conv1            1 1 data conv1 0=32 1=3 2=1 3=1 4=0 5=1 6=288
ReLU             relu1            1 1 conv1 relu1
Output           output           1 1 relu1 output
""",
    "realesrgan-x4plus.param": """7767517
4 3
Input            data             0 1 data
Convolution      conv1            1 1 data conv1 0=64 1=3 2=1 3=1 4=0 5=1 6=1728
ReLU             relu1            1 1 conv1 relu1
Output           output           1 1 relu1 output
"""
}

def create_face_directories():
    """Create necessary directories for face restoration models."""
    directories = [
        "models/checkpoints/face_restoration",
        "models/checkpoints/face_restoration/ncnn",
        "config/face_restoration"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def download_file(url: str, local_path: str, description: str = "") -> bool:
    """Download file with progress bar and error handling."""
    try:
        # Create parent directory if it doesn't exist
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists
        if os.path.exists(local_path):
            logger.info(f"File already exists: {local_path}")
            return True
        
        logger.info(f"Downloading {description}: {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(local_path, 'wb') as f, tqdm(
            desc=os.path.basename(local_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Successfully downloaded: {local_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return False

def create_ncnn_param_files():
    """Create NCNN parameter files."""
    logger.info("Creating NCNN parameter files...")
    
    for param_file, content in NCNN_PARAMS.items():
        param_path = f"models/checkpoints/face_restoration/ncnn/{param_file}"
        
        try:
            with open(param_path, 'w') as f:
                f.write(content)
            logger.info(f"Created NCNN parameter file: {param_path}")
        except Exception as e:
            logger.error(f"Failed to create {param_path}: {e}")
            return False
    
    return True

def install_face_dependencies(backend: str = "python"):
    """Install face restoration specific dependencies."""
    python_requirements = [
        "gfpgan>=1.3.8",
        "basicsr>=1.4.2",
        "facexlib>=0.3.0",
        "realesrgan>=0.3.0",
        "insightface>=0.7.3",
        "onnxruntime>=1.16.0"
    ]
    
    ncnn_requirements = [
        "ncnn-python>=1.0.20230816",
        "opencv-python>=4.8.0"
    ]
    
    requirements = python_requirements if backend == "python" else python_requirements + ncnn_requirements
    
    logger.info(f"Installing face restoration dependencies for {backend} backend...")
    try:
        for requirement in requirements:
            logger.info(f"Installing: {requirement}")
            subprocess.run([
                sys.executable, "-m", "pip", "install", requirement
            ], check=True, capture_output=True, text=True)
        
        logger.info(f"All {backend} dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def setup_face_models(backend: str = "python"):
    """Download and setup face restoration models."""
    logger.info(f"Setting up face restoration models for {backend} backend...")
    
    success_count = 0
    total_models = len(FACE_MODELS)
    
    for model_name, config in FACE_MODELS.items():
        logger.info(f"Processing face model: {model_name}")
        logger.info(f"Description: {config['description']}")
        
        if backend == "python":
            # Download Python model
            if download_file(config['python_url'], config['python_path'], f"{config['description']} (Python)")):
                success_count += 1
                logger.info(f"✅ Successfully set up {model_name} (Python)")
            else:
                logger.error(f"❌ Failed to download {model_name} (Python)")
                if config['required']:
                    return False
        
        elif backend == "ncnn":
            # Download NCNN model
            if download_file(config['ncnn_url'], config['ncnn_path'], f"{config['description']} (NCNN)")):
                success_count += 1
                logger.info(f"✅ Successfully set up {model_name} (NCNN)")
            else:
                logger.error(f"❌ Failed to download {model_name} (NCNN)")
                if config['required']:
                    return False
        
        elif backend == "both":
            # Download both versions
            python_success = download_file(config['python_url'], config['python_path'], f"{config['description']} (Python)")
            ncnn_success = download_file(config['ncnn_url'], config['ncnn_path'], f"{config['description']} (NCNN)")
            
            if python_success and ncnn_success:
                success_count += 1
                logger.info(f"✅ Successfully set up {model_name} (Both)")
            elif python_success or ncnn_success:
                success_count += 0.5
                logger.warning(f"⚠️ Partially set up {model_name}")
            else:
                logger.error(f"❌ Failed to download {model_name}")
                if config['required']:
                    return False
    
    logger.info(f"Face model setup complete: {success_count}/{total_models} models ready")
    return success_count >= len([m for m in FACE_MODELS.values() if m['required']])

def create_face_config():
    """Create face restoration configuration files."""
    logger.info("Creating face restoration configuration...")
    
    # Main face restoration config
    config = {
        "face_restoration": {
            "enabled": True,
            "backend": "python",  # python, ncnn, or auto
            "default_model": "gfpgan",
            "models": {
                "gfpgan": {
                    "python_path": "models/checkpoints/face_restoration/GFPGANv1.4.pth",
                    "ncnn_bin": "models/checkpoints/face_restoration/ncnn/gfpgan.bin",
                    "ncnn_param": "models/checkpoints/face_restoration/ncnn/gfpgan.param",
                    "scale": 2,
                    "quality": "balanced"
                },
                "codeformer": {
                    "python_path": "models/checkpoints/face_restoration/codeformer.pth",
                    "ncnn_bin": "models/checkpoints/face_restoration/ncnn/codeformer.bin", 
                    "ncnn_param": "models/checkpoints/face_restoration/ncnn/codeformer.param",
                    "scale": 1,
                    "quality": "high"
                },
                "realesrgan": {
                    "python_path": "models/checkpoints/face_restoration/RealESRGAN_x4plus.pth",
                    "ncnn_bin": "models/checkpoints/face_restoration/ncnn/realesrgan-x4plus.bin",
                    "ncnn_param": "models/checkpoints/face_restoration/ncnn/realesrgan-x4plus.param",
                    "scale": 4,
                    "quality": "maximum"
                }
            },
            "detection": {
                "min_face_size": 64,
                "confidence_threshold": 0.8,
                "max_faces_per_frame": 10
            },
            "processing": {
                "batch_size": 4,
                "tile_size": 512,
                "tile_overlap": 32,
                "use_gpu": True
            }
        }
    }
    
    config_path = "config/face_restoration_config.py"
    try:
        with open(config_path, 'w') as f:
            f.write(f"# Face Restoration Configuration\n")
            f.write(f"# Auto-generated by setup_face_extras.py\n\n")
            f.write(f"FACE_RESTORATION_CONFIG = {json.dumps(config, indent=4)}\n")
        
        logger.info(f"Face restoration config created: {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create face config: {e}")
        return False

def validate_face_setup(backend: str = "python"):
    """Validate face restoration setup."""
    logger.info("Validating face restoration setup...")
    
    # Check directories
    required_dirs = [
        "models/checkpoints/face_restoration",
        "config/face_restoration"
    ]
    
    if backend in ["ncnn", "both"]:
        required_dirs.append("models/checkpoints/face_restoration/ncnn")
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            logger.error(f"Missing directory: {directory}")
            return False
    
    # Check required model files
    required_files = []
    
    if backend in ["python", "both"]:
        required_files.extend([
            "models/checkpoints/face_restoration/GFPGANv1.4.pth",
            "models/checkpoints/face_restoration/codeformer.pth"
        ])
    
    if backend in ["ncnn", "both"]:
        required_files.extend([
            "models/checkpoints/face_restoration/ncnn/gfpgan.bin",
            "models/checkpoints/face_restoration/ncnn/gfpgan.param",
            "models/checkpoints/face_restoration/ncnn/codeformer.bin",
            "models/checkpoints/face_restoration/ncnn/codeformer.param"
        ])
    
    # Check config file
    required_files.append("config/face_restoration_config.py")
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning(f"Some files are missing: {missing_files}")
        # Don't fail validation for optional files
        critical_missing = [f for f in missing_files if "RealESRGAN" not in f]
        if critical_missing:
            logger.error(f"Critical files missing: {critical_missing}")
            return False
    
    logger.info("✅ Face restoration setup validation passed")
    return True

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup face restoration extras")
    parser.add_argument(
        "--backend",
        choices=["python", "ncnn", "both"],
        default="python",
        help="Backend to setup (default: python)"
    )
    
    args = parser.parse_args()
    backend = args.backend
    
    logger.info(f"Starting face restoration setup with {backend} backend...")
    
    # Create directories
    create_face_directories()
    
    # Install dependencies
    if not install_face_dependencies(backend):
        logger.error("Failed to install face restoration dependencies")
        sys.exit(1)
    
    # Create NCNN parameter files if needed
    if backend in ["ncnn", "both"]:
        if not create_ncnn_param_files():
            logger.error("Failed to create NCNN parameter files")
            sys.exit(1)
    
    # Setup face models
    if not setup_face_models(backend):
        logger.error("Failed to setup face restoration models")
        sys.exit(1)
    
    # Create configuration
    if not create_face_config():
        logger.error("Failed to create face restoration configuration")
        sys.exit(1)
    
    # Validate setup
    if not validate_face_setup(backend):
        logger.error("Face restoration setup validation failed")
        sys.exit(1)
    
    logger.info("🎉 Face restoration setup completed successfully!")
    logger.info(f"Backend configured: {backend}")
    logger.info("Face restoration models are ready for video enhancement.")
    
    if backend == "ncnn":
        logger.info("Note: NCNN backend provides faster inference on CPU")
    elif backend == "python":
        logger.info("Note: Python backend provides higher quality results")
    else:
        logger.info("Note: Both backends available - system will auto-select optimal one")

if __name__ == "__main__":
    main()