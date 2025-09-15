#!/usr/bin/env python3
"""
Setup script for SOTA video enhancement models - "Topaz Killer" reference.
Downloads and prepares all necessary models for production deployment.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import requests
import hashlib
from typing import Dict, List, Optional
import json
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('setup_topaz_killer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model configurations
SOTA_MODELS = {
    "vsrm": {
        "description": "Video Super-Resolution Model (VSRM) - SOTA enhancement",
        "url": "https://huggingface.co/evalstate/vsrm-large/resolve/main/vsrm_large.pth",
        "local_path": "models/checkpoints/vsrm_large.pth",
        "sha256": "",  # Will be filled after download
        "required": True
    },
    "seedvr2": {
        "description": "SeedVR2 - Diffusion-based video restoration",
        "url": "https://huggingface.co/evalstate/seedvr2-base/resolve/main/seedvr2_base.safetensors", 
        "local_path": "models/checkpoints/seedvr2_base.safetensors",
        "sha256": "",
        "required": True
    },
    "ditvr": {
        "description": "DiTVR - Transformer-based video restoration",
        "url": "https://huggingface.co/evalstate/ditvr-base/resolve/main/ditvr_base.pth",
        "local_path": "models/checkpoints/ditvr_base.pth", 
        "sha256": "",
        "required": True
    },
    "fast_mamba_vsr": {
        "description": "Fast Mamba VSR - State-space model for video super-resolution",
        "url": "https://huggingface.co/evalstate/fast-mamba-vsr/resolve/main/fast_mamba_vsr.pth",
        "local_path": "models/checkpoints/fast_mamba_vsr.pth",
        "sha256": "",
        "required": True
    },
    "rife": {
        "description": "RIFE - Real-time intermediate flow estimation for interpolation",
        "url": "https://huggingface.co/evalstate/RIFE/resolve/main/flownet.pkl",
        "local_path": "models/interpolation/RIFE/flownet.pkl",
        "sha256": "",
        "required": True
    }
}

def create_directories():
    """Create necessary directories for models and checkpoints."""
    directories = [
        "models/checkpoints",
        "models/interpolation/RIFE",
        "data/temp",
        "logs"
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

def verify_file_integrity(file_path: str, expected_sha256: str = "") -> bool:
    """Verify file integrity using SHA256 hash."""
    if not expected_sha256:
        logger.info(f"No checksum provided for {file_path}, skipping verification")
        return True
    
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        calculated_hash = sha256_hash.hexdigest()
        if calculated_hash == expected_sha256:
            logger.info(f"File integrity verified: {file_path}")
            return True
        else:
            logger.error(f"File integrity check failed for {file_path}")
            logger.error(f"Expected: {expected_sha256}")
            logger.error(f"Got: {calculated_hash}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to verify file integrity for {file_path}: {e}")
        return False

def install_python_dependencies():
    """Install required Python dependencies."""
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "gradio>=4.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
        "safetensors>=0.4.0",
        "transformers>=4.35.0",
        "diffusers>=0.25.0",
        "accelerate>=0.25.0",
        "xformers>=0.0.22",
        "einops>=0.7.0",
        "timm>=0.9.10"
    ]
    
    logger.info("Installing Python dependencies...")
    try:
        for requirement in requirements:
            logger.info(f"Installing: {requirement}")
            subprocess.run([
                sys.executable, "-m", "pip", "install", requirement
            ], check=True, capture_output=True, text=True)
        
        logger.info("All Python dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def setup_models():
    """Download and setup all SOTA models."""
    logger.info("Setting up SOTA models...")
    
    success_count = 0
    total_models = len(SOTA_MODELS)
    
    for model_name, config in SOTA_MODELS.items():
        logger.info(f"Processing model: {model_name}")
        logger.info(f"Description: {config['description']}")
        
        # Download model
        if download_file(config['url'], config['local_path'], config['description']):
            # Verify integrity if checksum provided
            if verify_file_integrity(config['local_path'], config['sha256']):
                success_count += 1
                logger.info(f"✅ Successfully set up {model_name}")
            else:
                logger.error(f"❌ Failed integrity check for {model_name}")
                if config['required']:
                    return False
        else:
            logger.error(f"❌ Failed to download {model_name}")
            if config['required']:
                return False
    
    logger.info(f"Model setup complete: {success_count}/{total_models} models ready")
    return success_count == total_models

def create_model_registry():
    """Create a model registry file for the application."""
    registry = {
        "version": "1.0.0",
        "created_at": "",
        "models": {}
    }
    
    for model_name, config in SOTA_MODELS.items():
        if os.path.exists(config['local_path']):
            registry["models"][model_name] = {
                "path": config['local_path'],
                "description": config['description'],
                "status": "ready",
                "file_size": os.path.getsize(config['local_path'])
            }
        else:
            registry["models"][model_name] = {
                "path": config['local_path'],
                "description": config['description'],
                "status": "missing",
                "file_size": 0
            }
    
    registry_path = "models/model_registry.json"
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    logger.info(f"Model registry created: {registry_path}")

def validate_setup():
    """Validate that all components are properly set up."""
    logger.info("Validating setup...")
    
    # Check directories
    required_dirs = ["models/checkpoints", "models/interpolation/RIFE", "data/temp", "logs"]
    for directory in required_dirs:
        if not os.path.exists(directory):
            logger.error(f"Missing directory: {directory}")
            return False
    
    # Check critical files
    critical_files = [
        "models/checkpoints/vsrm_large.pth",
        "models/checkpoints/ditvr_base.pth",
        "models/interpolation/RIFE/flownet.pkl"
    ]
    
    missing_files = []
    for file_path in critical_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"Missing critical files: {missing_files}")
        return False
    
    logger.info("✅ Setup validation passed")
    return True

def main():
    """Main setup function."""
    logger.info("Starting SOTA Video Enhancer setup (Topaz Killer)...")
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_python_dependencies():
        logger.error("Failed to install Python dependencies")
        sys.exit(1)
    
    # Setup models
    if not setup_models():
        logger.error("Failed to setup models")
        sys.exit(1)
    
    # Create model registry
    create_model_registry()
    
    # Validate setup
    if not validate_setup():
        logger.error("Setup validation failed")
        sys.exit(1)
    
    logger.info("🎉 Setup completed successfully!")
    logger.info("The SOTA Video Enhancer is ready for deployment.")
    logger.info("Next steps:")
    logger.info("1. Run: python setup_face_extras.py (for face restoration)")
    logger.info("2. Run: python app.py (to start the application)")

if __name__ == "__main__":
    main()