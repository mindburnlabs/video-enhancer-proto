#!/usr/bin/env python3
"""
Model Validation Script
Checks all SOTA models, weights, and dependencies are properly configured.
"""

import os
import sys
import logging
from pathlib import Path
import traceback
import json
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelValidator:
    """Validates all models and their configurations."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results = {
            "models_checked": [],
            "weights_found": [],
            "weights_missing": [],
            "dependencies_ok": [],
            "dependencies_missing": [],
            "errors": []
        }
        
    def validate_all(self):
        """Run comprehensive model validation."""
        logger.info("🔍 Starting comprehensive model validation...")
        
        # Check basic dependencies
        self._validate_dependencies()
        
        # Check model structures
        self._validate_model_structures()
        
        # Check model weights
        self._validate_model_weights()
        
        # Check SOTA model imports
        self._validate_sota_imports()
        
        # Generate report
        self._generate_report()
        
    def _validate_dependencies(self):
        """Validate core dependencies."""
        logger.info("📦 Validating core dependencies...")
        
        core_deps = [
            ("torch", "PyTorch"),
            ("cv2", "OpenCV"),
            ("numpy", "NumPy"),
            ("PIL", "Pillow"),
            ("gradio", "Gradio"),
        ]
        
        for module, name in core_deps:
            try:
                __import__(module)
                self.results["dependencies_ok"].append(name)
                logger.info(f"✅ {name}: OK")
            except ImportError as e:
                self.results["dependencies_missing"].append({"name": name, "error": str(e)})
                logger.error(f"❌ {name}: Missing - {e}")
                
        # Check optional dependencies
        optional_deps = [
            ("transformers", "HuggingFace Transformers"),
            ("diffusers", "HuggingFace Diffusers"),
            ("psutil", "Process utilities"),
        ]
        
        for module, name in optional_deps:
            try:
                __import__(module)
                self.results["dependencies_ok"].append(name)
                logger.info(f"✅ {name}: OK")
            except ImportError:
                logger.warning(f"⚠️ {name}: Optional dependency missing")
                
    def _validate_model_structures(self):
        """Validate model directory structures."""
        logger.info("🏗️ Validating model structures...")
        
        model_paths = [
            "models/analysis",
            "models/backbones/diffusion",
            "models/backbones/mamba", 
            "models/backbones/rvrt",
            "models/backbones/transformer",
            "models/enhancement/zeroshot",
            "models/enhancement/vsr",
            "models/enhancement/frame",
            "models/interpolation",
        ]
        
        for model_path in model_paths:
            full_path = self.project_root / model_path
            if full_path.exists():
                logger.info(f"✅ {model_path}: Structure OK")
                self.results["models_checked"].append(model_path)
                
                # Check for Python files
                py_files = list(full_path.glob("*.py"))
                if py_files:
                    logger.info(f"   📄 Found {len(py_files)} Python files")
                else:
                    logger.warning(f"   ⚠️ No Python files found in {model_path}")
            else:
                logger.error(f"❌ {model_path}: Missing")
                self.results["errors"].append(f"Missing model directory: {model_path}")
                
    def _validate_model_weights(self):
        """Validate model weight files."""
        logger.info("⚖️ Validating model weights...")
        
        # Common weight file patterns
        weight_patterns = [
            "**/*.pth",
            "**/*.pt",
            "**/*.bin", 
            "**/*.safetensors",
            "**/*.ckpt",
        ]
        
        weights_dir = self.project_root / "models" / "weights"
        if weights_dir.exists():
            logger.info(f"📁 Checking weights directory: {weights_dir}")
            
            for pattern in weight_patterns:
                weight_files = list(weights_dir.glob(pattern))
                for weight_file in weight_files:
                    if weight_file.stat().st_size > 0:
                        self.results["weights_found"].append(str(weight_file.relative_to(self.project_root)))
                        logger.info(f"✅ Weight file: {weight_file.name} ({weight_file.stat().st_size / 1024 / 1024:.1f}MB)")
                    else:
                        self.results["weights_missing"].append(str(weight_file.relative_to(self.project_root)))
                        logger.error(f"❌ Empty weight file: {weight_file.name}")
        else:
            logger.warning("⚠️ No weights directory found")
            
        # Check for specific model weights
        expected_weights = [
            "models/weights/SeedVR2-3B/pytorch_model.bin",
            "models/weights/DiTVR/model.pth",
            "models/weights/FastMambaVSR/checkpoint.pth",
            "models/weights/VSRM/model.safetensors",
        ]
        
        for weight_path in expected_weights:
            full_path = self.project_root / weight_path
            if full_path.exists() and full_path.stat().st_size > 0:
                self.results["weights_found"].append(weight_path)
                logger.info(f"✅ Expected weight: {weight_path}")
            else:
                self.results["weights_missing"].append(weight_path)
                logger.warning(f"⚠️ Missing expected weight: {weight_path}")
                
    def _validate_sota_imports(self):
        """Validate SOTA model imports."""
        logger.info("🚀 Validating SOTA model imports...")
        
        sota_modules = [
            ("models.analysis.degradation_router", "DegradationRouter"),
            ("models.enhancement.zeroshot.seedvr2_handler", "SeedVR2Handler"),
            ("models.enhancement.zeroshot.ditvr_handler", "DiTVRHandler"),
            ("models.enhancement.vsr.vsrm_handler", "VSRMHandler"),
            ("models.enhancement.vsr.fast_mamba_vsr_handler", "FastMambaVSRHandler"),
            ("models.interpolation.enhanced_rife_handler", "EnhancedRIFEHandler"),
        ]
        
        # Add project root to path
        sys.path.insert(0, str(self.project_root))
        
        for module_path, class_name in sota_modules:
            try:
                module = __import__(module_path, fromlist=[class_name])
                model_class = getattr(module, class_name)
                logger.info(f"✅ {class_name}: Import OK")
                self.results["models_checked"].append(f"{module_path}.{class_name}")
                
                # Try to instantiate (with CPU to avoid CUDA issues)
                try:
                    instance = model_class(device='cpu')
                    logger.info(f"   ✅ Instantiation OK")
                except Exception as inst_error:
                    logger.warning(f"   ⚠️ Instantiation failed: {inst_error}")
                    
            except ImportError as e:
                logger.error(f"❌ {class_name}: Import failed - {e}")
                self.results["errors"].append(f"Import failed: {module_path}.{class_name} - {e}")
            except Exception as e:
                logger.error(f"❌ {class_name}: Unexpected error - {e}")
                self.results["errors"].append(f"Unexpected error: {module_path}.{class_name} - {e}")
                
    def _validate_configuration_files(self):
        """Validate configuration files."""
        logger.info("⚙️ Validating configuration files...")
        
        config_files = [
            "config/model_config.py",
            "config/model_registry.json",
            "config/logging_config.py",
            "config/production_config.py",
            "requirements.txt",
        ]
        
        for config_file in config_files:
            full_path = self.project_root / config_file
            if full_path.exists():
                logger.info(f"✅ Config file: {config_file}")
                
                # Validate JSON files
                if config_file.endswith('.json'):
                    try:
                        with open(full_path, 'r') as f:
                            json.load(f)
                        logger.info(f"   ✅ JSON syntax valid")
                    except json.JSONDecodeError as e:
                        logger.error(f"   ❌ Invalid JSON: {e}")
                        self.results["errors"].append(f"Invalid JSON in {config_file}: {e}")
            else:
                logger.warning(f"⚠️ Missing config file: {config_file}")
                
    def _generate_report(self):
        """Generate comprehensive validation report."""
        total_models = len(self.results["models_checked"])
        found_weights = len(self.results["weights_found"])
        missing_weights = len(self.results["weights_missing"])
        ok_deps = len(self.results["dependencies_ok"])
        errors = len(self.results["errors"])
        
        print("\n" + "="*80)
        print("🔍 MODEL VALIDATION REPORT")
        print("="*80)
        print(f"📦 Dependencies OK: {ok_deps}")
        print(f"🏗️ Models Checked: {total_models}")
        print(f"✅ Weights Found: {found_weights}")
        print(f"❌ Weights Missing: {missing_weights}")
        print(f"🚨 Errors: {errors}")
        print()
        
        if self.results["weights_found"]:
            print("✅ FOUND WEIGHTS:")
            for weight in self.results["weights_found"]:
                print(f"   • {weight}")
            print()
            
        if self.results["weights_missing"]:
            print("❌ MISSING WEIGHTS:")
            for weight in self.results["weights_missing"]:
                print(f"   • {weight}")
            print()
            
        if self.results["errors"]:
            print("🚨 ERRORS:")
            for error in self.results["errors"]:
                print(f"   • {error}")
            print()
            
        print("💡 RECOMMENDATIONS:")
        
        if missing_weights > 0:
            print("   • Run `python setup_topaz_killer.py` to download missing model weights")
            
        if errors > 0:
            print("   • Fix import errors by checking model implementations")
            
        if self.results["dependencies_missing"]:
            print("   • Install missing dependencies with `pip install -r requirements.txt`")
            
        print("   • For production deployment, ensure all models have valid weights")
        print("   • Test models individually with `python -m models.enhancement.zeroshot.seedvr2_handler`")
        
        print("="*80)
        
        # Save detailed report
        report_path = self.project_root / "model_validation_report.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"📄 Detailed report saved to: {report_path}")
        
        # Return success status
        success_rate = (ok_deps + found_weights) / max(1, ok_deps + found_weights + missing_weights + errors)
        return success_rate >= 0.7  # 70% success threshold

def main():
    """Main validation execution."""
    try:
        validator = ModelValidator()
        success = validator.validate_all()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())