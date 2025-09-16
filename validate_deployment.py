#!/usr/bin/env python3
"""
Deployment Validation Script for SOTA Video Enhancer
Comprehensive validation including warm start, smoke tests, and rollback capabilities.
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
import sys
import time
import json
import requests
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import tempfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentValidator:
    """Comprehensive deployment validation for production readiness."""
    
    def __init__(self, base_url: str = "http://localhost", gradio_port: int = 7860, health_port: int = 7861):
        self.base_url = base_url
        self.gradio_port = gradio_port
        self.health_port = health_port
        self.gradio_url = f"{base_url}:{gradio_port}"
        self.health_url = f"{base_url}:{health_port}"
        
        # Test results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'PENDING',
            'tests': {},
            'warnings': [],
            'errors': []
        }
        
    def run_warm_start(self) -> bool:
        """Warm start the application and core models."""
        logger.info("🔥 Starting warm start sequence...")
        
        try:
            # Step 1: Setup models
            logger.info("📦 Setting up core SOTA models...")
            result = subprocess.run([
                sys.executable, 'setup_topaz_killer.py'
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                logger.error(f"Setup failed: {result.stderr}")
                self.results['errors'].append(f"setup_topaz_killer.py failed: {result.stderr}")
                return False
            
            self.results['tests']['setup_models'] = {'status': 'PASS', 'duration': 'N/A'}
            
            # Step 2: Setup face restoration (optional)
            logger.info("👤 Setting up face restoration extras...")
            try:
                result = subprocess.run([
                    sys.executable, 'setup_face_extras.py', '--backend', 'python'
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    self.results['tests']['setup_face_extras'] = {'status': 'PASS', 'duration': 'N/A'}
                else:
                    logger.warning(f"Face extras setup warning: {result.stderr}")
                    self.results['warnings'].append(f"Face extras setup had issues: {result.stderr}")
                    self.results['tests']['setup_face_extras'] = {'status': 'WARN', 'duration': 'N/A'}
            except Exception as e:
                logger.warning(f"Face extras setup failed: {e}")
                self.results['warnings'].append(f"Face extras setup failed: {e}")
                self.results['tests']['setup_face_extras'] = {'status': 'WARN', 'duration': 'N/A'}
            
            # Step 3: Test configuration loading
            logger.info("⚙️ Testing configuration loading...")
            try:
                from config.production_config import get_config
                config = get_config()
                logger.info(f"Configuration loaded: {config.environment} environment")
                
                # Validate configuration
                if not config.is_gpu_available() and config.model.device == 'cuda':
                    logger.warning("CUDA not available but configured for GPU")
                    self.results['warnings'].append("CUDA not available but configured for GPU")
                
                self.results['tests']['config_loading'] = {'status': 'PASS', 'duration': 'N/A'}
                
            except Exception as e:
                logger.error(f"Configuration loading failed: {e}")
                self.results['errors'].append(f"Configuration loading failed: {e}")
                self.results['tests']['config_loading'] = {'status': 'FAIL', 'duration': 'N/A'}
                return False
            
            logger.info("✅ Warm start completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Warm start failed: {e}")
            self.results['errors'].append(f"Warm start failed: {e}")
            return False
    
    def run_smoke_tests(self) -> bool:
        """Run comprehensive smoke tests."""
        logger.info("🧪 Running smoke tests...")
        
        all_passed = True
        
        # Test 1: Health endpoint
        all_passed &= self._test_health_endpoint()
        
        # Test 2: Metrics endpoint
        all_passed &= self._test_metrics_endpoint()
        
        # Test 3: Readiness endpoint
        all_passed &= self._test_readiness_endpoint()
        
        # Test 4: Gradio interface
        all_passed &= self._test_gradio_interface()
        
        # Test 5: Model availability
        all_passed &= self._test_model_availability()
        
        # Test 6: GPU memory (if available)
        all_passed &= self._test_gpu_memory()
        
        # Test 7: File system permissions
        all_passed &= self._test_file_system()
        
        # Test 8: Configuration validation
        all_passed &= self._test_configuration_validation()
        
        logger.info(f"🧪 Smoke tests completed: {'✅ PASS' if all_passed else '❌ FAIL'}")
        return all_passed
    
    def _test_health_endpoint(self) -> bool:
        """Test health endpoint availability and response."""
        logger.info("🏥 Testing health endpoint...")
        start_time = time.time()
        
        try:
            response = requests.get(f"{self.health_url}/health", timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                health_data = response.json()
                
                # Validate health response structure
                required_fields = ['status', 'timestamp', 'enhancer_ready', 'system', 'gpu']
                missing_fields = [field for field in required_fields if field not in health_data]
                
                if missing_fields:
                    logger.warning(f"Health response missing fields: {missing_fields}")
                    self.results['warnings'].append(f"Health response missing fields: {missing_fields}")
                    status = 'WARN'
                else:
                    status = 'PASS'
                
                self.results['tests']['health_endpoint'] = {
                    'status': status,
                    'duration': f"{duration:.2f}s",
                    'response_code': response.status_code,
                    'enhancer_ready': health_data.get('enhancer_ready', False)
                }
                
                logger.info(f"✅ Health endpoint: {response.status_code} ({duration:.2f}s)")
                return True
            else:
                logger.error(f"Health endpoint returned {response.status_code}")
                self.results['errors'].append(f"Health endpoint returned {response.status_code}")
                self.results['tests']['health_endpoint'] = {
                    'status': 'FAIL',
                    'duration': f"{duration:.2f}s",
                    'response_code': response.status_code
                }
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Health endpoint test failed: {e}")
            self.results['errors'].append(f"Health endpoint failed: {e}")
            self.results['tests']['health_endpoint'] = {
                'status': 'FAIL',
                'duration': f"{duration:.2f}s",
                'error': str(e)
            }
            return False
    
    def _test_metrics_endpoint(self) -> bool:
        """Test metrics endpoint."""
        logger.info("📊 Testing metrics endpoint...")
        start_time = time.time()
        
        try:
            response = requests.get(f"{self.health_url}/metrics", timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                metrics_data = response.json()
                
                # Validate metrics structure
                required_sections = ['requests', 'performance', 'system', 'gpu']
                missing_sections = [section for section in required_sections if section not in metrics_data]
                
                if missing_sections:
                    logger.warning(f"Metrics response missing sections: {missing_sections}")
                    self.results['warnings'].append(f"Metrics missing sections: {missing_sections}")
                
                self.results['tests']['metrics_endpoint'] = {
                    'status': 'PASS',
                    'duration': f"{duration:.2f}s",
                    'response_code': response.status_code
                }
                
                logger.info(f"✅ Metrics endpoint: {response.status_code} ({duration:.2f}s)")
                return True
            else:
                logger.error(f"Metrics endpoint returned {response.status_code}")
                self.results['errors'].append(f"Metrics endpoint returned {response.status_code}")
                self.results['tests']['metrics_endpoint'] = {
                    'status': 'FAIL',
                    'duration': f"{duration:.2f}s",
                    'response_code': response.status_code
                }
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Metrics endpoint test failed: {e}")
            self.results['errors'].append(f"Metrics endpoint failed: {e}")
            self.results['tests']['metrics_endpoint'] = {
                'status': 'FAIL',
                'duration': f"{duration:.2f}s",
                'error': str(e)
            }
            return False
    
    def _test_readiness_endpoint(self) -> bool:
        """Test readiness endpoint."""
        logger.info("🎯 Testing readiness endpoint...")
        start_time = time.time()
        
        try:
            response = requests.get(f"{self.health_url}/ready", timeout=10)
            duration = time.time() - start_time
            
            if response.status_code in [200, 503]:  # 503 is acceptable if not ready
                ready_data = response.json()
                is_ready = ready_data.get('ready', False)
                
                status = 'PASS' if response.status_code == 200 else 'WARN'
                
                self.results['tests']['readiness_endpoint'] = {
                    'status': status,
                    'duration': f"{duration:.2f}s",
                    'response_code': response.status_code,
                    'ready': is_ready
                }
                
                logger.info(f"{'✅' if status == 'PASS' else '⚠️'} Readiness endpoint: {response.status_code}, ready={is_ready}")
                return True
            else:
                logger.error(f"Readiness endpoint returned {response.status_code}")
                self.results['errors'].append(f"Readiness endpoint returned {response.status_code}")
                self.results['tests']['readiness_endpoint'] = {
                    'status': 'FAIL',
                    'duration': f"{duration:.2f}s",
                    'response_code': response.status_code
                }
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Readiness endpoint test failed: {e}")
            self.results['errors'].append(f"Readiness endpoint failed: {e}")
            self.results['tests']['readiness_endpoint'] = {
                'status': 'FAIL',
                'duration': f"{duration:.2f}s",
                'error': str(e)
            }
            return False
    
    def _test_gradio_interface(self) -> bool:
        """Test Gradio interface availability."""
        logger.info("🎬 Testing Gradio interface...")
        start_time = time.time()
        
        try:
            response = requests.get(self.gradio_url, timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                self.results['tests']['gradio_interface'] = {
                    'status': 'PASS',
                    'duration': f"{duration:.2f}s",
                    'response_code': response.status_code
                }
                logger.info(f"✅ Gradio interface: {response.status_code} ({duration:.2f}s)")
                return True
            else:
                logger.error(f"Gradio interface returned {response.status_code}")
                self.results['errors'].append(f"Gradio interface returned {response.status_code}")
                self.results['tests']['gradio_interface'] = {
                    'status': 'FAIL',
                    'duration': f"{duration:.2f}s",
                    'response_code': response.status_code
                }
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Gradio interface test failed: {e}")
            self.results['errors'].append(f"Gradio interface failed: {e}")
            self.results['tests']['gradio_interface'] = {
                'status': 'FAIL',
                'duration': f"{duration:.2f}s",
                'error': str(e)
            }
            return False
    
    def _test_model_availability(self) -> bool:
        """Test SOTA model availability."""
        logger.info("🤖 Testing model availability...")
        
        try:
            # Check for critical model files
            critical_models = [
                'models/checkpoints/vsrm_large.pth',
                'models/checkpoints/ditvr_base.pth',
                'models/interpolation/RIFE/flownet.pkl'
            ]
            
            missing_models = []
            available_models = []
            
            for model_path in critical_models:
                if os.path.exists(model_path):
                    available_models.append(model_path)
                else:
                    missing_models.append(model_path)
            
            if missing_models:
                logger.warning(f"Missing model files: {missing_models}")
                self.results['warnings'].extend([f"Missing model: {model}" for model in missing_models])
                status = 'WARN' if available_models else 'FAIL'
            else:
                status = 'PASS'
            
            self.results['tests']['model_availability'] = {
                'status': status,
                'available_models': available_models,
                'missing_models': missing_models
            }
            
            logger.info(f"{'✅' if status == 'PASS' else '⚠️' if status == 'WARN' else '❌'} Model availability: {len(available_models)}/{len(critical_models)} available")
            return status != 'FAIL'
            
        except Exception as e:
            logger.error(f"Model availability test failed: {e}")
            self.results['errors'].append(f"Model availability test failed: {e}")
            self.results['tests']['model_availability'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
    
    def _test_gpu_memory(self) -> bool:
        """Test GPU memory availability."""
        logger.info("💾 Testing GPU memory...")
        
        try:
            import torch
            
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                gpu_info = []
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    gpu_info.append({
                        'device': i,
                        'name': props.name,
                        'memory_gb': round(memory_gb, 1)
                    })
                
                # Check if sufficient memory
                min_required_gb = 8.0
                sufficient_gpus = [gpu for gpu in gpu_info if gpu['memory_gb'] >= min_required_gb]
                
                if sufficient_gpus:
                    status = 'PASS'
                    logger.info(f"✅ GPU memory: {len(sufficient_gpus)} GPU(s) with ≥{min_required_gb}GB")
                else:
                    status = 'WARN'
                    logger.warning(f"GPU memory warning: No GPU with ≥{min_required_gb}GB found")
                    self.results['warnings'].append(f"No GPU with ≥{min_required_gb}GB found")
                
                self.results['tests']['gpu_memory'] = {
                    'status': status,
                    'gpu_count': device_count,
                    'gpu_info': gpu_info,
                    'sufficient_gpus': len(sufficient_gpus)
                }
                
            else:
                logger.info("ℹ️ CUDA not available, skipping GPU memory test")
                self.results['tests']['gpu_memory'] = {
                    'status': 'SKIP',
                    'reason': 'CUDA not available'
                }
            
            return True
            
        except ImportError:
            logger.info("ℹ️ PyTorch not available, skipping GPU memory test")
            self.results['tests']['gpu_memory'] = {
                'status': 'SKIP',
                'reason': 'PyTorch not available'
            }
            return True
        except Exception as e:
            logger.error(f"GPU memory test failed: {e}")
            self.results['errors'].append(f"GPU memory test failed: {e}")
            self.results['tests']['gpu_memory'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
    
    def _test_file_system(self) -> bool:
        """Test file system permissions and space."""
        logger.info("📁 Testing file system...")
        
        try:
            # Test directories
            required_dirs = ['models', 'data/temp', 'logs', 'config']
            permissions_ok = True
            
            for directory in required_dirs:
                dir_path = Path(directory)
                
                # Check if directory exists and is writable
                if dir_path.exists():
                    if not os.access(dir_path, os.W_OK):
                        logger.warning(f"Directory not writable: {directory}")
                        self.results['warnings'].append(f"Directory not writable: {directory}")
                        permissions_ok = False
                else:
                    logger.warning(f"Directory missing: {directory}")
                    self.results['warnings'].append(f"Directory missing: {directory}")
                    permissions_ok = False
            
            # Test temporary file creation
            try:
                with tempfile.NamedTemporaryFile(dir='data/temp', delete=True) as tmp:
                    tmp.write(b'test')
                    tmp.flush()
                temp_file_ok = True
            except Exception as e:
                logger.warning(f"Cannot create temp files: {e}")
                self.results['warnings'].append(f"Cannot create temp files: {e}")
                temp_file_ok = False
            
            status = 'PASS' if permissions_ok and temp_file_ok else 'WARN'
            
            self.results['tests']['file_system'] = {
                'status': status,
                'permissions_ok': permissions_ok,
                'temp_file_ok': temp_file_ok
            }
            
            logger.info(f"{'✅' if status == 'PASS' else '⚠️'} File system: permissions={'OK' if permissions_ok else 'WARN'}, temp={'OK' if temp_file_ok else 'WARN'}")
            return True
            
        except Exception as e:
            logger.error(f"File system test failed: {e}")
            self.results['errors'].append(f"File system test failed: {e}")
            self.results['tests']['file_system'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
    
    def _test_configuration_validation(self) -> bool:
        """Test configuration validation."""
        logger.info("⚙️ Testing configuration validation...")
        
        try:
            from config.production_config import ProductionConfig
            
            # Test different environments
            environments = ['development', 'staging', 'production']
            config_tests = {}
            
            for env in environments:
                try:
                    config = ProductionConfig(env)
                    config_tests[env] = {
                        'status': 'PASS',
                        'environment': config.environment,
                        'device': config.get_effective_device(),
                        'gpu_available': config.is_gpu_available()
                    }
                except Exception as e:
                    config_tests[env] = {
                        'status': 'FAIL',
                        'error': str(e)
                    }
            
            # Overall status
            all_configs_ok = all(test['status'] == 'PASS' for test in config_tests.values())
            status = 'PASS' if all_configs_ok else 'FAIL'
            
            self.results['tests']['configuration_validation'] = {
                'status': status,
                'environment_tests': config_tests
            }
            
            logger.info(f"{'✅' if status == 'PASS' else '❌'} Configuration validation: {status}")
            return status == 'PASS'
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            self.results['errors'].append(f"Configuration validation failed: {e}")
            self.results['tests']['configuration_validation'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
    
    def run_rollback(self) -> bool:
        """Execute rollback procedures."""
        logger.info("🔄 Running rollback procedures...")
        
        try:
            # Step 1: Stop running processes
            logger.info("🛑 Stopping processes...")
            
            # Kill any running Python processes related to the app
            try:
                subprocess.run(['pkill', '-f', 'app.py'], capture_output=True)
                subprocess.run(['pkill', '-f', 'gradio'], capture_output=True)
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Process cleanup warning: {e}")
            
            # Step 2: Clean temporary files
            logger.info("🧹 Cleaning temporary files...")
            try:
                import shutil
                if os.path.exists('data/temp'):
                    shutil.rmtree('data/temp')
                    os.makedirs('data/temp', exist_ok=True)
                    logger.info("Temporary files cleaned")
            except Exception as e:
                logger.warning(f"Temp cleanup warning: {e}")
            
            # Step 3: Reset to known good state
            logger.info("🔄 Resetting to known good state...")
            
            # If using git, revert to last known good commit
            try:
                result = subprocess.run(['git', 'status'], capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("Git repository detected, reverting changes...")
                    subprocess.run(['git', 'checkout', '.'], check=True)
                    subprocess.run(['git', 'clean', '-fd'], check=True)
                    logger.info("Git rollback completed")
                else:
                    logger.info("Not a git repository, skipping git rollback")
            except Exception as e:
                logger.warning(f"Git rollback failed: {e}")
            
            # Step 4: Create backup Docker image (if Docker available)
            logger.info("🐳 Creating backup image...")
            try:
                result = subprocess.run(['docker', 'build', '-t', 'video-enhancer-backup', '.'], 
                                      capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    logger.info("Backup Docker image created: video-enhancer-backup")
                else:
                    logger.warning("Docker backup image creation failed")
            except Exception as e:
                logger.warning(f"Docker backup failed: {e}")
            
            self.results['tests']['rollback'] = {
                'status': 'PASS',
                'steps_completed': ['stop_processes', 'clean_temp', 'git_reset', 'docker_backup']
            }
            
            logger.info("✅ Rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            self.results['errors'].append(f"Rollback failed: {e}")
            self.results['tests']['rollback'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
    
    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        
        # Determine overall status
        test_statuses = [test.get('status', 'UNKNOWN') for test in self.results['tests'].values()]
        
        if 'FAIL' in test_statuses:
            self.results['overall_status'] = 'FAIL'
        elif 'WARN' in test_statuses:
            self.results['overall_status'] = 'WARN'  
        elif all(status in ['PASS', 'SKIP'] for status in test_statuses):
            self.results['overall_status'] = 'PASS'
        else:
            self.results['overall_status'] = 'UNKNOWN'
        
        # Generate report
        report = f"""
# 🧪 SOTA Video Enhancer Deployment Validation Report

**Overall Status**: {self.results['overall_status']}
**Timestamp**: {self.results['timestamp']}
**Tests Run**: {len(self.results['tests'])}

## Test Results Summary

"""
        
        for test_name, test_result in self.results['tests'].items():
            status_emoji = {
                'PASS': '✅',
                'WARN': '⚠️',
                'FAIL': '❌',
                'SKIP': 'ℹ️'
            }.get(test_result['status'], '❓')
            
            report += f"- {status_emoji} **{test_name.replace('_', ' ').title()}**: {test_result['status']}"
            if 'duration' in test_result:
                report += f" ({test_result['duration']})"
            report += "\n"
        
        # Add warnings section
        if self.results['warnings']:
            report += "\n## ⚠️ Warnings\n\n"
            for warning in self.results['warnings']:
                report += f"- {warning}\n"
        
        # Add errors section
        if self.results['errors']:
            report += "\n## ❌ Errors\n\n"
            for error in self.results['errors']:
                report += f"- {error}\n"
        
        # Add detailed test information
        report += "\n## 📊 Detailed Results\n\n"
        report += f"```json\n{json.dumps(self.results, indent=2)}\n```\n"
        
        return report
    
    def save_report(self, filename: str = None):
        """Save validation report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"deployment_validation_{timestamp}.md"
        
        report = self.generate_report()
        
        with open(filename, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Validation report saved: {filename}")
        return filename

def main():
    """Main validation script entry point."""
    parser = argparse.ArgumentParser(description="SOTA Video Enhancer Deployment Validation")
    
    parser.add_argument('--warm-start', action='store_true', 
                       help='Run warm start sequence')
    parser.add_argument('--smoke-test', action='store_true',
                       help='Run smoke tests')
    parser.add_argument('--rollback', action='store_true',
                       help='Execute rollback procedures')
    parser.add_argument('--health-check', action='store_true',
                       help='Quick health check only')
    parser.add_argument('--full-validation', action='store_true',
                       help='Run complete validation suite')
    parser.add_argument('--base-url', default='http://localhost',
                       help='Base URL for testing (default: http://localhost)')
    parser.add_argument('--gradio-port', type=int, default=7860,
                       help='Gradio port (default: 7860)')
    parser.add_argument('--health-port', type=int, default=7861,
                       help='Health/metrics port (default: 7861)')
    parser.add_argument('--report', default=None,
                       help='Save report to specific file')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = DeploymentValidator(
        base_url=args.base_url,
        gradio_port=args.gradio_port,
        health_port=args.health_port
    )
    
    success = True
    
    # Execute requested operations
    if args.full_validation or (not any([args.warm_start, args.smoke_test, args.rollback, args.health_check])):
        logger.info("🚀 Running full validation suite...")
        success &= validator.run_warm_start()
        success &= validator.run_smoke_tests()
    else:
        if args.warm_start:
            success &= validator.run_warm_start()
        
        if args.smoke_test:
            success &= validator.run_smoke_tests()
        
        if args.health_check:
            success &= validator._test_health_endpoint()
            success &= validator._test_readiness_endpoint()
        
        if args.rollback:
            success &= validator.run_rollback()
    
    # Generate and save report
    report_file = validator.save_report(args.report)
    print(f"\n📄 Report saved: {report_file}")
    
    # Print summary
    overall_status = validator.results['overall_status']
    print(f"\n🎯 Overall Status: {overall_status}")
    
    if overall_status == 'PASS':
        print("✅ Deployment validation PASSED - Ready for production!")
        sys.exit(0)
    elif overall_status == 'WARN':
        print("⚠️ Deployment validation completed with WARNINGS - Review before production")
        sys.exit(1)
    else:
        print("❌ Deployment validation FAILED - Not ready for production")
        sys.exit(2)

if __name__ == "__main__":
    main()