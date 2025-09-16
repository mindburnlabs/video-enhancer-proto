#!/usr/bin/env python3
"""
License Audit and Compliance Script

Scans the project for dependencies, license information, and generates
compliance documentation.
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
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class LicenseAuditor:
    """Comprehensive license auditing for the video enhancement project"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self.licenses_found: Dict[str, Dict] = {}
        self.dependencies: Dict[str, Dict] = {}
        self.problematic_licenses: Set[str] = {
            'GPL-3.0', 'AGPL-3.0', 'GPL-2.0', 'AGPL-1.0', 
            'SSPL', 'BUSL', 'Commons Clause'
        }
        self.permissive_licenses: Set[str] = {
            'MIT', 'Apache-2.0', 'BSD-2-Clause', 'BSD-3-Clause', 
            'ISC', 'Unlicense', 'CC0-1.0', 'Python-2.0'
        }
        
    def audit_python_dependencies(self) -> Dict[str, Dict]:
        """Audit Python package dependencies and their licenses"""
        python_deps = {}
        
        try:
            # Get license information using pip list directly
            # Note: pip-licenses isn't used directly in code, just for installation checking
            
            # Get license information using pip-licenses
            result = subprocess.run([
                sys.executable, "-m", "pip", "list", "--format=json"
            ], capture_output=True, text=True, check=True)
            
            packages = json.loads(result.stdout)
            
            for package in packages:
                name = package['name']
                version = package['version']
                
                # Get detailed license info
                license_info = self._get_package_license_info(name, version)
                
                python_deps[name] = {
                    'version': version,
                    'license': license_info.get('license', 'Unknown'),
                    'license_classifier': license_info.get('license_classifier', []),
                    'home_page': license_info.get('home_page', ''),
                    'author': license_info.get('author', ''),
                    'summary': license_info.get('summary', ''),
                    'is_problematic': license_info.get('license', '') in self.problematic_licenses,
                    'is_permissive': license_info.get('license', '') in self.permissive_licenses
                }
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get Python dependencies: {e}")
        except Exception as e:
            logger.error(f"Error analyzing Python dependencies: {e}")
        
        return python_deps
    
    def _get_package_license_info(self, package_name: str, version: str) -> Dict:
        """Get detailed license information for a specific package"""
        try:
            # Try to get package metadata
            result = subprocess.run([
                sys.executable, "-m", "pip", "show", package_name
            ], capture_output=True, text=True)
            
            info = {}
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower().replace('-', '_')
                        info[key] = value.strip()
            
            # Extract license from metadata
            license_text = info.get('license', 'Unknown')
            
            # Clean up license text
            if license_text and license_text != 'Unknown':
                license_text = self._normalize_license_name(license_text)
            
            return {
                'license': license_text,
                'license_classifier': info.get('classifier', '').split('\n') if info.get('classifier') else [],
                'home_page': info.get('home_page', ''),
                'author': info.get('author', ''),
                'summary': info.get('summary', '')
            }
            
        except Exception as e:
            logger.warning(f"Could not get license info for {package_name}: {e}")
            return {'license': 'Unknown'}
    
    def _normalize_license_name(self, license_text: str) -> str:
        """Normalize license names to standard identifiers"""
        license_text = license_text.strip()
        
        # Common license mappings
        mappings = {
            'MIT License': 'MIT',
            'Apache Software License': 'Apache-2.0',
            'Apache License 2.0': 'Apache-2.0',
            'Apache': 'Apache-2.0',
            'BSD License': 'BSD-3-Clause',
            'BSD': 'BSD-3-Clause',
            'GNU General Public License v3 or later (GPLv3+)': 'GPL-3.0+',
            'GNU General Public License v3 (GPLv3)': 'GPL-3.0',
            'Mozilla Public License 2.0 (MPL 2.0)': 'MPL-2.0',
            'ISC License (ISCL)': 'ISC',
            'Python Software Foundation License': 'Python-2.0'
        }
        
        for pattern, standard in mappings.items():
            if pattern.lower() in license_text.lower():
                return standard
        
        return license_text
    
    def audit_project_files(self) -> Dict[str, List[str]]:
        """Scan project files for license headers and copyright notices"""
        file_licenses = {}
        
        # File patterns to scan
        patterns = [
            "**/*.py", "**/*.js", "**/*.ts", "**/*.java", "**/*.cpp", 
            "**/*.c", "**/*.h", "**/*.md", "**/*.txt", "**/LICENSE*", 
            "**/COPYING*", "**/COPYRIGHT*"
        ]
        
        for pattern in patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file() and not self._should_skip_file(file_path):
                    license_info = self._extract_license_from_file(file_path)
                    if license_info:
                        file_licenses[str(file_path.relative_to(self.project_root))] = license_info
        
        return file_licenses
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during license scanning"""
        skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv'}
        skip_files = {'.pyc', '.pyo', '.pyd', '.so', '.dylib', '.dll'}
        
        # Check if in skip directory
        for part in file_path.parts:
            if part in skip_dirs:
                return True
        
        # Check file extension
        if file_path.suffix in skip_files:
            return True
        
        # Check file size (skip very large files)
        try:
            if file_path.stat().st_size > 1024 * 1024:  # 1MB
                return True
        except:
            pass
        
        return False
    
    def _extract_license_from_file(self, file_path: Path) -> List[str]:
        """Extract license information from file headers"""
        licenses = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first 50 lines to check for license headers
                content = ""
                for i, line in enumerate(f):
                    if i >= 50:
                        break
                    content += line.lower()
            
            # License patterns to look for
            patterns = {
                'MIT': r'mit license|mit\s+licensed',
                'Apache-2.0': r'apache license.*version 2\.0|apache-2\.0',
                'GPL-3.0': r'gnu general public license.*version 3|gpl-3\.0|gplv3',
                'GPL-2.0': r'gnu general public license.*version 2|gpl-2\.0|gplv2',
                'BSD-3-Clause': r'bsd 3-clause|bsd license.*3-clause',
                'BSD-2-Clause': r'bsd 2-clause|bsd license.*2-clause',
                'ISC': r'isc license',
                'MPL-2.0': r'mozilla public license.*version 2\.0',
            }
            
            for license_name, pattern in patterns.items():
                if re.search(pattern, content):
                    licenses.append(license_name)
            
            # Look for copyright notices
            copyright_pattern = r'copyright\s+\(c\)\s+\d{4}'
            if re.search(copyright_pattern, content):
                licenses.append('Copyright-Notice')
                
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
        
        return licenses
    
    def check_requirements_files(self) -> Dict[str, List[str]]:
        """Check requirements files for dependencies"""
        requirements = {}
        
        req_files = [
            'requirements.txt', 
            'requirements-dev.txt', 
            'requirements-test.txt',
            'requirements-minimal.txt',
            'pyproject.toml',
            'setup.py'
        ]
        
        for req_file in req_files:
            file_path = self.project_root / req_file
            if file_path.exists():
                requirements[req_file] = self._parse_requirements_file(file_path)
        
        return requirements
    
    def _parse_requirements_file(self, file_path: Path) -> List[str]:
        """Parse a requirements file and extract package names"""
        packages = []
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('-'):
                        # Extract package name (before version specifiers)
                        package = re.split(r'[>=<!=]', line)[0].strip()
                        if package:
                            packages.append(package)
        except Exception as e:
            logger.warning(f"Could not parse {file_path}: {e}")
        
        return packages
    
    def generate_license_report(self) -> Dict:
        """Generate comprehensive license report"""
        print("🔍 Starting license audit...")
        
        # Gather all license information
        python_deps = self.audit_python_dependencies()
        file_licenses = self.audit_project_files()
        requirements = self.check_requirements_files()
        
        # Analyze findings
        problematic_deps = {
            name: info for name, info in python_deps.items() 
            if info['is_problematic']
        }
        
        unknown_licenses = {
            name: info for name, info in python_deps.items() 
            if info['license'] == 'Unknown'
        }
        
        permissive_deps = {
            name: info for name, info in python_deps.items() 
            if info['is_permissive']
        }
        
        # License summary
        license_counts = {}
        for dep_info in python_deps.values():
            license_name = dep_info['license']
            license_counts[license_name] = license_counts.get(license_name, 0) + 1
        
        report = {
            'audit_timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'summary': {
                'total_dependencies': len(python_deps),
                'problematic_licenses': len(problematic_deps),
                'unknown_licenses': len(unknown_licenses),
                'permissive_licenses': len(permissive_deps),
                'license_distribution': license_counts
            },
            'dependencies': python_deps,
            'problematic_dependencies': problematic_deps,
            'unknown_licenses': unknown_licenses,
            'permissive_dependencies': permissive_deps,
            'file_licenses': file_licenses,
            'requirements_files': requirements,
            'compliance_status': 'PASS' if len(problematic_deps) == 0 else 'REVIEW_REQUIRED',
            'recommendations': self._generate_recommendations(problematic_deps, unknown_licenses)
        }
        
        return report
    
    def _generate_recommendations(self, problematic_deps: Dict, unknown_licenses: Dict) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        if problematic_deps:
            recommendations.append(
                f"⚠️ {len(problematic_deps)} dependencies have potentially problematic licenses that may "
                "require legal review or replacement with permissive alternatives."
            )
            
            for name, info in problematic_deps.items():
                recommendations.append(f"  - {name} ({info['license']}): Consider alternatives")
        
        if unknown_licenses:
            recommendations.append(
                f"❓ {len(unknown_licenses)} dependencies have unknown licenses. "
                "Manual review required to determine compliance."
            )
        
        if not problematic_deps and not unknown_licenses:
            recommendations.append("✅ All identified licenses appear to be permissive and compatible.")
        
        recommendations.extend([
            "📋 Consider adding license headers to project source files",
            "📄 Create/update LICENSES.md with all dependency licenses",
            "🔄 Set up automated license scanning in CI/CD pipeline",
            "📝 Document license policy and compliance procedures"
        ])
        
        return recommendations

def main():
    """Main function to run license audit"""
    
    project_root = Path(__file__).parent.parent
    auditor = LicenseAuditor(project_root)
    
    print("🔍 Starting comprehensive license audit...")
    report = auditor.generate_license_report()
    
    # Save report
    output_file = project_root / "AUDIT_LICENSE_REPORT.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print(f"\n📊 License Audit Summary:")
    print(f"Total dependencies: {report['summary']['total_dependencies']}")
    print(f"Problematic licenses: {report['summary']['problematic_licenses']}")
    print(f"Unknown licenses: {report['summary']['unknown_licenses']}")
    print(f"Permissive licenses: {report['summary']['permissive_licenses']}")
    print(f"Compliance status: {report['compliance_status']}")
    
    print(f"\n📋 Top licenses:")
    for license_name, count in sorted(report['summary']['license_distribution'].items(), 
                                     key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {license_name}: {count} packages")
    
    if report['problematic_dependencies']:
        print(f"\n⚠️ Problematic dependencies:")
        for name, info in report['problematic_dependencies'].items():
            print(f"  {name} ({info['version']}): {info['license']}")
    
    print(f"\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"  {rec}")
    
    print(f"\n📄 Full report saved to: {output_file}")
    
    return report['compliance_status'] == 'PASS'

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)