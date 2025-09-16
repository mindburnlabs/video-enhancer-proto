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
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, BinaryIO, Any
import mimetypes
import subprocess
import re

from utils.error_handler import (
    InputValidationError, ErrorCode, error_handler
)
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SecurityThreat:
    """Represents a security threat detected in a file"""
    threat_type: str
    description: str
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float  # 0.0 to 1.0
    details: Dict[str, Any]
    detected_at: datetime = None
    
    def __post_init__(self):
        if self.detected_at is None:
            self.detected_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'threat_type': self.threat_type,
            'description': self.description,
            'risk_level': self.risk_level,
            'confidence': self.confidence,
            'details': self.details,
            'detected_at': self.detected_at.isoformat()
        }

# Video file magic numbers (first few bytes that identify file types)
VIDEO_MAGIC_NUMBERS = {
    # MP4/QuickTime
    b'\x00\x00\x00\x18ftypmp4': 'video/mp4',
    b'\x00\x00\x00\x20ftypM4V': 'video/mp4',
    b'\x00\x00\x00\x18ftypM4A': 'video/mp4',
    b'\x00\x00\x00\x18ftypisom': 'video/mp4',
    b'\x00\x00\x00\x18ftypqt': 'video/quicktime',
    
    # AVI
    b'RIFF': 'video/x-msvideo',  # Need additional check for AVI
    
    # WebM
    b'\x1A\x45\xDF\xA3': 'video/webm',
    
    # Matroska (MKV)
    b'\x1A\x45\xDF\xA3': 'video/x-matroska',
    
    # FLV
    b'FLV': 'video/x-flv',
    
    # 3GP
    b'\x00\x00\x00\x14ftyp3gp': 'video/3gpp',
    b'\x00\x00\x00\x18ftyp3g2': 'video/3gpp2',
    
    # WMV/ASF
    b'\x30\x26\xB2\x75\x8E\x66\xCF\x11': 'video/x-ms-wmv',
    
    # OGV (Ogg Video)
    b'OggS': 'video/ogg',
    
    # MOV (additional patterns)
    b'\x00\x00\x00\x14ftypqt': 'video/quicktime',
}

# Allowed video extensions and their expected MIME types
ALLOWED_VIDEO_EXTENSIONS = {
    '.mp4': 'video/mp4',
    '.avi': 'video/x-msvideo',
    '.mov': 'video/quicktime',
    '.mkv': 'video/x-matroska',
    '.webm': 'video/webm',
    '.flv': 'video/x-flv',
    '.wmv': 'video/x-ms-wmv',
    '.ogv': 'video/ogg',
    '.3gp': 'video/3gpp',
    '.m4v': 'video/mp4',
}

# Suspicious file characteristics
SUSPICIOUS_PATTERNS = [
    # Executable signatures
    b'MZ',  # Windows PE
    b'\x7fELF',  # Linux ELF
    b'\xfe\xed\xfa\xce',  # Mach-O (macOS)
    b'\xce\xfa\xed\xfe',  # Mach-O reverse
    b'\xcf\xfa\xed\xfe',  # Mach-O 64-bit
    
    # Script signatures
    b'#!/bin/sh',
    b'#!/bin/bash',
    b'#!/usr/bin/python',
    b'<script>',
    b'<?php',
    
    # Archive signatures that could contain malware
    b'PK\x03\x04',  # ZIP
    b'Rar!',  # RAR
    b'\x1f\x8b',  # GZIP
]

class FileSecurityValidator:
    """Comprehensive file security validator for video uploads"""
    
    def __init__(self, max_file_size: int = 500 * 1024 * 1024):  # 500MB default
        self.max_file_size = max_file_size
        self.temp_dir = Path(tempfile.gettempdir()) / "video_enhancement_secure"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Virus scanning simulation (in production, integrate with real scanner)
        self.virus_scanning_enabled = os.getenv('VIRUS_SCANNING_ENABLED', 'false').lower() == 'true'
    
    def validate_file_upload(
        self, 
        file_content: bytes, 
        filename: str, 
        content_type: str
    ) -> Dict[str, Any]:
        """
        Comprehensive file validation including security checks
        
        Returns validation result with security assessment
        """
        validation_result = {
            'valid': False,
            'filename': filename,
            'content_type': content_type,
            'file_size': len(file_content),
            'detected_type': None,
            'security_checks': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # 1. Basic file validation
            self._validate_basic_file_properties(file_content, filename, validation_result)
            
            # 2. Magic number validation
            self._validate_file_magic_numbers(file_content, validation_result)
            
            # 3. Extension and MIME type validation
            self._validate_extension_and_mime_type(filename, content_type, validation_result)
            
            # 4. Suspicious content detection
            self._detect_suspicious_content(file_content, validation_result)
            
            # 5. Path traversal prevention
            self._validate_filename_security(filename, validation_result)
            
            # 6. Virus scanning (if enabled)
            if self.virus_scanning_enabled:
                self._scan_for_viruses(file_content, validation_result)
            
            # 7. Metadata extraction and validation
            self._validate_file_metadata(file_content, validation_result)
            
            # Determine overall validation result
            validation_result['valid'] = (
                len(validation_result['errors']) == 0 and
                validation_result['security_checks'].get('magic_number_valid', False) and
                validation_result['security_checks'].get('extension_valid', False) and
                not validation_result['security_checks'].get('suspicious_content', False)
            )
            
            logger.info(
                f"File validation completed: {filename} "
                f"({'VALID' if validation_result['valid'] else 'INVALID'})"
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"File validation error for {filename}: {e}")
            validation_result['errors'].append(f"Validation error: {str(e)}")
            return validation_result
    
    def _validate_basic_file_properties(
        self, 
        file_content: bytes, 
        filename: str, 
        result: Dict[str, Any]
    ):
        """Validate basic file properties"""
        
        # File size validation
        if len(file_content) > self.max_file_size:
            result['errors'].append(
                f"File too large: {len(file_content)} bytes (max: {self.max_file_size} bytes)"
            )
        
        if len(file_content) < 1024:  # Minimum 1KB
            result['errors'].append(
                f"File too small: {len(file_content)} bytes (suspicious)"
            )
        
        # Filename validation
        if not filename or len(filename) > 255:
            result['errors'].append("Invalid filename length")
        
        result['security_checks']['size_valid'] = len(result['errors']) == 0
    
    def _validate_file_magic_numbers(self, file_content: bytes, result: Dict[str, Any]):
        """Validate file content using magic numbers"""
        
        detected_type = None
        magic_valid = False
        
        # Check against known video magic numbers
        for magic_bytes, mime_type in VIDEO_MAGIC_NUMBERS.items():
            if file_content.startswith(magic_bytes):
                detected_type = mime_type
                magic_valid = True
                break
            
            # Special case for RIFF containers (AVI)
            if magic_bytes == b'RIFF' and file_content.startswith(b'RIFF'):
                # Check if it's actually an AVI file
                if b'AVI ' in file_content[8:12]:
                    detected_type = 'video/x-msvideo'
                    magic_valid = True
                    break
        
        result['detected_type'] = detected_type
        result['security_checks']['magic_number_valid'] = magic_valid
        
        if not magic_valid:
            result['errors'].append("File content doesn't match a known video format")
    
    def _validate_extension_and_mime_type(
        self, 
        filename: str, 
        content_type: str, 
        result: Dict[str, Any]
    ):
        """Validate file extension and MIME type consistency"""
        
        file_extension = Path(filename).suffix.lower()
        extension_valid = file_extension in ALLOWED_VIDEO_EXTENSIONS
        
        result['security_checks']['extension_valid'] = extension_valid
        
        if not extension_valid:
            result['errors'].append(f"Unsupported file extension: {file_extension}")
            return
        
        # Check MIME type consistency
        expected_mime = ALLOWED_VIDEO_EXTENSIONS[file_extension]
        mime_consistent = (
            content_type == expected_mime or
            result['detected_type'] == expected_mime
        )
        
        result['security_checks']['mime_consistent'] = mime_consistent
        
        if not mime_consistent:
            result['warnings'].append(
                f"MIME type mismatch: got {content_type}, expected {expected_mime}"
            )
    
    def _detect_suspicious_content(self, file_content: bytes, result: Dict[str, Any]):
        """Detect suspicious content patterns"""
        
        suspicious_found = False
        suspicious_patterns = []
        
        # Check for suspicious signatures
        for pattern in SUSPICIOUS_PATTERNS:
            if pattern in file_content[:1024]:  # Check first 1KB
                suspicious_found = True
                suspicious_patterns.append(pattern.hex())
        
        # Check for embedded scripts or executables
        suspicious_strings = [
            b'<script>',
            b'javascript:',
            b'vbscript:',
            b'<?php',
            b'#!/bin/',
            b'PowerShell',
            b'cmd.exe',
            b'System.Diagnostics'
        ]
        
        for sus_string in suspicious_strings:
            if sus_string in file_content:
                suspicious_found = True
                suspicious_patterns.append(sus_string.decode('utf-8', errors='ignore'))
        
        result['security_checks']['suspicious_content'] = suspicious_found
        result['security_checks']['suspicious_patterns'] = suspicious_patterns
        
        if suspicious_found:
            result['errors'].append(
                f"Suspicious content detected: {', '.join(suspicious_patterns[:3])}"
            )
    
    def _validate_filename_security(self, filename: str, result: Dict[str, Any]):
        """Prevent path traversal and other filename-based attacks"""
        
        filename_safe = True
        security_issues = []
        
        # Check for path traversal patterns
        dangerous_patterns = [
            '..',      # Directory traversal
            '/',       # Absolute paths
            '\\',      # Windows path separators
            ':',       # Drive letters or alternate streams
            '<', '>',  # HTML/XML injection
            '|', '&',  # Command injection
            ';',       # Command chaining
            '\x00',    # Null byte injection
        ]
        
        for pattern in dangerous_patterns:
            if pattern in filename:
                filename_safe = False
                security_issues.append(f"Dangerous character: {repr(pattern)}")
        
        # Check for reserved filenames (Windows)
        reserved_names = [
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        ]
        
        filename_base = Path(filename).stem.upper()
        if filename_base in reserved_names:
            filename_safe = False
            security_issues.append(f"Reserved filename: {filename_base}")
        
        # Check for overly long filenames
        if len(filename) > 255:
            filename_safe = False
            security_issues.append("Filename too long")
        
        result['security_checks']['filename_safe'] = filename_safe
        result['security_checks']['filename_issues'] = security_issues
        
        if not filename_safe:
            result['errors'].extend([f"Unsafe filename: {issue}" for issue in security_issues])
    
    def _scan_for_viruses(self, file_content: bytes, result: Dict[str, Any]):
        """Virus scanning (simulation - integrate with real scanner in production)"""
        
        # This is a simulation - in production, integrate with:
        # - ClamAV
        # - VirusTotal API
        # - Cloud-based scanning services
        
        virus_found = False
        scan_result = "clean"
        
        # Simulate virus detection based on suspicious patterns
        if result['security_checks'].get('suspicious_content', False):
            # In a real implementation, this would be actual virus scanning
            virus_found = len(result['security_checks'].get('suspicious_patterns', [])) > 2
            scan_result = "suspicious" if virus_found else "clean"
        
        result['security_checks']['virus_scan_performed'] = True
        result['security_checks']['virus_found'] = virus_found
        result['security_checks']['virus_scan_result'] = scan_result
        
        if virus_found:
            result['errors'].append("Potential malware detected")
        
        logger.info(f"Virus scan result: {scan_result}")
    
    def _validate_file_metadata(self, file_content: bytes, result: Dict[str, Any]):
        """Extract and validate file metadata"""
        
        try:
            # Basic metadata validation
            # In production, use libraries like python-magic or ffprobe
            
            metadata = {
                'file_hash': hashlib.sha256(file_content).hexdigest(),
                'content_length': len(file_content),
                'estimated_type': result.get('detected_type', 'unknown')
            }
            
            # Check for embedded metadata that could be problematic
            # (scripts in metadata, excessive metadata, etc.)
            
            result['metadata'] = metadata
            result['security_checks']['metadata_valid'] = True
            
        except Exception as e:
            logger.warning(f"Metadata validation failed: {e}")
            result['warnings'].append(f"Metadata validation failed: {e}")
            result['security_checks']['metadata_valid'] = False

def create_secure_temp_file(
    file_content: bytes, 
    filename: str, 
    validation_result: Dict[str, Any]
) -> Path:
    """Create a secure temporary file with proper permissions"""
    
    if not validation_result['valid']:
        raise InputValidationError(
            message="Cannot create temp file for invalid upload",
            error_code=ErrorCode.INPUT_INVALID_FORMAT
        )
    
    # Generate secure filename
    file_hash = hashlib.sha256(file_content).hexdigest()[:16]
    original_extension = Path(filename).suffix.lower()
    secure_filename = f"secure_{file_hash}{original_extension}"
    
    # Create secure temp directory
    secure_temp_dir = Path(tempfile.mkdtemp(prefix="video_enhancement_"))
    temp_file_path = secure_temp_dir / secure_filename
    
    try:
        # Write file with restrictive permissions
        with open(temp_file_path, 'wb') as f:
            f.write(file_content)
        
        # Set restrictive permissions (owner read/write only)
        os.chmod(temp_file_path, 0o600)
        os.chmod(secure_temp_dir, 0o700)
        
        logger.info(f"Created secure temp file: {temp_file_path}")
        return temp_file_path
        
    except Exception as e:
        # Cleanup on failure
        if temp_file_path.exists():
            temp_file_path.unlink()
        if secure_temp_dir.exists():
            secure_temp_dir.rmdir()
        raise

def secure_file_cleanup(file_path: Path):
    """Securely delete a file by overwriting it before deletion"""
    
    try:
        if not file_path.exists():
            return
        
        # Get file size
        file_size = file_path.stat().st_size
        
        # Overwrite with random data (3 passes)
        with open(file_path, 'r+b') as f:
            for _ in range(3):
                f.seek(0)
                f.write(os.urandom(file_size))
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
        
        # Remove file
        file_path.unlink()
        
        # Remove parent directory if empty and it's our temp dir
        parent = file_path.parent
        if parent.name.startswith("tmp") and not list(parent.iterdir()):
            parent.rmdir()
        
        logger.debug(f"Securely deleted file: {file_path}")
        
    except Exception as e:
        logger.error(f"Secure file cleanup failed for {file_path}: {e}")

def sanitize_path(path: str) -> str:
    """Sanitize a file path to prevent directory traversal"""
    
    # Remove any path separators and dangerous characters
    sanitized = re.sub(r'[^\w\s\.-]', '', path)
    
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip('. ')
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "unknown_file"
    
    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    
    return sanitized

class FileSecurityManager:
    """Manager class for file security operations"""
    
    def __init__(self, max_file_size: int = 500 * 1024 * 1024):
        self.validator = FileSecurityValidator(max_file_size)
        self.max_file_size = max_file_size
    
    def scan_file(self, file_path: Path) -> List[SecurityThreat]:
        """Scan a file and return list of security threats"""
        threats = []
        
        try:
            if not file_path.exists():
                threats.append(SecurityThreat(
                    threat_type="FILE_NOT_FOUND",
                    description=f"File does not exist: {file_path}",
                    risk_level="HIGH",
                    confidence=1.0,
                    details={"file_path": str(file_path)}
                ))
                return threats
            
            # Read file content
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Run validation
            validation_result = self.validator.validate_file_upload(
                file_content, file_path.name, 
                mimetypes.guess_type(str(file_path))[0] or 'application/octet-stream'
            )
            
            # Convert validation results to security threats
            if not validation_result['valid']:
                for error in validation_result.get('errors', []):
                    threats.append(SecurityThreat(
                        threat_type="VALIDATION_ERROR",
                        description=error,
                        risk_level="MEDIUM",
                        confidence=0.8,
                        details=validation_result
                    ))
            
            # Check for suspicious content
            if validation_result['security_checks'].get('suspicious_content', False):
                patterns = validation_result['security_checks'].get('suspicious_patterns', [])
                threats.append(SecurityThreat(
                    threat_type="SUSPICIOUS_CONTENT",
                    description=f"Suspicious patterns detected: {', '.join(patterns[:3])}",
                    risk_level="HIGH",
                    confidence=0.9,
                    details={
                        'patterns': patterns,
                        'pattern_count': len(patterns)
                    }
                ))
            
            # Check for virus scan results (simulation)
            if validation_result['security_checks'].get('virus_found', False):
                threats.append(SecurityThreat(
                    threat_type="MALWARE",
                    description="Potential malware detected",
                    risk_level="CRITICAL",
                    confidence=0.95,
                    details={
                        'scan_result': validation_result['security_checks'].get('virus_scan_result')
                    }
                ))
            
            # Add low-risk informational threats for warnings
            for warning in validation_result.get('warnings', []):
                threats.append(SecurityThreat(
                    threat_type="WARNING",
                    description=warning,
                    risk_level="LOW",
                    confidence=0.5,
                    details={"warning": warning}
                ))
            
            logger.info(f"File security scan completed for {file_path.name}: {len(threats)} threats detected")
            return threats
            
        except Exception as e:
            logger.error(f"File security scan failed for {file_path}: {e}")
            threats.append(SecurityThreat(
                threat_type="SCAN_ERROR",
                description=f"Security scan failed: {str(e)}",
                risk_level="MEDIUM",
                confidence=0.7,
                details={"error": str(e), "file_path": str(file_path)}
            ))
            return threats
    
    def validate_file(self, file_path: Path) -> Tuple[bool, List[SecurityThreat]]:
        """Validate a file and return validation result and threats"""
        threats = self.scan_file(file_path)
        
        # Consider file valid if no high or critical risk threats
        high_risk_threats = [
            t for t in threats 
            if t.risk_level in ['HIGH', 'CRITICAL']
        ]
        
        is_valid = len(high_risk_threats) == 0
        return is_valid, threats
    
    def create_secure_temp_file(self, file_content: bytes, filename: str) -> Path:
        """Create a secure temporary file"""
        validation_result = self.validator.validate_file_upload(
            file_content, filename,
            mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        )
        
        if not validation_result['valid']:
            raise InputValidationError(
                message=f"File validation failed: {validation_result.get('errors', [])[0] if validation_result.get('errors') else 'Unknown error'}",
                error_code=ErrorCode.INPUT_INVALID_FORMAT
            )
        
        return create_secure_temp_file(file_content, filename, validation_result)
    
    def cleanup_file(self, file_path: Path):
        """Securely clean up a file"""
        secure_file_cleanup(file_path)

# Global instances
file_security_validator = FileSecurityValidator()
file_security_manager = FileSecurityManager()
