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
import secrets
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import threading

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64

from utils.error_handler import (
    error_handler, SystemError, ErrorCode
)

logger = logging.getLogger(__name__)

class DataCategory(Enum):
    """Categories of data for different protection levels"""
    PERSONAL = "personal"          # Contains personal information
    ANONYMOUS = "anonymous"        # Anonymized or non-personal data
    TEMPORARY = "temporary"        # Temporary processing data
    METADATA = "metadata"          # Processing metadata only
    PUBLIC = "public"              # Safe for public access

class RetentionPolicy(Enum):
    """Data retention policies"""
    IMMEDIATE = "immediate"        # Delete immediately after use
    SHORT_TERM = "short_term"     # 24 hours
    MEDIUM_TERM = "medium_term"   # 7 days
    LONG_TERM = "long_term"       # 30 days
    PERMANENT = "permanent"       # Keep until explicit deletion

@dataclass
class DataProtectionPolicy:
    """Data protection policy configuration"""
    category: DataCategory
    retention: RetentionPolicy
    encryption_required: bool = True
    anonymization_required: bool = False
    audit_required: bool = True
    deletion_method: str = "secure"  # secure, standard
    geographic_restrictions: List[str] = field(default_factory=list)
    consent_required: bool = False

@dataclass
class DataRecord:
    """Record of protected data"""
    record_id: str
    file_path: Path
    category: DataCategory
    policy: DataProtectionPolicy
    created_at: datetime
    last_accessed: datetime
    expires_at: Optional[datetime]
    encrypted: bool = False
    anonymized: bool = False
    original_filename: str = ""
    user_consent: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class EncryptionManager:
    """Handles file encryption and decryption"""
    
    def __init__(self, key_file: Optional[str] = None):
        self.key_file = key_file or "data_protection_key"
        self.encryption_key = self._get_or_create_key()
        self.fernet = Fernet(self.encryption_key)
    
    def _get_or_create_key(self) -> bytes:
        """Get existing encryption key or create a new one"""
        key_path = Path(self.key_file)
        
        if key_path.exists():
            try:
                with open(key_path, 'rb') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Could not read existing key: {e}")
        
        # Create new key
        key = Fernet.generate_key()
        
        try:
            # Save key with restrictive permissions
            with open(key_path, 'wb') as f:
                f.write(key)
            os.chmod(key_path, 0o600)  # Owner read/write only
            logger.info(f"Created new encryption key: {key_path}")
        except Exception as e:
            logger.error(f"Failed to save encryption key: {e}")
        
        return key
    
    def encrypt_file(self, file_path: Path, output_path: Optional[Path] = None) -> Path:
        """Encrypt a file and return the encrypted file path"""
        try:
            output_path = output_path or file_path.with_suffix(file_path.suffix + '.encrypted')
            
            with open(file_path, 'rb') as infile:
                data = infile.read()
            
            encrypted_data = self.fernet.encrypt(data)
            
            with open(output_path, 'wb') as outfile:
                outfile.write(encrypted_data)
            
            # Set restrictive permissions
            os.chmod(output_path, 0o600)
            
            logger.debug(f"Encrypted file: {file_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to encrypt file {file_path}: {e}")
            raise SystemError(
                message=f"File encryption failed: {str(e)}",
                error_code=ErrorCode.SYSTEM_UNKNOWN
            )
    
    def decrypt_file(self, encrypted_path: Path, output_path: Optional[Path] = None) -> Path:
        """Decrypt a file and return the decrypted file path"""
        try:
            if output_path is None:
                if encrypted_path.suffix == '.encrypted':
                    output_path = encrypted_path.with_suffix('')
                else:
                    output_path = encrypted_path.with_suffix('.decrypted')
            
            with open(encrypted_path, 'rb') as infile:
                encrypted_data = infile.read()
            
            decrypted_data = self.fernet.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as outfile:
                outfile.write(decrypted_data)
            
            # Set restrictive permissions
            os.chmod(output_path, 0o600)
            
            logger.debug(f"Decrypted file: {encrypted_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to decrypt file {encrypted_path}: {e}")
            raise SystemError(
                message=f"File decryption failed: {str(e)}",
                error_code=ErrorCode.SYSTEM_UNKNOWN
            )
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt raw data"""
        return self.fernet.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt raw data"""
        return self.fernet.decrypt(encrypted_data)

class DataProtectionManager:
    """Manages data protection, encryption, and privacy compliance"""
    
    def __init__(self, data_dir: str = "protected_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, mode=0o700)
        
        self.encryption_manager = EncryptionManager()
        self.data_records: Dict[str, DataRecord] = {}
        self.protection_policies = self._load_default_policies()
        self._lock = threading.Lock()
        
        # Load existing records
        self._load_data_records()
        
        # Start cleanup scheduler
        self._last_cleanup = time.time()
    
    def _load_default_policies(self) -> Dict[DataCategory, DataProtectionPolicy]:
        """Load default data protection policies"""
        return {
            DataCategory.PERSONAL: DataProtectionPolicy(
                category=DataCategory.PERSONAL,
                retention=RetentionPolicy.SHORT_TERM,
                encryption_required=True,
                anonymization_required=True,
                audit_required=True,
                consent_required=True
            ),
            DataCategory.ANONYMOUS: DataProtectionPolicy(
                category=DataCategory.ANONYMOUS,
                retention=RetentionPolicy.MEDIUM_TERM,
                encryption_required=True,
                audit_required=True
            ),
            DataCategory.TEMPORARY: DataProtectionPolicy(
                category=DataCategory.TEMPORARY,
                retention=RetentionPolicy.IMMEDIATE,
                encryption_required=True,
                audit_required=False
            ),
            DataCategory.METADATA: DataProtectionPolicy(
                category=DataCategory.METADATA,
                retention=RetentionPolicy.LONG_TERM,
                encryption_required=False,
                audit_required=True
            ),
            DataCategory.PUBLIC: DataProtectionPolicy(
                category=DataCategory.PUBLIC,
                retention=RetentionPolicy.PERMANENT,
                encryption_required=False,
                audit_required=False
            )
        }
    
    def _load_data_records(self):
        """Load existing data records from metadata file"""
        metadata_file = self.data_dir / "data_records.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    records_data = json.load(f)
                
                for record_id, record_data in records_data.items():
                    self.data_records[record_id] = DataRecord(
                        record_id=record_data['record_id'],
                        file_path=Path(record_data['file_path']),
                        category=DataCategory(record_data['category']),
                        policy=self.protection_policies[DataCategory(record_data['category'])],
                        created_at=datetime.fromisoformat(record_data['created_at']),
                        last_accessed=datetime.fromisoformat(record_data['last_accessed']),
                        expires_at=datetime.fromisoformat(record_data['expires_at']) if record_data.get('expires_at') else None,
                        encrypted=record_data.get('encrypted', False),
                        anonymized=record_data.get('anonymized', False),
                        original_filename=record_data.get('original_filename', ''),
                        user_consent=record_data.get('user_consent', False),
                        metadata=record_data.get('metadata', {})
                    )
                
                logger.info(f"Loaded {len(self.data_records)} data protection records")
                
            except Exception as e:
                logger.error(f"Failed to load data records: {e}")
    
    def _save_data_records(self):
        """Save data records to metadata file"""
        metadata_file = self.data_dir / "data_records.json"
        
        try:
            records_data = {}
            for record_id, record in self.data_records.items():
                records_data[record_id] = {
                    'record_id': record.record_id,
                    'file_path': str(record.file_path),
                    'category': record.category.value,
                    'created_at': record.created_at.isoformat(),
                    'last_accessed': record.last_accessed.isoformat(),
                    'expires_at': record.expires_at.isoformat() if record.expires_at else None,
                    'encrypted': record.encrypted,
                    'anonymized': record.anonymized,
                    'original_filename': record.original_filename,
                    'user_consent': record.user_consent,
                    'metadata': record.metadata
                }
            
            with open(metadata_file, 'w') as f:
                json.dump(records_data, f, indent=2)
            
            # Set restrictive permissions
            os.chmod(metadata_file, 0o600)
            
        except Exception as e:
            logger.error(f"Failed to save data records: {e}")
    
    def protect_file(
        self,
        file_path: Path,
        category: DataCategory,
        original_filename: str = "",
        user_consent: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Apply data protection to a file and return record ID"""
        
        try:
            with self._lock:
                # Generate unique record ID
                record_id = f"dp_{int(time.time())}_{secrets.token_urlsafe(8)}"
                
                # Get protection policy
                policy = self.protection_policies[category]
                
                # Calculate expiration
                expires_at = None
                if policy.retention != RetentionPolicy.PERMANENT:
                    retention_hours = {
                        RetentionPolicy.IMMEDIATE: 0,
                        RetentionPolicy.SHORT_TERM: 24,
                        RetentionPolicy.MEDIUM_TERM: 168,  # 7 days
                        RetentionPolicy.LONG_TERM: 720,   # 30 days
                    }
                    expires_at = datetime.now() + timedelta(hours=retention_hours[policy.retention])
                
                # Create secure storage path
                secure_filename = f"{record_id}_{hashlib.sha256(str(file_path).encode()).hexdigest()[:16]}"
                secure_path = self.data_dir / secure_filename
                
                # Apply encryption if required
                if policy.encryption_required:
                    secure_path = self.encryption_manager.encrypt_file(file_path, secure_path)
                    encrypted = True
                else:
                    # Just copy to secure location
                    import shutil
                    shutil.copy2(file_path, secure_path)
                    os.chmod(secure_path, 0o600)
                    encrypted = False
                
                # Create data record
                record = DataRecord(
                    record_id=record_id,
                    file_path=secure_path,
                    category=category,
                    policy=policy,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    expires_at=expires_at,
                    encrypted=encrypted,
                    original_filename=original_filename,
                    user_consent=user_consent,
                    metadata=metadata or {}
                )
                
                # Check consent requirements
                if policy.consent_required and not user_consent:
                    logger.warning(f"Data protection applied without required consent: {record_id}")
                
                self.data_records[record_id] = record
                self._save_data_records()
                
                logger.info(f"Applied data protection: {record_id} ({category.value})")
                return record_id
                
        except Exception as e:
            logger.error(f"Failed to protect file {file_path}: {e}")
            raise SystemError(
                message=f"Data protection failed: {str(e)}",
                error_code=ErrorCode.SYSTEM_UNKNOWN
            )
    
    def access_protected_file(self, record_id: str) -> Optional[Path]:
        """Access a protected file (decrypt if necessary)"""
        
        with self._lock:
            if record_id not in self.data_records:
                logger.warning(f"Attempted access to unknown record: {record_id}")
                return None
            
            record = self.data_records[record_id]
            
            # Check expiration
            if record.expires_at and datetime.now() > record.expires_at:
                logger.info(f"Access denied to expired record: {record_id}")
                self._delete_record(record_id)
                return None
            
            # Update access time
            record.last_accessed = datetime.now()
            self._save_data_records()
            
            # Return decrypted file if encrypted
            if record.encrypted:
                try:
                    # Create temporary decrypted file
                    temp_path = self.data_dir / f"temp_{record_id}_{int(time.time())}"
                    decrypted_path = self.encryption_manager.decrypt_file(record.file_path, temp_path)
                    
                    logger.debug(f"Provided access to encrypted record: {record_id}")
                    return decrypted_path
                    
                except Exception as e:
                    logger.error(f"Failed to decrypt record {record_id}: {e}")
                    return None
            else:
                logger.debug(f"Provided access to record: {record_id}")
                return record.file_path
    
    def delete_protected_data(self, record_id: str, force: bool = False) -> bool:
        """Delete protected data"""
        
        with self._lock:
            if record_id not in self.data_records:
                return False
            
            record = self.data_records[record_id]
            
            # Check if deletion is allowed
            if not force and record.policy.retention == RetentionPolicy.PERMANENT:
                logger.warning(f"Attempted deletion of permanent record: {record_id}")
                return False
            
            return self._delete_record(record_id)
    
    def _delete_record(self, record_id: str) -> bool:
        """Internal method to delete a record"""
        
        try:
            record = self.data_records[record_id]
            
            # Secure deletion
            if record.policy.deletion_method == "secure":
                self._secure_delete_file(record.file_path)
            else:
                record.file_path.unlink(missing_ok=True)
            
            # Remove from records
            del self.data_records[record_id]
            self._save_data_records()
            
            logger.info(f"Deleted protected data record: {record_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete record {record_id}: {e}")
            return False
    
    def _secure_delete_file(self, file_path: Path):
        """Securely delete a file by overwriting"""
        
        try:
            if not file_path.exists():
                return
            
            file_size = file_path.stat().st_size
            
            # Overwrite with random data (3 passes)
            with open(file_path, 'r+b') as f:
                for _ in range(3):
                    f.seek(0)
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            
            # Delete the file
            file_path.unlink()
            
        except Exception as e:
            logger.error(f"Secure deletion failed for {file_path}: {e}")
            # Fallback to regular deletion
            file_path.unlink(missing_ok=True)
    
    def cleanup_expired_data(self):
        """Clean up expired data records"""
        
        now = datetime.now()
        expired_records = []
        
        with self._lock:
            for record_id, record in self.data_records.items():
                if record.expires_at and now > record.expires_at:
                    expired_records.append(record_id)
        
        for record_id in expired_records:
            logger.info(f"Cleaning up expired record: {record_id}")
            self._delete_record(record_id)
        
        if expired_records:
            logger.info(f"Cleaned up {len(expired_records)} expired records")
        
        self._last_cleanup = time.time()
    
    def get_protection_summary(self) -> Dict[str, Any]:
        """Get summary of protected data"""
        
        with self._lock:
            summary = {
                "total_records": len(self.data_records),
                "by_category": {},
                "by_retention": {},
                "encrypted_records": 0,
                "records_with_consent": 0,
                "expired_records": 0
            }
            
            now = datetime.now()
            
            for record in self.data_records.values():
                # By category
                category = record.category.value
                summary["by_category"][category] = summary["by_category"].get(category, 0) + 1
                
                # By retention
                retention = record.policy.retention.value
                summary["by_retention"][retention] = summary["by_retention"].get(retention, 0) + 1
                
                # Other stats
                if record.encrypted:
                    summary["encrypted_records"] += 1
                
                if record.user_consent:
                    summary["records_with_consent"] += 1
                
                if record.expires_at and now > record.expires_at:
                    summary["expired_records"] += 1
            
            return summary
    
    def anonymize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or hash personally identifiable information from metadata"""
        
        anonymized = {}
        
        # Fields to hash instead of removing
        fields_to_hash = {'user_id', 'client_id', 'session_id', 'ip_address'}
        
        # Fields to remove completely
        fields_to_remove = {'email', 'name', 'address', 'phone'}
        
        for key, value in metadata.items():
            if key.lower() in fields_to_remove:
                continue  # Skip PII fields
            elif key.lower() in fields_to_hash:
                # Hash the value
                anonymized[key] = hashlib.sha256(str(value).encode()).hexdigest()[:16]
            else:
                anonymized[key] = value
        
        return anonymized
    
    def audit_data_access(self, record_id: str, action: str, user_info: Dict[str, Any]):
        """Audit data access for compliance"""
        
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "record_id": record_id,
            "action": action,
            "user_info": self.anonymize_metadata(user_info)
        }
        
        # In production, send to audit logging system
        logger.info(f"Data access audit: {json.dumps(audit_entry)}")

# Global data protection manager
data_protection_manager = DataProtectionManager()