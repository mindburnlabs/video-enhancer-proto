#!/usr/bin/env python3
"""
Storage Management and Retention Policy

Comprehensive storage management system for video enhancement outputs
with configurable retention policies and automated cleanup.
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
import shutil
import logging
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import aiofiles
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

class StoragePolicy(str, Enum):
    """Storage retention policies"""
    DEVELOPMENT = "development"     # Keep for 1 day
    TESTING = "testing"            # Keep for 3 days  
    PRODUCTION = "production"      # Keep for 30 days
    ARCHIVAL = "archival"         # Keep for 1 year
    PERMANENT = "permanent"        # Never delete

@dataclass
class StorageItem:
    """Storage item metadata"""
    file_path: str
    job_id: str
    created_at: datetime
    file_size_bytes: int
    content_type: str
    policy: StoragePolicy
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Optional[Dict] = None
    checksum: Optional[str] = None

class StorageManager:
    """Comprehensive storage management with retention policies"""
    
    def __init__(self, 
                 base_path: str = "data",
                 default_policy: StoragePolicy = StoragePolicy.PRODUCTION):
        
        self.base_path = Path(base_path)
        self.default_policy = default_policy
        
        # Storage directories
        self.output_dir = self.base_path / "output"
        self.temp_dir = self.base_path / "temp" 
        self.uploads_dir = self.base_path / "uploads"
        self.cache_dir = self.base_path / "cache"
        self.metadata_dir = self.base_path / "metadata"
        
        # Retention periods
        self.retention_periods = {
            StoragePolicy.DEVELOPMENT: timedelta(days=1),
            StoragePolicy.TESTING: timedelta(days=3),
            StoragePolicy.PRODUCTION: timedelta(days=30),
            StoragePolicy.ARCHIVAL: timedelta(days=365),
            StoragePolicy.PERMANENT: None  # Never expires
        }
        
        # Storage limits (in GB)
        self.storage_limits = {
            "total_limit_gb": float(os.getenv("STORAGE_LIMIT_GB", "100")),
            "temp_limit_gb": float(os.getenv("TEMP_STORAGE_LIMIT_GB", "20")),
            "cache_limit_gb": float(os.getenv("CACHE_STORAGE_LIMIT_GB", "10")),
        }
        
        # Index file for tracking items
        self.index_file = self.metadata_dir / "storage_index.json"
        self.storage_index: Dict[str, StorageItem] = {}
        
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize storage directories and load index"""
        # Create directories
        for directory in [self.output_dir, self.temp_dir, self.uploads_dir, 
                         self.cache_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Load existing index
        self._load_storage_index()
        
        logger.info(f"Storage Manager initialized:")
        logger.info(f"  Base path: {self.base_path}")
        logger.info(f"  Default policy: {self.default_policy.value}")
        logger.info(f"  Total limit: {self.storage_limits['total_limit_gb']}GB")
        logger.info(f"  Tracked items: {len(self.storage_index)}")
    
    def _load_storage_index(self):
        """Load storage index from file"""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    data = json.load(f)
                    
                for item_id, item_data in data.items():
                    # Convert datetime strings back to datetime objects
                    item_data['created_at'] = datetime.fromisoformat(item_data['created_at'])
                    if item_data.get('last_accessed'):
                        item_data['last_accessed'] = datetime.fromisoformat(item_data['last_accessed'])
                    
                    self.storage_index[item_id] = StorageItem(**item_data)
                    
                logger.info(f"Loaded {len(self.storage_index)} items from storage index")
        except Exception as e:
            logger.warning(f"Could not load storage index: {e}")
            self.storage_index = {}
    
    def _save_storage_index(self):
        """Save storage index to file"""
        try:
            # Convert to serializable format
            serializable_data = {}
            for item_id, item in self.storage_index.items():
                item_dict = asdict(item)
                
                # Convert datetime objects to strings
                item_dict['created_at'] = item.created_at.isoformat()
                if item.last_accessed:
                    item_dict['last_accessed'] = item.last_accessed.isoformat()
                else:
                    item_dict['last_accessed'] = None
                    
                serializable_data[item_id] = item_dict
            
            with open(self.index_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save storage index: {e}")
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate checksum for {file_path}: {e}")
            return None
    
    async def store_file(self, 
                        file_path: str,
                        job_id: str,
                        content_type: str = "application/octet-stream",
                        policy: Optional[StoragePolicy] = None,
                        metadata: Optional[Dict] = None) -> str:
        """Store file with metadata and return storage ID"""
        
        policy = policy or self.default_policy
        source_path = Path(file_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")
        
        # Generate storage ID
        storage_id = f"{job_id}_{int(datetime.utcnow().timestamp())}"
        
        # Determine target directory based on content type
        if content_type.startswith('video/'):
            target_dir = self.output_dir
            extension = source_path.suffix or '.mp4'
        else:
            target_dir = self.uploads_dir
            extension = source_path.suffix
            
        target_path = target_dir / f"{storage_id}{extension}"
        
        try:
            # Copy file to storage
            shutil.copy2(source_path, target_path)
            
            # Calculate file info
            file_size = target_path.stat().st_size
            checksum = self._calculate_checksum(str(target_path))
            
            # Create storage item
            storage_item = StorageItem(
                file_path=str(target_path),
                job_id=job_id,
                created_at=datetime.utcnow(),
                file_size_bytes=file_size,
                content_type=content_type,
                policy=policy,
                access_count=0,
                metadata=metadata or {},
                checksum=checksum
            )
            
            # Add to index
            self.storage_index[storage_id] = storage_item
            self._save_storage_index()
            
            logger.info(f"Stored file {source_path.name} as {storage_id} "
                       f"({file_size / (1024*1024):.2f} MB, policy: {policy.value})")
            
            return storage_id
            
        except Exception as e:
            logger.error(f"Failed to store file: {e}")
            if target_path.exists():
                target_path.unlink()  # Cleanup on failure
            raise
    
    async def retrieve_file(self, storage_id: str) -> Optional[str]:
        """Retrieve file path and update access tracking"""
        
        if storage_id not in self.storage_index:
            return None
            
        storage_item = self.storage_index[storage_id]
        file_path = Path(storage_item.file_path)
        
        if not file_path.exists():
            logger.warning(f"Stored file missing: {file_path}")
            # Remove from index
            del self.storage_index[storage_id]
            self._save_storage_index()
            return None
        
        # Update access tracking
        storage_item.access_count += 1
        storage_item.last_accessed = datetime.utcnow()
        self._save_storage_index()
        
        return str(file_path)
    
    async def delete_file(self, storage_id: str, force: bool = False) -> bool:
        """Delete stored file"""
        
        if storage_id not in self.storage_index:
            return False
            
        storage_item = self.storage_index[storage_id]
        
        # Check if permanent and not forced
        if storage_item.policy == StoragePolicy.PERMANENT and not force:
            logger.warning(f"Attempted to delete permanent file {storage_id} without force")
            return False
        
        file_path = Path(storage_item.file_path)
        
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted stored file: {storage_id}")
            
            # Remove from index
            del self.storage_index[storage_id]
            self._save_storage_index()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file {storage_id}: {e}")
            return False
    
    async def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired files based on retention policy"""
        
        now = datetime.utcnow()
        cleanup_stats = {
            "scanned": 0,
            "expired": 0,
            "deleted": 0,
            "failed": 0,
            "bytes_freed": 0
        }
        
        expired_items = []
        
        for storage_id, storage_item in self.storage_index.items():
            cleanup_stats["scanned"] += 1
            
            # Skip permanent files
            if storage_item.policy == StoragePolicy.PERMANENT:
                continue
                
            retention_period = self.retention_periods[storage_item.policy]
            if retention_period is None:
                continue
                
            expiry_time = storage_item.created_at + retention_period
            
            if now > expiry_time:
                expired_items.append(storage_id)
                cleanup_stats["expired"] += 1
        
        # Delete expired items
        for storage_id in expired_items:
            storage_item = self.storage_index[storage_id]
            file_size = storage_item.file_size_bytes
            
            if await self.delete_file(storage_id):
                cleanup_stats["deleted"] += 1
                cleanup_stats["bytes_freed"] += file_size
            else:
                cleanup_stats["failed"] += 1
        
        if cleanup_stats["deleted"] > 0:
            logger.info(f"Cleanup completed: deleted {cleanup_stats['deleted']} expired files "
                       f"({cleanup_stats['bytes_freed'] / (1024*1024):.2f} MB freed)")
        
        return cleanup_stats
    
    async def cleanup_by_size(self, target_size_gb: float) -> Dict[str, int]:
        """Clean up files to achieve target storage size"""
        
        current_size = await self.get_storage_usage()
        target_bytes = target_size_gb * 1024 * 1024 * 1024
        
        cleanup_stats = {
            "target_gb": target_size_gb,
            "current_gb": current_size["total_gb"],
            "deleted": 0,
            "bytes_freed": 0
        }
        
        if current_size["total_bytes"] <= target_bytes:
            logger.info(f"Storage usage ({current_size['total_gb']:.2f} GB) "
                       f"already under target ({target_size_gb} GB)")
            return cleanup_stats
        
        # Sort files by access pattern (least recently used first)
        # Prioritize: temp files, then by last access, then by age
        sortable_items = []
        
        for storage_id, item in self.storage_index.items():
            if item.policy == StoragePolicy.PERMANENT:
                continue
                
            # Calculate priority score (higher = more likely to delete)
            priority = 0
            
            # Temp files get highest priority for deletion
            if "temp" in item.file_path:
                priority += 1000
                
            # Files that haven't been accessed recently
            if item.last_accessed:
                days_since_access = (datetime.utcnow() - item.last_accessed).days
                priority += days_since_access * 10
            else:
                priority += 365 * 10  # Never accessed
                
            # Older files get higher priority
            age_days = (datetime.utcnow() - item.created_at).days
            priority += age_days
            
            # Less accessed files get higher priority
            priority += max(0, 100 - item.access_count * 10)
            
            sortable_items.append((priority, storage_id, item))
        
        # Sort by priority (highest first)
        sortable_items.sort(key=lambda x: x[0], reverse=True)
        
        # Delete files until we reach target size
        current_bytes = current_size["total_bytes"]
        
        for priority, storage_id, item in sortable_items:
            if current_bytes <= target_bytes:
                break
                
            file_size = item.file_size_bytes
            
            if await self.delete_file(storage_id):
                cleanup_stats["deleted"] += 1
                cleanup_stats["bytes_freed"] += file_size
                current_bytes -= file_size
                
                logger.debug(f"Deleted {storage_id} to free space "
                           f"(priority: {priority}, size: {file_size / (1024*1024):.2f} MB)")
        
        final_gb = current_bytes / (1024 * 1024 * 1024)
        logger.info(f"Size-based cleanup: deleted {cleanup_stats['deleted']} files, "
                   f"storage now {final_gb:.2f} GB")
        
        return cleanup_stats
    
    async def get_storage_usage(self) -> Dict[str, any]:
        """Get detailed storage usage information"""
        
        usage = {
            "total_bytes": 0,
            "total_gb": 0.0,
            "by_directory": {},
            "by_policy": {},
            "file_count": len(self.storage_index),
            "orphaned_files": []
        }
        
        # Calculate usage from index
        for storage_id, item in self.storage_index.items():
            file_path = Path(item.file_path)
            
            # Check if file still exists
            if not file_path.exists():
                usage["orphaned_files"].append(storage_id)
                continue
                
            file_size = item.file_size_bytes
            usage["total_bytes"] += file_size
            
            # By directory
            directory = file_path.parent.name
            if directory not in usage["by_directory"]:
                usage["by_directory"][directory] = {"bytes": 0, "count": 0}
            usage["by_directory"][directory]["bytes"] += file_size
            usage["by_directory"][directory]["count"] += 1
            
            # By policy
            policy = item.policy.value
            if policy not in usage["by_policy"]:
                usage["by_policy"][policy] = {"bytes": 0, "count": 0}
            usage["by_policy"][policy]["bytes"] += file_size
            usage["by_policy"][policy]["count"] += 1
        
        usage["total_gb"] = usage["total_bytes"] / (1024 * 1024 * 1024)
        
        # Convert bytes to GB in subdicts
        for directory in usage["by_directory"]:
            usage["by_directory"][directory]["gb"] = \
                usage["by_directory"][directory]["bytes"] / (1024 * 1024 * 1024)
                
        for policy in usage["by_policy"]:
            usage["by_policy"][policy]["gb"] = \
                usage["by_policy"][policy]["bytes"] / (1024 * 1024 * 1024)
        
        return usage
    
    async def verify_integrity(self) -> Dict[str, any]:
        """Verify integrity of stored files"""
        
        verification = {
            "total_files": len(self.storage_index),
            "verified": 0,
            "missing": 0,
            "corrupted": 0,
            "orphaned_cleaned": 0,
            "checksum_failures": []
        }
        
        items_to_remove = []
        
        for storage_id, item in self.storage_index.items():
            file_path = Path(item.file_path)
            
            # Check if file exists
            if not file_path.exists():
                verification["missing"] += 1
                items_to_remove.append(storage_id)
                logger.warning(f"Missing stored file: {storage_id} ({file_path})")
                continue
            
            # Verify checksum if available
            if item.checksum:
                current_checksum = self._calculate_checksum(str(file_path))
                if current_checksum != item.checksum:
                    verification["corrupted"] += 1
                    verification["checksum_failures"].append({
                        "storage_id": storage_id,
                        "expected": item.checksum,
                        "actual": current_checksum
                    })
                    logger.error(f"Checksum mismatch for {storage_id}")
                    continue
            
            verification["verified"] += 1
        
        # Clean up orphaned entries
        for storage_id in items_to_remove:
            del self.storage_index[storage_id]
            verification["orphaned_cleaned"] += 1
        
        if items_to_remove:
            self._save_storage_index()
        
        logger.info(f"Storage verification: {verification['verified']} verified, "
                   f"{verification['missing']} missing, {verification['corrupted']} corrupted")
        
        return verification
    
    async def get_file_info(self, storage_id: str) -> Optional[Dict]:
        """Get detailed information about a stored file"""
        
        if storage_id not in self.storage_index:
            return None
            
        item = self.storage_index[storage_id]
        file_path = Path(item.file_path)
        
        info = asdict(item)
        info["exists"] = file_path.exists()
        info["file_size_mb"] = item.file_size_bytes / (1024 * 1024)
        
        if file_path.exists():
            stat = file_path.stat()
            info["actual_size_bytes"] = stat.st_size
            info["modified_time"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        # Convert datetime objects to strings for JSON serialization
        info["created_at"] = item.created_at.isoformat()
        if item.last_accessed:
            info["last_accessed"] = item.last_accessed.isoformat()
        
        return info
    
    async def run_maintenance(self) -> Dict[str, any]:
        """Run comprehensive storage maintenance"""
        
        logger.info("Starting storage maintenance...")
        
        maintenance_report = {
            "started_at": datetime.utcnow().isoformat(),
            "tasks_completed": [],
            "errors": []
        }
        
        try:
            # 1. Verify integrity
            logger.info("Running integrity verification...")
            integrity_result = await self.verify_integrity()
            maintenance_report["tasks_completed"].append({
                "task": "integrity_verification",
                "result": integrity_result
            })
            
            # 2. Cleanup expired files
            logger.info("Cleaning up expired files...")
            cleanup_result = await self.cleanup_expired()
            maintenance_report["tasks_completed"].append({
                "task": "expired_cleanup", 
                "result": cleanup_result
            })
            
            # 3. Check storage limits
            usage = await self.get_storage_usage()
            if usage["total_gb"] > self.storage_limits["total_limit_gb"]:
                logger.info(f"Storage usage ({usage['total_gb']:.2f} GB) exceeds limit "
                           f"({self.storage_limits['total_limit_gb']} GB), running size cleanup...")
                           
                size_cleanup_result = await self.cleanup_by_size(
                    self.storage_limits["total_limit_gb"] * 0.8  # Clean to 80% of limit
                )
                maintenance_report["tasks_completed"].append({
                    "task": "size_cleanup",
                    "result": size_cleanup_result
                })
            
            # 4. Final usage report
            final_usage = await self.get_storage_usage()
            maintenance_report["final_usage"] = final_usage
            
        except Exception as e:
            logger.error(f"Maintenance task failed: {e}")
            maintenance_report["errors"].append(str(e))
        
        maintenance_report["completed_at"] = datetime.utcnow().isoformat()
        
        logger.info("Storage maintenance completed")
        return maintenance_report


# Convenience functions for use in other modules
async def store_video_result(job_id: str, 
                           video_path: str, 
                           metadata: Optional[Dict] = None) -> str:
    """Store processed video result"""
    manager = StorageManager()
    return await manager.store_file(
        file_path=video_path,
        job_id=job_id,
        content_type="video/mp4",
        policy=StoragePolicy.PRODUCTION,
        metadata=metadata
    )

async def cleanup_temp_files():
    """Clean up temporary files"""
    manager = StorageManager()
    return await manager.cleanup_expired()

async def get_storage_stats():
    """Get storage statistics"""
    manager = StorageManager()
    return await manager.get_storage_usage()

# Export main class
__all__ = ['StorageManager', 'StoragePolicy', 'store_video_result', 
           'cleanup_temp_files', 'get_storage_stats']