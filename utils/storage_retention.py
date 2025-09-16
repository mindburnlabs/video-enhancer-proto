#!/usr/bin/env python3
"""
Storage Retention Policy and Cleanup Management

Implements comprehensive storage management with configurable retention policies,
automated cleanup jobs, and disk usage monitoring.
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
import time
import shutil
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import json
import fnmatch
from concurrent.futures import ThreadPoolExecutor
# import schedule  # Optional dependency, fallback to basic scheduler

logger = logging.getLogger(__name__)

@dataclass
class RetentionRule:
    """Configuration for retention policy rule"""
    name: str
    path_pattern: str  # glob pattern for paths to match
    max_age_hours: int  # maximum age before cleanup
    max_size_mb: Optional[int] = None  # maximum total size
    max_count: Optional[int] = None  # maximum file count
    preserve_recent: int = 5  # always preserve N most recent files
    enabled: bool = True
    description: str = ""

@dataclass
class CleanupResult:
    """Result of cleanup operation"""
    rule_name: str
    files_removed: int
    bytes_freed: int
    errors: List[str]
    execution_time: float
    timestamp: datetime

class StorageRetentionManager:
    """Manages storage retention policies and cleanup operations"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/storage_retention.json"
        self.base_dir = Path.cwd()
        self.cleanup_history: List[CleanupResult] = []
        self.max_history = 100
        
        # Default retention rules
        self.default_rules = [
            RetentionRule(
                name="temp_files",
                path_pattern="data/temp/*",
                max_age_hours=24,
                max_size_mb=1024,  # 1GB
                preserve_recent=10,
                description="Temporary processing files"
            ),
            RetentionRule(
                name="uploads",
                path_pattern="data/uploads/*",
                max_age_hours=72,  # 3 days
                max_size_mb=2048,  # 2GB
                preserve_recent=20,
                description="User uploaded files"
            ),
            RetentionRule(
                name="test_output",
                path_pattern="data/test_output/*",
                max_age_hours=168,  # 1 week
                max_count=50,
                preserve_recent=10,
                description="Test output files"
            ),
            RetentionRule(
                name="api_outputs",
                path_pattern="data/api_output/*",
                max_age_hours=72,  # 3 days
                max_size_mb=5120,  # 5GB
                preserve_recent=15,
                description="API processing outputs"
            ),
            RetentionRule(
                name="logs",
                path_pattern="logs/*.log*",
                max_age_hours=336,  # 2 weeks
                max_size_mb=512,  # 512MB
                preserve_recent=5,
                description="Application log files"
            ),
            RetentionRule(
                name="cache",
                path_pattern="cache/*",
                max_age_hours=504,  # 3 weeks
                max_size_mb=1024,  # 1GB
                preserve_recent=3,
                description="Cache files"
            )
        ]
        
        # Load configuration
        self.rules = self._load_config()
        
        # Background cleanup scheduler
        self.scheduler_thread = None
        self.scheduler_running = False
        
        logger.info(f"Storage retention manager initialized with {len(self.rules)} rules")
    
    def _load_config(self) -> List[RetentionRule]:
        """Load retention rules from configuration file"""
        config_file = Path(self.config_path)
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                rules = []
                for rule_data in config_data.get('retention_rules', []):
                    rule = RetentionRule(**rule_data)
                    rules.append(rule)
                
                logger.info(f"Loaded {len(rules)} retention rules from {config_file}")
                return rules
                
            except Exception as e:
                logger.error(f"Failed to load retention config from {config_file}: {e}")
                logger.info("Using default retention rules")
        
        return self.default_rules.copy()
    
    def save_config(self):
        """Save current retention rules to configuration file"""
        try:
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            config_data = {
                'retention_rules': [
                    {
                        'name': rule.name,
                        'path_pattern': rule.path_pattern,
                        'max_age_hours': rule.max_age_hours,
                        'max_size_mb': rule.max_size_mb,
                        'max_count': rule.max_count,
                        'preserve_recent': rule.preserve_recent,
                        'enabled': rule.enabled,
                        'description': rule.description
                    }
                    for rule in self.rules
                ]
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Saved retention configuration to {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save retention config: {e}")
    
    def add_rule(self, rule: RetentionRule):
        """Add a new retention rule"""
        self.rules.append(rule)
        logger.info(f"Added retention rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a retention rule by name"""
        original_count = len(self.rules)
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        
        if len(self.rules) < original_count:
            logger.info(f"Removed retention rule: {rule_name}")
            return True
        
        return False
    
    def get_matching_files(self, rule: RetentionRule) -> List[Path]:
        """Get files matching a retention rule pattern"""
        try:
            pattern_path = Path(rule.path_pattern)
            
            if pattern_path.is_absolute():
                search_root = pattern_path.parent
                pattern = pattern_path.name
            else:
                search_root = self.base_dir / pattern_path.parent
                pattern = pattern_path.name
            
            if not search_root.exists():
                return []
            
            matching_files = []
            
            # Handle recursive patterns
            if '**' in str(pattern_path):
                for file_path in search_root.rglob('*'):
                    if file_path.is_file() and fnmatch.fnmatch(str(file_path), str(self.base_dir / rule.path_pattern)):
                        matching_files.append(file_path)
            else:
                for file_path in search_root.iterdir():
                    if file_path.is_file() and fnmatch.fnmatch(file_path.name, pattern):
                        matching_files.append(file_path)
            
            return matching_files
            
        except Exception as e:
            logger.error(f"Error finding files for pattern {rule.path_pattern}: {e}")
            return []
    
    def should_cleanup_file(self, file_path: Path, rule: RetentionRule) -> bool:
        """Determine if a file should be cleaned up based on rule"""
        try:
            # Check if file exists
            if not file_path.exists():
                return False
            
            stat = file_path.stat()
            file_age = datetime.now() - datetime.fromtimestamp(stat.st_mtime)
            
            # Check age limit
            if rule.max_age_hours and file_age > timedelta(hours=rule.max_age_hours):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking cleanup criteria for {file_path}: {e}")
            return False
    
    def cleanup_by_rule(self, rule: RetentionRule) -> CleanupResult:
        """Execute cleanup for a specific rule"""
        start_time = time.time()
        files_removed = 0
        bytes_freed = 0
        errors = []
        
        try:
            if not rule.enabled:
                logger.debug(f"Skipping disabled rule: {rule.name}")
                return CleanupResult(
                    rule_name=rule.name,
                    files_removed=0,
                    bytes_freed=0,
                    errors=["Rule disabled"],
                    execution_time=0,
                    timestamp=datetime.now()
                )
            
            logger.info(f"Running cleanup for rule: {rule.name}")
            
            # Get matching files
            matching_files = self.get_matching_files(rule)
            
            if not matching_files:
                logger.debug(f"No files found for rule {rule.name}")
                return CleanupResult(
                    rule_name=rule.name,
                    files_removed=0,
                    bytes_freed=0,
                    errors=[],
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now()
                )
            
            # Sort by modification time (newest first)
            matching_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Always preserve recent files
            preserved_files = matching_files[:rule.preserve_recent]
            candidates = matching_files[rule.preserve_recent:]
            
            logger.debug(f"Found {len(matching_files)} files, preserving {len(preserved_files)} recent files")
            
            # Apply cleanup criteria
            files_to_remove = []
            
            # Age-based cleanup
            for file_path in candidates:
                if self.should_cleanup_file(file_path, rule):
                    files_to_remove.append(file_path)
            
            # Size-based cleanup
            if rule.max_size_mb:
                current_size = sum(f.stat().st_size for f in matching_files if f.exists()) / (1024 * 1024)
                
                if current_size > rule.max_size_mb:
                    # Remove oldest files until under size limit
                    size_to_free = current_size - rule.max_size_mb
                    bytes_to_free = size_to_free * 1024 * 1024
                    
                    for file_path in reversed(candidates):  # Start with oldest
                        if file_path not in files_to_remove and file_path.exists():
                            file_size = file_path.stat().st_size
                            files_to_remove.append(file_path)
                            bytes_to_free -= file_size
                            
                            if bytes_to_free <= 0:
                                break
            
            # Count-based cleanup
            if rule.max_count and len(matching_files) > rule.max_count:
                excess_count = len(matching_files) - rule.max_count
                
                # Add oldest files to removal list
                for file_path in reversed(candidates[:excess_count]):
                    if file_path not in files_to_remove:
                        files_to_remove.append(file_path)
            
            # Remove duplicates and execute cleanup
            files_to_remove = list(set(files_to_remove))
            
            logger.info(f"Removing {len(files_to_remove)} files for rule {rule.name}")
            
            for file_path in files_to_remove:
                try:
                    if file_path.exists():
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        files_removed += 1
                        bytes_freed += file_size
                        logger.debug(f"Removed: {file_path}")
                    
                except Exception as e:
                    error_msg = f"Failed to remove {file_path}: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
        except Exception as e:
            error_msg = f"Error during cleanup for rule {rule.name}: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        execution_time = time.time() - start_time
        
        result = CleanupResult(
            rule_name=rule.name,
            files_removed=files_removed,
            bytes_freed=bytes_freed,
            errors=errors,
            execution_time=execution_time,
            timestamp=datetime.now()
        )
        
        logger.info(f"Cleanup completed for {rule.name}: {files_removed} files removed, "
                   f"{bytes_freed / (1024*1024):.1f}MB freed in {execution_time:.2f}s")
        
        return result
    
    def cleanup_all(self) -> List[CleanupResult]:
        """Execute cleanup for all enabled rules"""
        logger.info("Starting comprehensive storage cleanup")
        
        results = []
        total_files = 0
        total_bytes = 0
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit cleanup tasks
            futures = {
                executor.submit(self.cleanup_by_rule, rule): rule 
                for rule in self.rules if rule.enabled
            }
            
            # Collect results
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per rule
                    results.append(result)
                    total_files += result.files_removed
                    total_bytes += result.bytes_freed
                    
                except Exception as e:
                    rule = futures[future]
                    error_result = CleanupResult(
                        rule_name=rule.name,
                        files_removed=0,
                        bytes_freed=0,
                        errors=[f"Cleanup failed: {e}"],
                        execution_time=0,
                        timestamp=datetime.now()
                    )
                    results.append(error_result)
                    logger.error(f"Cleanup failed for rule {rule.name}: {e}")
        
        # Store results in history
        self.cleanup_history.extend(results)
        
        # Trim history if needed
        if len(self.cleanup_history) > self.max_history:
            self.cleanup_history = self.cleanup_history[-self.max_history:]
        
        logger.info(f"Storage cleanup completed: {total_files} files removed, "
                   f"{total_bytes / (1024*1024):.1f}MB freed")
        
        return results
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get current storage statistics"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'rules': [],
            'total_disk_usage': {},
            'recent_cleanup': []
        }
        
        # Per-rule statistics
        for rule in self.rules:
            matching_files = self.get_matching_files(rule)
            
            if matching_files:
                total_size = sum(f.stat().st_size for f in matching_files if f.exists())
                oldest_file = min(matching_files, key=lambda x: x.stat().st_mtime)
                newest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
                
                rule_stats = {
                    'name': rule.name,
                    'enabled': rule.enabled,
                    'file_count': len(matching_files),
                    'total_size_mb': total_size / (1024 * 1024),
                    'oldest_file_age_hours': (datetime.now() - datetime.fromtimestamp(oldest_file.stat().st_mtime)).total_seconds() / 3600,
                    'newest_file_age_hours': (datetime.now() - datetime.fromtimestamp(newest_file.stat().st_mtime)).total_seconds() / 3600,
                    'pattern': rule.path_pattern
                }
            else:
                rule_stats = {
                    'name': rule.name,
                    'enabled': rule.enabled,
                    'file_count': 0,
                    'total_size_mb': 0,
                    'oldest_file_age_hours': 0,
                    'newest_file_age_hours': 0,
                    'pattern': rule.path_pattern
                }
            
            stats['rules'].append(rule_stats)
        
        # Overall disk usage
        try:
            for directory in ['data', 'logs', 'cache']:
                dir_path = Path(directory)
                if dir_path.exists():
                    total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                    stats['total_disk_usage'][directory] = {
                        'size_mb': total_size / (1024 * 1024),
                        'file_count': len([f for f in dir_path.rglob('*') if f.is_file()])
                    }
        except Exception as e:
            logger.error(f"Error calculating disk usage: {e}")
        
        # Recent cleanup history
        stats['recent_cleanup'] = [
            {
                'rule_name': result.rule_name,
                'files_removed': result.files_removed,
                'bytes_freed': result.bytes_freed,
                'timestamp': result.timestamp.isoformat(),
                'had_errors': len(result.errors) > 0
            }
            for result in self.cleanup_history[-10:]  # Last 10 cleanup operations
        ]
        
        return stats
    
    def start_scheduler(self, cleanup_interval_hours: int = 6):
        """Start automatic cleanup scheduler"""
        if self.scheduler_running:
            logger.warning("Cleanup scheduler is already running")
            return
        
        def run_scheduler():
            logger.info(f"Storage cleanup scheduler started (interval: {cleanup_interval_hours}h)")
            
            cleanup_interval_seconds = cleanup_interval_hours * 3600
            last_cleanup = time.time()
            
            while self.scheduler_running:
                try:
                    current_time = time.time()
                    
                    # Check if it's time for cleanup
                    if current_time - last_cleanup >= cleanup_interval_seconds:
                        logger.info("Running scheduled storage cleanup")
                        self.cleanup_all()
                        last_cleanup = current_time
                    
                    time.sleep(300)  # Check every 5 minutes
                    
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
                    time.sleep(300)
        
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
    
    def stop_scheduler(self):
        """Stop automatic cleanup scheduler"""
        if self.scheduler_running:
            self.scheduler_running = False
            logger.info("Storage cleanup scheduler stopped")
    
    def emergency_cleanup(self, target_free_mb: int = 1024) -> Dict[str, Any]:
        """Perform emergency cleanup to free specified amount of disk space"""
        logger.warning(f"Emergency cleanup initiated - target: {target_free_mb}MB")
        
        # Get current disk usage
        stats = self.get_storage_stats()
        
        # Sort rules by urgency (temp files first, then by age limits)
        urgent_rules = sorted(
            [rule for rule in self.rules if rule.enabled],
            key=lambda r: (r.name == 'temp_files', -r.max_age_hours)
        )
        
        total_freed = 0
        results = []
        
        for rule in urgent_rules:
            if total_freed >= target_free_mb * 1024 * 1024:  # Convert MB to bytes
                break
            
            # Temporarily reduce preserve_recent for emergency
            original_preserve = rule.preserve_recent
            rule.preserve_recent = max(1, rule.preserve_recent // 2)
            
            try:
                result = self.cleanup_by_rule(rule)
                results.append(result)
                total_freed += result.bytes_freed
                
            finally:
                # Restore original preserve count
                rule.preserve_recent = original_preserve
        
        logger.info(f"Emergency cleanup completed: {total_freed / (1024*1024):.1f}MB freed")
        
        return {
            'target_mb': target_free_mb,
            'actual_freed_mb': total_freed / (1024 * 1024),
            'success': total_freed >= target_free_mb * 1024 * 1024 * 0.8,  # 80% of target
            'results': results
        }

# Global instance
_retention_manager: Optional[StorageRetentionManager] = None

def get_retention_manager() -> StorageRetentionManager:
    """Get or create global retention manager instance"""
    global _retention_manager
    if _retention_manager is None:
        _retention_manager = StorageRetentionManager()
    return _retention_manager

# Convenience functions
def cleanup_storage() -> List[CleanupResult]:
    """Run storage cleanup with current retention policies"""
    return get_retention_manager().cleanup_all()

def get_storage_stats() -> Dict[str, Any]:
    """Get current storage statistics"""
    return get_retention_manager().get_storage_stats()

def emergency_cleanup(target_free_mb: int = 1024) -> Dict[str, Any]:
    """Perform emergency cleanup to free disk space"""
    return get_retention_manager().emergency_cleanup(target_free_mb)

def start_cleanup_scheduler(interval_hours: int = 6):
    """Start automatic cleanup scheduler"""
    return get_retention_manager().start_scheduler(interval_hours)

# Export main classes and functions
__all__ = [
    'StorageRetentionManager', 'RetentionRule', 'CleanupResult',
    'get_retention_manager', 'cleanup_storage', 'get_storage_stats',
    'emergency_cleanup', 'start_cleanup_scheduler'
]