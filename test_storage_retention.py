#!/usr/bin/env python3
"""
Test script for storage retention and cleanup system
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
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from utils.storage_retention import (
    StorageRetentionManager,
    RetentionRule,
    get_retention_manager,
    cleanup_storage,
    get_storage_stats,
    emergency_cleanup
)

def create_test_files():
    """Create test files for cleanup testing"""
    print("📁 Creating test files...")
    
    # Create test directories
    test_dirs = [
        'data/temp',
        'data/uploads', 
        'data/test_output',
        'data/api_output',
        'logs'
    ]
    
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create test files with different ages
    test_files = [
        # Recent temp files (should be preserved)
        ('data/temp/recent1.tmp', 0),  # Created now
        ('data/temp/recent2.tmp', 0),
        
        # Old temp files (should be cleaned up)
        ('data/temp/old1.tmp', 48),  # 48 hours old
        ('data/temp/old2.tmp', 36),  # 36 hours old
        ('data/temp/old3.tmp', 72),  # 72 hours old
        
        # Upload files
        ('data/uploads/upload1.mp4', 0),
        ('data/uploads/upload2.mp4', 12),  # 12 hours old
        ('data/uploads/old_upload.mp4', 96),  # 4 days old (should be cleaned)
        
        # Test output files
        ('data/test_output/test1.mp4', 0),
        ('data/test_output/test2.mp4', 24),
        ('data/test_output/old_test.mp4', 200),  # > 1 week (should be cleaned)
        
        # Log files
        ('logs/app.log', 0),
        ('logs/old.log', 400)  # Very old log
    ]
    
    created_count = 0
    
    for file_path, age_hours in test_files:
        try:
            path = Path(file_path)
            
            # Create file with some content
            with open(path, 'w') as f:
                f.write(f"Test file created at {datetime.now()}\n")
                f.write("This is test content for storage retention testing.\n" * 10)
            
            # Modify file timestamp if needed
            if age_hours > 0:
                old_time = datetime.now() - timedelta(hours=age_hours)
                timestamp = old_time.timestamp()
                os.utime(path, (timestamp, timestamp))
            
            created_count += 1
            print(f"  Created: {file_path} (age: {age_hours}h)")
            
        except Exception as e:
            print(f"  ❌ Failed to create {file_path}: {e}")
    
    print(f"✅ Created {created_count} test files")
    return created_count

def test_storage_stats():
    """Test storage statistics collection"""
    print("\n📊 Testing storage statistics...")
    
    try:
        stats = get_storage_stats()
        
        print(f"  Total directories tracked: {len(stats.get('total_disk_usage', {}))}")
        print(f"  Retention rules: {len(stats.get('rules', []))}")
        
        # Show per-rule stats
        for rule_stats in stats.get('rules', []):
            print(f"    {rule_stats['name']}: {rule_stats['file_count']} files, "
                  f"{rule_stats['total_size_mb']:.1f}MB")
        
        print("✅ Storage stats test passed")
        return True
        
    except Exception as e:
        print(f"❌ Storage stats test failed: {e}")
        return False

def test_retention_manager():
    """Test retention manager functionality"""
    print("\n🔧 Testing retention manager...")
    
    try:
        # Create custom retention manager for testing
        manager = StorageRetentionManager()
        
        # Add a custom test rule
        test_rule = RetentionRule(
            name="test_rule",
            path_pattern="data/temp/*.tmp",
            max_age_hours=24,  # 1 day
            preserve_recent=2,
            description="Test rule for temp files"
        )
        
        manager.add_rule(test_rule)
        print(f"  Added test rule: {test_rule.name}")
        
        # Test file matching
        matching_files = manager.get_matching_files(test_rule)
        print(f"  Found {len(matching_files)} matching files")
        
        # Test cleanup by rule
        result = manager.cleanup_by_rule(test_rule)
        print(f"  Cleanup result: {result.files_removed} files removed, "
              f"{result.bytes_freed / 1024:.1f}KB freed")
        
        print("✅ Retention manager test passed")
        return True
        
    except Exception as e:
        print(f"❌ Retention manager test failed: {e}")
        return False

def test_cleanup_operations():
    """Test cleanup operations"""
    print("\n🧹 Testing cleanup operations...")
    
    try:
        # Get initial stats
        initial_stats = get_storage_stats()
        initial_files = {}
        for rule in initial_stats['rules']:
            initial_files[rule['name']] = rule['file_count']
        
        # Run cleanup
        print("  Running comprehensive cleanup...")
        results = cleanup_storage()
        
        total_files_removed = sum(r.files_removed for r in results)
        total_mb_freed = sum(r.bytes_freed for r in results) / (1024 * 1024)
        
        print(f"  Cleanup completed:")
        print(f"    Files removed: {total_files_removed}")
        print(f"    Space freed: {total_mb_freed:.2f}MB")
        
        # Show per-rule results
        for result in results:
            if result.files_removed > 0:
                print(f"    {result.rule_name}: {result.files_removed} files, "
                      f"{result.bytes_freed / 1024:.1f}KB")
        
        # Verify some files were cleaned up
        final_stats = get_storage_stats()
        
        print("✅ Cleanup operations test passed")
        return True
        
    except Exception as e:
        print(f"❌ Cleanup operations test failed: {e}")
        return False

def test_emergency_cleanup():
    """Test emergency cleanup functionality"""
    print("\n🚨 Testing emergency cleanup...")
    
    try:
        # Create some large test files if needed
        large_file_path = Path('data/temp/large_test.tmp')
        if not large_file_path.exists():
            with open(large_file_path, 'w') as f:
                f.write('x' * (1024 * 1024))  # 1MB file
            print("  Created large test file for emergency cleanup")
        
        # Run emergency cleanup
        result = emergency_cleanup(target_free_mb=2)
        
        print(f"  Emergency cleanup result:")
        print(f"    Target: {result['target_mb']}MB")
        print(f"    Actual freed: {result['actual_freed_mb']:.2f}MB")
        print(f"    Success: {result['success']}")
        
        print("✅ Emergency cleanup test passed")
        return True
        
    except Exception as e:
        print(f"❌ Emergency cleanup test failed: {e}")
        return False

def test_config_persistence():
    """Test configuration saving and loading"""
    print("\n💾 Testing configuration persistence...")
    
    try:
        manager = StorageRetentionManager()
        
        # Add a custom rule
        custom_rule = RetentionRule(
            name="test_config_rule",
            path_pattern="test_pattern/*",
            max_age_hours=48,
            description="Test rule for config persistence"
        )
        
        manager.add_rule(custom_rule)
        
        # Save config
        manager.save_config()
        print("  Configuration saved")
        
        # Create new manager to test loading
        new_manager = StorageRetentionManager()
        
        # Check if custom rule was loaded
        rule_names = [rule.name for rule in new_manager.rules]
        
        if "test_config_rule" in rule_names:
            print("  ✅ Custom rule loaded successfully")
        else:
            print("  ⚠️  Custom rule not found in loaded config")
        
        # Clean up test rule
        new_manager.remove_rule("test_config_rule")
        new_manager.save_config()
        
        print("✅ Configuration persistence test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration persistence test failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files created during testing"""
    print("\n🧽 Cleaning up test files...")
    
    test_patterns = [
        'data/temp/*.tmp',
        'data/uploads/old_upload.mp4',
        'data/test_output/old_test.mp4',
        'logs/old.log',
        'config/storage_retention.json'
    ]
    
    cleaned_count = 0
    
    for pattern in test_patterns:
        try:
            for file_path in Path('.').glob(pattern):
                if file_path.exists():
                    file_path.unlink()
                    cleaned_count += 1
        except Exception as e:
            print(f"  Warning: Could not clean {pattern}: {e}")
    
    print(f"  Cleaned {cleaned_count} test files")

if __name__ == "__main__":
    print("🚀 Starting Storage Retention Tests")
    print("=" * 50)
    
    # Track test results
    test_results = []
    
    try:
        # Create test files
        create_test_files()
        
        # Run tests
        test_results.append(("Storage Stats", test_storage_stats()))
        test_results.append(("Retention Manager", test_retention_manager()))
        test_results.append(("Cleanup Operations", test_cleanup_operations()))
        test_results.append(("Emergency Cleanup", test_emergency_cleanup()))
        test_results.append(("Config Persistence", test_config_persistence()))
        
    finally:
        # Always clean up
        cleanup_test_files()
    
    # Print results summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    
    passed_count = 0
    total_count = len(test_results)
    
    for test_name, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if passed:
            passed_count += 1
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("🎉 All storage retention tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        exit(1)