#!/usr/bin/env python3
"""
Performance Monitoring and Instrumentation

Comprehensive performance monitoring system for video enhancement pipeline
with per-strategy latency tracking, VRAM monitoring, and detailed metrics.
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


import time
import logging
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
from pathlib import Path
import asyncio
from contextlib import contextmanager
import torch

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation"""
    operation_id: str
    operation_type: str  # 'enhancement', 'analysis', 'api_request', etc.
    strategy: Optional[str]  # 'vsrm', 'seedvr2', etc.
    
    # Timing metrics
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    
    # Resource metrics
    peak_memory_mb: Optional[float] = None
    peak_gpu_memory_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    cpu_utilization_percent: Optional[float] = None
    
    # Processing metrics
    frames_processed: Optional[int] = None
    input_resolution: Optional[tuple] = None
    output_resolution: Optional[tuple] = None
    scale_factor: Optional[float] = None
    
    # Quality metrics
    quality_score: Optional[float] = None
    enhancement_factor: Optional[float] = None
    
    # Status
    success: bool = True
    error_message: Optional[str] = None
    
    # Custom metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def finish(self, success: bool = True, error_message: Optional[str] = None):
        """Mark the operation as finished and calculate duration"""
        self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time
        self.success = success
        self.error_message = error_message

class ResourceMonitor:
    """Monitor system resources during operations"""
    
    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self.monitoring = False
        self.samples = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return peak values"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        if not self.samples:
            return {}
        
        # Calculate peak values
        peak_memory = max(sample['memory_mb'] for sample in self.samples)
        peak_cpu = max(sample['cpu_percent'] for sample in self.samples)
        
        result = {
            'peak_memory_mb': peak_memory,
            'peak_cpu_percent': peak_cpu,
            'sample_count': len(self.samples)
        }
        
        # Add GPU metrics if available
        gpu_samples = [s for s in self.samples if 'gpu_memory_mb' in s]
        if gpu_samples:
            result['peak_gpu_memory_mb'] = max(s['gpu_memory_mb'] for s in gpu_samples)
            result['peak_gpu_utilization'] = max(s['gpu_utilization'] for s in gpu_samples)
        
        return result
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # System metrics
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent()
                
                sample = {
                    'timestamp': time.time(),
                    'memory_mb': memory_info.rss / (1024 * 1024),
                    'cpu_percent': cpu_percent
                }
                
                # GPU metrics if available
                if torch.cuda.is_available():
                    try:
                        gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                        gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                        sample.update({
                            'gpu_memory_mb': gpu_memory,
                            'gpu_utilization': gpu_utilization
                        })
                    except Exception as e:
                        logger.debug(f"GPU monitoring error: {e}")
                
                self.samples.append(sample)
                time.sleep(self.sample_interval)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                time.sleep(self.sample_interval)

class PerformanceTracker:
    """Main performance tracking system"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.active_operations: Dict[str, PerformanceMetrics] = {}
        self.resource_monitors: Dict[str, ResourceMonitor] = {}
        
        # Aggregated statistics
        self.stats_by_operation: Dict[str, Dict] = defaultdict(lambda: {
            'count': 0,
            'total_duration': 0,
            'avg_duration': 0,
            'min_duration': float('inf'),
            'max_duration': 0,
            'success_count': 0,
            'error_count': 0,
            'success_rate': 0.0
        })
        
        self.stats_by_strategy: Dict[str, Dict] = defaultdict(lambda: {
            'count': 0,
            'total_duration': 0,
            'avg_duration': 0,
            'frames_processed': 0,
            'avg_fps': 0,
            'peak_memory_mb': 0,
            'peak_gpu_memory_mb': 0
        })
        
        self._lock = threading.Lock()
    
    def start_operation(self, 
                       operation_type: str, 
                       strategy: Optional[str] = None,
                       metadata: Optional[Dict] = None) -> str:
        """Start tracking a new operation"""
        
        operation_id = f"{operation_type}_{int(time.time() * 1000000)}"
        
        metrics = PerformanceMetrics(
            operation_id=operation_id,
            operation_type=operation_type,
            strategy=strategy,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self.active_operations[operation_id] = metrics
            
            # Start resource monitoring
            monitor = ResourceMonitor()
            monitor.start_monitoring()
            self.resource_monitors[operation_id] = monitor
        
        logger.debug(f"Started tracking operation: {operation_id} ({operation_type}, {strategy})")
        return operation_id
    
    def update_operation(self, 
                        operation_id: str,
                        **kwargs):
        """Update metrics for an active operation"""
        
        with self._lock:
            if operation_id in self.active_operations:
                metrics = self.active_operations[operation_id]
                
                # Update fields
                for key, value in kwargs.items():
                    if hasattr(metrics, key):
                        setattr(metrics, key, value)
                    else:
                        metrics.metadata[key] = value
    
    def finish_operation(self, 
                        operation_id: str,
                        success: bool = True,
                        error_message: Optional[str] = None,
                        **final_metrics):
        """Finish tracking an operation"""
        
        with self._lock:
            if operation_id not in self.active_operations:
                logger.warning(f"Operation {operation_id} not found in active operations")
                return
            
            metrics = self.active_operations[operation_id]
            
            # Stop resource monitoring and get peak values
            if operation_id in self.resource_monitors:
                resource_peaks = self.resource_monitors[operation_id].stop_monitoring()
                del self.resource_monitors[operation_id]
                
                # Update metrics with resource data
                metrics.peak_memory_mb = resource_peaks.get('peak_memory_mb')
                metrics.peak_gpu_memory_mb = resource_peaks.get('peak_gpu_memory_mb')
                metrics.cpu_utilization_percent = resource_peaks.get('peak_cpu_percent')
                metrics.gpu_utilization_percent = resource_peaks.get('peak_gpu_utilization')
            
            # Apply final updates
            for key, value in final_metrics.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
                else:
                    metrics.metadata[key] = value
            
            # Finish the metrics
            metrics.finish(success, error_message)
            
            # Move to history
            self.metrics_history.append(metrics)
            del self.active_operations[operation_id]
            
            # Update aggregated stats
            self._update_stats(metrics)
        
        logger.debug(f"Finished tracking operation: {operation_id} "
                    f"({metrics.duration_seconds:.2f}s, success: {success})")
    
    def _update_stats(self, metrics: PerformanceMetrics):
        """Update aggregated statistics"""
        
        # Update operation type stats
        op_stats = self.stats_by_operation[metrics.operation_type]
        op_stats['count'] += 1
        op_stats['total_duration'] += metrics.duration_seconds or 0
        op_stats['avg_duration'] = op_stats['total_duration'] / op_stats['count']
        
        if metrics.duration_seconds:
            op_stats['min_duration'] = min(op_stats['min_duration'], metrics.duration_seconds)
            op_stats['max_duration'] = max(op_stats['max_duration'], metrics.duration_seconds)
        
        if metrics.success:
            op_stats['success_count'] += 1
        else:
            op_stats['error_count'] += 1
            
        op_stats['success_rate'] = op_stats['success_count'] / op_stats['count'] * 100
        
        # Update strategy stats if applicable
        if metrics.strategy:
            strat_stats = self.stats_by_strategy[metrics.strategy]
            strat_stats['count'] += 1
            strat_stats['total_duration'] += metrics.duration_seconds or 0
            strat_stats['avg_duration'] = strat_stats['total_duration'] / strat_stats['count']
            
            if metrics.frames_processed:
                strat_stats['frames_processed'] += metrics.frames_processed
                strat_stats['avg_fps'] = strat_stats['frames_processed'] / strat_stats['total_duration']
            
            if metrics.peak_memory_mb:
                strat_stats['peak_memory_mb'] = max(strat_stats['peak_memory_mb'], metrics.peak_memory_mb)
            
            if metrics.peak_gpu_memory_mb:
                strat_stats['peak_gpu_memory_mb'] = max(strat_stats['peak_gpu_memory_mb'], metrics.peak_gpu_memory_mb)
    
    @contextmanager
    def track_operation(self, 
                       operation_type: str, 
                       strategy: Optional[str] = None,
                       metadata: Optional[Dict] = None):
        """Context manager for tracking operations"""
        
        operation_id = self.start_operation(operation_type, strategy, metadata)
        
        try:
            yield operation_id
            self.finish_operation(operation_id, success=True)
        except Exception as e:
            self.finish_operation(operation_id, success=False, error_message=str(e))
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        with self._lock:
            current_time = time.time()
            
            # Recent metrics (last hour)
            recent_metrics = [
                m for m in self.metrics_history 
                if current_time - m.start_time < 3600
            ]
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'summary': {
                    'total_operations': len(self.metrics_history),
                    'recent_operations': len(recent_metrics),
                    'active_operations': len(self.active_operations),
                    'tracking_since': datetime.fromtimestamp(
                        min(m.start_time for m in self.metrics_history)
                    ).isoformat() if self.metrics_history else None
                },
                'by_operation_type': dict(self.stats_by_operation),
                'by_strategy': dict(self.stats_by_strategy),
                'recent_activity': self._get_recent_activity(),
                'performance_trends': self._get_performance_trends()
            }
    
    def _get_recent_activity(self) -> Dict[str, Any]:
        """Get recent activity summary"""
        
        if not self.metrics_history:
            return {}
        
        # Last 10 operations
        recent = list(self.metrics_history)[-10:]
        
        return {
            'last_10_operations': [
                {
                    'operation_type': m.operation_type,
                    'strategy': m.strategy,
                    'duration_seconds': m.duration_seconds,
                    'success': m.success,
                    'timestamp': datetime.fromtimestamp(m.start_time).isoformat()
                }
                for m in recent
            ]
        }
    
    def _get_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        
        if len(self.metrics_history) < 10:
            return {}
        
        # Calculate trends over recent operations
        recent_50 = list(self.metrics_history)[-50:]
        first_25 = recent_50[:25]
        last_25 = recent_50[25:]
        
        if not (first_25 and last_25):
            return {}
        
        avg_duration_first = sum(m.duration_seconds or 0 for m in first_25) / len(first_25)
        avg_duration_last = sum(m.duration_seconds or 0 for m in last_25) / len(last_25)
        
        success_rate_first = sum(1 for m in first_25 if m.success) / len(first_25) * 100
        success_rate_last = sum(1 for m in last_25 if m.success) / len(last_25) * 100
        
        return {
            'duration_trend': {
                'earlier_avg_seconds': avg_duration_first,
                'recent_avg_seconds': avg_duration_last,
                'change_percent': ((avg_duration_last - avg_duration_first) / avg_duration_first * 100)
                if avg_duration_first > 0 else 0
            },
            'success_rate_trend': {
                'earlier_success_rate': success_rate_first,
                'recent_success_rate': success_rate_last,
                'change_percent': success_rate_last - success_rate_first
            }
        }
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export metrics to file"""
        
        with self._lock:
            if format == 'json':
                data = {
                    'export_timestamp': datetime.utcnow().isoformat(),
                    'stats': self.get_stats(),
                    'metrics_history': [
                        asdict(m) for m in self.metrics_history
                    ]
                }
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            
            elif format == 'csv':
                import csv
                
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow([
                        'operation_id', 'operation_type', 'strategy', 
                        'start_time', 'duration_seconds', 'success',
                        'peak_memory_mb', 'peak_gpu_memory_mb', 
                        'frames_processed', 'quality_score'
                    ])
                    
                    # Write data
                    for m in self.metrics_history:
                        writer.writerow([
                            m.operation_id, m.operation_type, m.strategy,
                            datetime.fromtimestamp(m.start_time).isoformat(),
                            m.duration_seconds, m.success,
                            m.peak_memory_mb, m.peak_gpu_memory_mb,
                            m.frames_processed, m.quality_score
                        ])

# Global performance tracker instance
_global_tracker: Optional[PerformanceTracker] = None

def get_performance_tracker() -> PerformanceTracker:
    """Get or create the global performance tracker"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerformanceTracker()
    return _global_tracker

def track_enhancement_performance(strategy: str, metadata: Optional[Dict] = None):
    """Decorator for tracking enhancement performance"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            tracker = get_performance_tracker()
            
            with tracker.track_operation('enhancement', strategy, metadata) as operation_id:
                # Execute function
                result = func(*args, **kwargs)
                
                # Try to extract performance metrics from result
                if isinstance(result, dict):
                    if 'frames_processed' in result:
                        tracker.update_operation(operation_id, frames_processed=result['frames_processed'])
                    if 'quality_score' in result:
                        tracker.update_operation(operation_id, quality_score=result['quality_score'])
                    if 'input_resolution' in result:
                        tracker.update_operation(operation_id, input_resolution=result['input_resolution'])
                    if 'output_resolution' in result:
                        tracker.update_operation(operation_id, output_resolution=result['output_resolution'])
                
                return result
        return wrapper
    return decorator

# Convenience functions
def get_performance_stats() -> Dict[str, Any]:
    """Get current performance statistics"""
    return get_performance_tracker().get_stats()

def export_performance_metrics(filepath: str, format: str = 'json'):
    """Export performance metrics to file"""
    return get_performance_tracker().export_metrics(filepath, format)

def start_performance_tracking(operation_type: str, strategy: Optional[str] = None) -> str:
    """Start tracking a performance operation"""
    return get_performance_tracker().start_operation(operation_type, strategy)

def finish_performance_tracking(operation_id: str, success: bool = True, **metrics):
    """Finish tracking a performance operation"""
    return get_performance_tracker().finish_operation(operation_id, success, **metrics)

# Export main classes and functions
__all__ = [
    'PerformanceTracker', 'PerformanceMetrics', 'ResourceMonitor',
    'get_performance_tracker', 'track_enhancement_performance',
    'get_performance_stats', 'export_performance_metrics',
    'start_performance_tracking', 'finish_performance_tracking'
]