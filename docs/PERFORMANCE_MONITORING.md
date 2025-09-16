# Performance Instrumentation System

## Overview

This document describes the comprehensive performance monitoring and instrumentation system implemented for the video enhancement pipeline. The system provides detailed tracking of per-strategy latency, VRAM peaks, segment counts, and comprehensive metrics for production monitoring.

## Features

### Core Performance Tracking
- **Per-strategy latency tracking** - Monitor execution time for each enhancement strategy (VSRM, SeedVR2, DiTVR, etc.)
- **Resource monitoring** - Track VRAM usage peaks, CPU utilization, and memory consumption 
- **Processing metrics** - Count frames processed, measure quality scores, track resolution changes
- **Real-time monitoring** - Background thread monitors resource usage during operations
- **Error tracking** - Capture and report failures with detailed error context

### Integration Points
- **Model Handlers**: All enhancement models (VSRM, SeedVR2, etc.) include performance tracking decorators
- **Analysis Pipeline**: Degradation router tracks analysis performance and confidence scores
- **API Endpoints**: REST API requests are tracked from start to completion with full metrics
- **Background Tasks**: Asynchronous processing jobs include comprehensive performance logging

## Architecture

### Core Components

1. **PerformanceTracker**: Main tracking system that manages operations and statistics
2. **PerformanceMetrics**: Data structure holding metrics for individual operations
3. **ResourceMonitor**: Background thread that monitors system resource usage
4. **Decorators & Context Managers**: Easy integration patterns for existing code

### Data Flow

```
Operation Start → Resource Monitoring Begin → Processing → Resource Monitoring End → Statistics Update
```

## Usage Examples

### Using Decorators

```python
from utils.performance_monitor import track_enhancement_performance

@track_enhancement_performance('vsrm')
def enhance_video_vsrm(frames):
    # Processing logic here
    return {
        'enhanced_frames': result_frames,
        'frames_processed': len(frames),
        'quality_score': 0.85,
        'input_resolution': (720, 1280),
        'output_resolution': (1440, 2560)
    }
```

### Manual Tracking

```python
from utils.performance_monitor import get_performance_tracker

tracker = get_performance_tracker()
operation_id = tracker.start_operation('enhancement', 'seedvr2', {
    'video_path': input_path,
    'quality_tier': 'high'
})

try:
    # Processing work
    result = process_video(input_path)
    
    # Update with results
    tracker.update_operation(operation_id,
        frames_processed=result['frames'],
        quality_score=result['quality']
    )
    
    tracker.finish_operation(operation_id, success=True)
    
except Exception as e:
    tracker.finish_operation(operation_id, success=False, error_message=str(e))
    raise
```

### Context Manager

```python
from utils.performance_monitor import get_performance_tracker

tracker = get_performance_tracker()

with tracker.track_operation('analysis', 'degradation_router') as op_id:
    analysis_result = analyze_video(video_path)
    tracker.update_operation(op_id, 
        frames_processed=analysis_result['frame_count'],
        quality_score=analysis_result['confidence']
    )
```

## Metrics Collected

### Timing Metrics
- **start_time**: Operation start timestamp
- **end_time**: Operation completion timestamp  
- **duration_seconds**: Total processing duration

### Resource Metrics
- **peak_memory_mb**: Maximum RAM usage during operation
- **peak_gpu_memory_mb**: Maximum VRAM usage during operation
- **cpu_utilization_percent**: Peak CPU usage
- **gpu_utilization_percent**: Peak GPU usage

### Processing Metrics
- **frames_processed**: Number of video frames processed
- **input_resolution**: Input video resolution (width, height)
- **output_resolution**: Output video resolution (width, height)
- **scale_factor**: Upscaling factor applied

### Quality Metrics
- **quality_score**: Estimated output quality (0.0-1.0)
- **enhancement_factor**: Improvement factor over input

### Status Tracking
- **success**: Whether operation completed successfully
- **error_message**: Error details if operation failed
- **metadata**: Custom operation-specific data

## API Endpoints

### Get Performance Statistics
```
GET /performance/stats
```

Returns comprehensive performance statistics including:
- Summary metrics (total operations, success rates)
- Per-strategy statistics (average latency, throughput)
- Recent activity (last 10 operations)
- Performance trends over time

### Export Performance Data
```
POST /performance/export?format=json
POST /performance/export?format=csv
```

Exports complete performance metrics to downloadable file in JSON or CSV format.

### System Metrics (Enhanced)
```
GET /metrics
```

Enhanced system metrics endpoint now includes performance data summary when available.

## Statistics & Analytics

### Aggregated Statistics

The system maintains running statistics including:

- **Operation Type Statistics**: 
  - Total count, average duration, min/max duration
  - Success/failure rates
  
- **Strategy Statistics**:
  - Per-strategy performance (VSRM, SeedVR2, etc.)
  - Average processing speed (frames per second)
  - Peak resource usage by strategy
  
- **Performance Trends**:
  - Duration trends over recent operations
  - Success rate changes over time

### Real-time Monitoring

The system provides:
- Active operation tracking
- Real-time resource monitoring during processing
- Automatic cleanup of completed operations

## Integration Examples

### Model Handler Integration

All enhancement model handlers now include performance tracking:

```python
# In VSRMHandler.enhance_video()
@track_enhancement_performance('vsrm')
def enhance_video(self, input_path, output_path, **kwargs):
    # Processing logic
    return {
        'frames_processed': processed_count,
        'input_resolution': (height, width),
        'output_resolution': (out_height, out_width),
        'quality_score': estimated_quality
    }
```

### API Request Tracking

API endpoints track the complete request lifecycle:

```python
# In process_endpoints.py
tracker = get_performance_tracker()
perf_operation_id = tracker.start_operation('api_request', request.vsr_strategy, {
    'job_id': job_id,
    'latency_class': request.latency_class,
    'quality_tier': request.quality_tier
})

# Processing happens here...

# On completion
tracker.update_operation(perf_operation_id,
    frames_processed=result['frames_processed'],
    quality_score=result['quality_score']
)
tracker.finish_operation(perf_operation_id, success=True)
```

## Production Deployment

### Configuration

The performance monitoring system is designed to be:
- **Low overhead**: Minimal impact on processing performance
- **Thread-safe**: Safe for concurrent operations
- **Memory efficient**: Automatic cleanup of old metrics
- **Configurable**: Adjustable history limits and sampling rates

### Monitoring Integration

Performance data can be:
- Exported to external monitoring systems
- Stored in time-series databases
- Visualized with dashboards
- Alerted on performance degradation

### Resource Management

The system includes:
- **Automatic cleanup**: Old metrics are automatically purged
- **Memory limits**: Configurable maximum history size
- **Background processing**: Resource monitoring runs in separate threads
- **Graceful degradation**: Continues working even if monitoring fails

## Testing

A comprehensive test suite (`test_performance_monitor.py`) verifies:
- Basic tracking functionality
- Decorator integration
- Context manager usage
- Error handling
- Statistics collection
- Export functionality
- Integration with model handlers

## Best Practices

1. **Use Decorators** for simple function tracking
2. **Use Context Managers** for complex operations with multiple stages
3. **Include meaningful metadata** to enhance analysis capabilities
4. **Update operations** with processing results when available
5. **Handle errors properly** to maintain tracking integrity

## Future Enhancements

Potential improvements include:
- Integration with external APM tools (DataDog, New Relic)
- Real-time streaming metrics to monitoring dashboards
- Automated performance regression detection
- GPU memory profiling enhancements
- Custom alerting on performance thresholds

This performance instrumentation system provides comprehensive visibility into the video enhancement pipeline's performance characteristics, enabling data-driven optimization and reliable production monitoring.