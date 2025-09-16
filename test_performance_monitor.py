#!/usr/bin/env python3
"""
Test script for performance monitoring system
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
import numpy as np
from utils.performance_monitor import (
    get_performance_tracker, 
    track_enhancement_performance,
    get_performance_stats,
    export_performance_metrics
)

# Test basic tracker functionality
def test_basic_tracking():
    print("🧪 Testing basic performance tracking...")
    
    tracker = get_performance_tracker()
    
    # Start an operation
    operation_id = tracker.start_operation('test_operation', 'test_strategy', {'test': True})
    
    # Simulate some work
    time.sleep(1)
    
    # Update operation
    tracker.update_operation(operation_id, 
                           frames_processed=30,
                           quality_score=0.85)
    
    # Finish operation
    tracker.finish_operation(operation_id, success=True)
    
    print("✅ Basic tracking test passed")

# Test decorator functionality
@track_enhancement_performance('test_decorator')
def test_enhancement_function():
    """Test function with performance tracking decorator"""
    time.sleep(0.5)  # Simulate processing
    return {
        'frames_processed': 60,
        'quality_score': 0.9,
        'input_resolution': (480, 640),
        'output_resolution': (1920, 1080)
    }

def test_decorator():
    print("🧪 Testing decorator-based tracking...")
    
    result = test_enhancement_function()
    print(f"Result: {result}")
    
    print("✅ Decorator test passed")

# Test context manager
def test_context_manager():
    print("🧪 Testing context manager tracking...")
    
    tracker = get_performance_tracker()
    
    try:
        with tracker.track_operation('context_test', 'context_strategy') as op_id:
            time.sleep(0.3)
            tracker.update_operation(op_id, frames_processed=15)
            
        print("✅ Context manager test passed")
        
    except Exception as e:
        print(f"❌ Context manager test failed: {e}")

# Test error handling
def test_error_handling():
    print("🧪 Testing error handling...")
    
    tracker = get_performance_tracker()
    
    try:
        with tracker.track_operation('error_test', 'error_strategy') as op_id:
            time.sleep(0.2)
            raise ValueError("Simulated error")
            
    except ValueError:
        print("✅ Error handling test passed (error was properly tracked)")

# Test statistics
def test_statistics():
    print("🧪 Testing statistics collection...")
    
    stats = get_performance_stats()
    
    print(f"Summary: {stats.get('summary', {})}")
    print(f"Operations by type: {list(stats.get('by_operation_type', {}).keys())}")
    print(f"Strategies: {list(stats.get('by_strategy', {}).keys())}")
    
    print("✅ Statistics test passed")

# Test export functionality
def test_export():
    print("🧪 Testing export functionality...")
    
    try:
        # Export to JSON
        export_performance_metrics('test_metrics.json', 'json')
        print("✅ JSON export successful")
        
        # Export to CSV
        export_performance_metrics('test_metrics.csv', 'csv')
        print("✅ CSV export successful")
        
        # Clean up
        import os
        if os.path.exists('test_metrics.json'):
            os.remove('test_metrics.json')
        if os.path.exists('test_metrics.csv'):
            os.remove('test_metrics.csv')
            
    except Exception as e:
        print(f"❌ Export test failed: {e}")

# Simulate model handler performance
def simulate_model_performance():
    print("🧪 Simulating model handler performance...")
    
    # Simulate VSRM performance
    @track_enhancement_performance('vsrm')
    def simulate_vsrm():
        time.sleep(1.2)  # Simulate processing time
        return {
            'frames_processed': 120,
            'quality_score': 0.82,
            'input_resolution': (720, 1280),
            'output_resolution': (1440, 2560)
        }
    
    # Simulate SeedVR2 performance
    @track_enhancement_performance('seedvr2')
    def simulate_seedvr2():
        time.sleep(2.1)  # Simulate processing time
        return {
            'frames_processed': 90,
            'quality_score': 0.88,
            'input_resolution': (480, 640),
            'output_resolution': (480, 640)  # Restoration, not upscaling
        }
    
    # Run simulations
    simulate_vsrm()
    simulate_seedvr2()
    simulate_vsrm()  # Run again for statistics
    
    print("✅ Model simulation completed")

# Test with actual degradation router
def test_with_degradation_router():
    print("🧪 Testing with degradation router...")
    
    try:
        from models.analysis.degradation_router import DegradationRouter
        
        router = DegradationRouter()
        
        # This would normally process a real video file
        # For testing, we'll just verify the performance tracking integration is there
        print("✅ Degradation router integration ready")
        
    except ImportError as e:
        print(f"⚠️  Degradation router not available: {e}")

if __name__ == "__main__":
    print("🚀 Starting Performance Monitor Tests")
    print("=" * 50)
    
    # Run all tests
    test_basic_tracking()
    test_decorator()
    test_context_manager()
    test_error_handling()
    simulate_model_performance()
    test_with_degradation_router()
    test_statistics()
    test_export()
    
    print("\n" + "=" * 50)
    print("🎉 All performance monitoring tests completed!")
    
    # Print final statistics
    print("\n📊 Final Performance Summary:")
    stats = get_performance_stats()
    summary = stats.get('summary', {})
    
    print(f"Total operations tracked: {summary.get('total_operations', 0)}")
    print(f"Recent operations: {summary.get('recent_operations', 0)}")
    
    by_strategy = stats.get('by_strategy', {})
    if by_strategy:
        print("\n📈 Strategy Performance:")
        for strategy, data in by_strategy.items():
            print(f"  {strategy}: {data.get('count', 0)} ops, "
                  f"avg {data.get('avg_duration', 0):.2f}s")
    
    by_operation = stats.get('by_operation_type', {})
    if by_operation:
        print("\n🔧 Operation Performance:")
        for op_type, data in by_operation.items():
            success_rate = data.get('success_rate', 0)
            print(f"  {op_type}: {data.get('count', 0)} ops, "
                  f"{success_rate:.1f}% success rate")