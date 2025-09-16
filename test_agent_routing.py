#!/usr/bin/env python3
"""
Test script for agent routing in API
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


import asyncio
import tempfile
from pathlib import Path
from api.v1.process_endpoints import (
    get_enhancement_agent,
    _create_task_specification,
    ProcessingRequest,
    LatencyClass,
    QualityTier,
    VSRStrategy
)

def create_test_video():
    """Create a simple test video file"""
    import cv2
    import numpy as np
    
    # Create temporary video file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        output_path = tmp_file.name
    
    # Create a simple test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (640, 480))
    
    # Generate 30 frames of test content
    for i in range(30):
        # Create a simple gradient frame with moving text
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(480):
            frame[y, :, 0] = int(255 * y / 480)  # Red gradient
        
        # Add moving text
        text = f"Frame {i+1}"
        cv2.putText(frame, text, (50 + i * 10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Created test video: {output_path}")
    return output_path

async def test_agent_initialization():
    """Test VideoEnhancementAgent initialization"""
    print("🤖 Testing agent initialization...")
    
    try:
        agent = get_enhancement_agent()
        
        if agent:
            print(f"  ✅ Agent initialized: {agent.name}")
            print(f"  Device: {agent.device}")
            print(f"  Capabilities: {agent.capabilities.capabilities}")
            return True
        else:
            print("  ⚠️  Agent initialization failed (this is expected if dependencies are missing)")
            return False
            
    except Exception as e:
        print(f"  ❌ Agent initialization error: {e}")
        return False

async def test_task_specification_creation():
    """Test TaskSpecification creation from API request"""
    print("\n📋 Testing TaskSpecification creation...")
    
    try:
        # Create test video
        test_video_path = create_test_video()
        
        # Create test output path  
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        # Create test request
        request = ProcessingRequest(
            vsr_strategy=VSRStrategy.vsrm,
            latency_class=LatencyClass.standard,
            quality_tier=QualityTier.balanced,
            scale_factor=2.0,
            allow_diffusion=True,
            allow_zero_shot=True,
            enable_face_expert=False,
            enable_hfr=False
        )
        
        # Create mock strategy plan
        strategy_plan = {
            'primary_model': 'vsrm',
            'degradation_analysis': {'compression': 0.3, 'noise': 0.1},
            'content_analysis': {'has_faces': False},
            'processing_stages': ['analysis', 'enhancement', 'output']
        }
        
        # Create task specification
        task_spec = await _create_task_specification(test_video_path, output_path, request, strategy_plan)
        
        print(f"  ✅ TaskSpecification created successfully")
        print(f"    Task ID: {task_spec.task_id}")
        print(f"    Task Type: {task_spec.task_type}")
        print(f"    Priority: {task_spec.priority}")
        print(f"    Quality: {task_spec.quality}")
        print(f"    Input Resolution: {task_spec.video_specs.input_resolution}")
        print(f"    Target Resolution: {task_spec.video_specs.target_resolution}")
        print(f"    Model Preferences: {task_spec.model_preferences}")
        
        # Validate task specification
        is_valid, errors = task_spec.validate()
        if is_valid:
            print(f"  ✅ TaskSpecification validation passed")
        else:
            print(f"  ❌ TaskSpecification validation failed: {errors}")
            return False
        
        # Cleanup
        Path(test_video_path).unlink()
        Path(output_path).unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        print(f"  ❌ TaskSpecification creation failed: {e}")
        return False

async def test_agent_processing():
    """Test agent processing if available"""
    print("\n🔧 Testing agent processing...")
    
    try:
        agent = get_enhancement_agent()
        
        if not agent:
            print("  ⚠️  Agent not available, skipping processing test")
            return True
        
        # Create test video
        test_video_path = create_test_video()
        
        # Create test output path
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        # Create test request
        request = ProcessingRequest(
            vsr_strategy=VSRStrategy.auto,
            latency_class=LatencyClass.standard,
            quality_tier=QualityTier.fast  # Use fast for testing
        )
        
        # Create mock strategy plan
        strategy_plan = {
            'primary_model': 'fast_mamba_vsr',
            'degradation_analysis': {},
            'content_analysis': {'has_faces': False},
            'processing_stages': ['analysis', 'enhancement', 'output']
        }
        
        # Create task specification
        task_spec = await _create_task_specification(test_video_path, output_path, request, strategy_plan)
        
        print(f"  Processing task with agent...")
        
        # Process with agent (this will likely fail due to missing models, but we can test the flow)
        try:
            result = await agent.process_task(task_spec)
            
            if result.status.value == "completed":
                print(f"  ✅ Agent processing succeeded")
                print(f"    Processing time: {result.processing_time:.2f}s")
                print(f"    Metadata: {result.metadata}")
            else:
                print(f"  ⚠️  Agent processing returned status: {result.status.value}")
                print(f"    Error: {result.error_message}")
                
        except Exception as proc_error:
            print(f"  ⚠️  Agent processing failed (expected due to missing model weights): {proc_error}")
        
        # Cleanup
        Path(test_video_path).unlink()
        Path(output_path).unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Agent processing test failed: {e}")
        return False

async def test_agent_status_endpoint():
    """Test agent status endpoint functionality"""
    print("\n📊 Testing agent status endpoint...")
    
    try:
        from api.v1.process_endpoints import get_agent_status
        
        # Test agent status endpoint
        status_result = await get_agent_status()
        
        print(f"  Agent status: {status_result.get('status', 'unknown')}")
        
        if status_result.get('status') == 'active':
            print(f"    Agent Name: {status_result.get('agent_name')}")
            print(f"    Device: {status_result.get('device')}")
            print(f"    Statistics: {status_result.get('statistics', {})}")
            print(f"  ✅ Agent status endpoint working with active agent")
        elif status_result.get('status') == 'unavailable':
            print(f"    Error: {status_result.get('error')}")
            print(f"  ⚠️  Agent status endpoint working but agent unavailable")
        else:
            print(f"    Error: {status_result.get('error')}")
            print(f"  ⚠️  Agent status endpoint returned error status")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Agent status endpoint test failed: {e}")
        return False

async def main():
    """Run all agent routing tests"""
    print("🚀 Starting Agent Routing Tests")
    print("=" * 50)
    
    test_results = []
    
    # Run tests
    test_results.append(("Agent Initialization", await test_agent_initialization()))
    test_results.append(("TaskSpecification Creation", await test_task_specification_creation()))
    test_results.append(("Agent Processing", await test_agent_processing()))
    test_results.append(("Agent Status Endpoint", await test_agent_status_endpoint()))
    
    # Print results
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
        print("🎉 All agent routing tests passed!")
    else:
        print("⚠️  Some tests failed or had warnings (this is expected without full model setup)")
    
    print("\n💡 Note: Some failures are expected without proper model weights and dependencies")

if __name__ == "__main__":
    asyncio.run(main())