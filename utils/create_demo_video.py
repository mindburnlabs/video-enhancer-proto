#!/usr/bin/env python3
"""
Create demo video for testing video enhancement.
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


import cv2
import numpy as np
import math
import os
from pathlib import Path

def create_demo_video(output_path: str, 
                     duration: int = 3, 
                     fps: int = 24, 
                     resolution: tuple = (320, 240),
                     add_artifacts: bool = True):
    """
    Create a demo video with various visual elements for testing enhancement.
    
    Args:
        output_path: Path to save the demo video
        duration: Duration in seconds
        fps: Frames per second
        resolution: Video resolution (width, height)
        add_artifacts: Whether to add compression artifacts and noise
    """
    width, height = resolution
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    print(f"🎬 Creating demo video: {output_path}")
    print(f"   Resolution: {width}x{height}")
    print(f"   Duration: {duration}s ({total_frames} frames)")
    print(f"   FPS: {fps}")
    print(f"   Add artifacts: {add_artifacts}")
    
    for frame_idx in range(total_frames):
        t = frame_idx / fps
        
        # Create base frame with gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Gradient background
        for y in range(height):
            for x in range(width):
                r = int(127 + 127 * math.sin(x * 0.02 + t))
                g = int(127 + 127 * math.cos(y * 0.02 + t * 1.5))
                b = int(127 + 127 * math.sin((x + y) * 0.01 + t * 0.5))
                frame[y, x] = [b, g, r]  # BGR format
        
        # Moving elements
        # Moving circle
        circle_x = int((math.sin(t * 2) * 0.3 + 0.5) * width)
        circle_y = int((math.cos(t * 1.5) * 0.3 + 0.5) * height)
        cv2.circle(frame, (circle_x, circle_y), 25, (0, 255, 255), -1)
        
        # Moving rectangle
        rect_x = int((math.cos(t * 3) * 0.25 + 0.75) * (width - 60))
        rect_y = int((math.sin(t * 2.5) * 0.25 + 0.25) * (height - 40))
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 60, rect_y + 40), (255, 0, 255), -1)
        
        # Rotating line
        center_x, center_y = width // 2, height // 2
        angle = t * 90  # degrees
        line_length = min(width, height) // 4
        end_x = int(center_x + line_length * math.cos(math.radians(angle)))
        end_y = int(center_y + line_length * math.sin(math.radians(angle)))
        cv2.line(frame, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3)
        
        # Text overlay
        text = f"Frame {frame_idx:03d} | Time {t:.2f}s"
        cv2.putText(frame, text, (10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add timestamp
        timestamp = f"Demo Video - t={t:.2f}"
        cv2.putText(frame, timestamp, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add artifacts if requested
        if add_artifacts:
            # Add noise
            noise = np.random.normal(0, 15, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Add compression-like artifacts (block effects)
            if frame_idx % 10 == 0:  # Every 10 frames
                # Simulate compression by downscaling and upscaling
                small = cv2.resize(frame, (width // 4, height // 4), interpolation=cv2.INTER_NEAREST)
                frame = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Add motion blur occasionally
            if frame_idx % 15 == 0:  # Every 15 frames
                kernel_size = 5
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[kernel_size // 2, :] = 1 / kernel_size  # Horizontal motion blur
                frame = cv2.filter2D(frame, -1, kernel)
        
        out.write(frame)
        
        # Progress
        if frame_idx % (total_frames // 10) == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"   Progress: {progress:.0f}%")
    
    out.release()
    print(f"✅ Demo video created: {output_path}")
    
    # Verify the file
    if Path(output_path).exists():
        file_size = Path(output_path).stat().st_size
        print(f"   File size: {file_size / 1024:.1f} KB")
        return True
    else:
        print(f"❌ Failed to create demo video")
        return False

def create_multiple_demo_videos():
    """Create multiple demo videos for different test scenarios."""
    base_dir = Path(__file__).parent.parent / "data" / "demo"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # High quality demo
    create_demo_video(
        str(base_dir / "demo_hq.mp4"),
        duration=3,
        fps=30,
        resolution=(640, 480),
        add_artifacts=False
    )
    
    # Low quality demo with artifacts
    create_demo_video(
        str(base_dir / "demo_lq.mp4"),
        duration=3,
        fps=24,
        resolution=(320, 240),
        add_artifacts=True
    )
    
    # Short test video
    create_demo_video(
        str(base_dir / "demo_short.mp4"),
        duration=2,
        fps=30,
        resolution=(480, 320),
        add_artifacts=True
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create demo videos for testing")
    parser.add_argument("--output", "-o", default="data/demo/demo.mp4", 
                       help="Output video path")
    parser.add_argument("--duration", "-d", type=int, default=3,
                       help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=24,
                       help="Frames per second")
    parser.add_argument("--width", type=int, default=320,
                       help="Video width")
    parser.add_argument("--height", type=int, default=240,
                       help="Video height")
    parser.add_argument("--no-artifacts", action="store_true",
                       help="Don't add compression artifacts and noise")
    parser.add_argument("--multiple", "-m", action="store_true",
                       help="Create multiple demo videos")
    
    args = parser.parse_args()
    
    if args.multiple:
        create_multiple_demo_videos()
    else:
        create_demo_video(
            args.output,
            duration=args.duration,
            fps=args.fps,
            resolution=(args.width, args.height),
            add_artifacts=not args.no_artifacts
        )