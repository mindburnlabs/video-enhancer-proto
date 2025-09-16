
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
import subprocess
import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

class VideoUtils:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract comprehensive video metadata using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"FFprobe failed: {result.stderr}")
                return self._get_basic_metadata_opencv(video_path)
            
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                return self._get_basic_metadata_opencv(video_path)
            
            format_info = data.get('format', {})
            
            metadata = {
                'duration': float(format_info.get('duration', 0)),
                'bitrate': int(format_info.get('bit_rate', 0)),
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'fps': self._parse_fps(video_stream.get('r_frame_rate', '30/1')),
                'codec': video_stream.get('codec_name', 'unknown'),
                'format': format_info.get('format_name', 'unknown'),
                'file_size': int(format_info.get('size', 0)),
                'pixel_format': video_stream.get('pix_fmt', 'unknown')
            }
            
            # Calculate additional metrics
            if metadata['duration'] > 0:
                metadata['total_frames'] = int(metadata['fps'] * metadata['duration'])
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {e}")
            return self._get_basic_metadata_opencv(video_path)
    
    def _parse_fps(self, fps_string: str) -> float:
        """Parse fps from fractional string like '30/1' or '30000/1001'"""
        try:
            if '/' in fps_string:
                num, den = fps_string.split('/')
                return float(num) / float(den)
            return float(fps_string)
        except:
            return 30.0
    
    def _get_basic_metadata_opencv(self, video_path: str) -> Dict[str, Any]:
        """Fallback metadata extraction using OpenCV"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            metadata = {
                'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'codec': 'unknown',
                'format': 'unknown',
                'bitrate': 0,
                'file_size': os.path.getsize(video_path) if os.path.exists(video_path) else 0
            }
            
            cap.release()
            return metadata
            
        except Exception as e:
            self.logger.error(f"OpenCV metadata extraction failed: {e}")
            return {
                'duration': 0, 'width': 0, 'height': 0, 'fps': 30,
                'total_frames': 0, 'codec': 'unknown', 'format': 'unknown',
                'bitrate': 0, 'file_size': 0
            }
    
    def extract_audio(self, video_path: str, audio_path: str, sample_rate: int = 16000) -> bool:
        """Extract audio from video file"""
        try:
            cmd = [
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(sample_rate), '-ac', '1', '-y', audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Audio extracted successfully: {audio_path}")
                return True
            else:
                self.logger.error(f"Audio extraction failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error extracting audio: {e}")
            return False
    
    def extract_clip(self, input_video: str, output_path: str, 
                    start_time: float, end_time: float) -> bool:
        """Extract a clip from video with precise timing"""
        try:
            duration = end_time - start_time
            
            cmd = [
                'ffmpeg', '-ss', str(start_time), '-i', input_video,
                '-t', str(duration), '-c', 'copy', '-avoid_negative_ts', 'make_zero',
                '-y', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Clip extracted: {start_time}s-{end_time}s -> {output_path}")
                return True
            else:
                self.logger.error(f"Clip extraction failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error extracting clip: {e}")
            return False
    
    def resize_video(self, input_path: str, output_path: str, 
                    width: int, height: int, maintain_aspect: bool = True) -> bool:
        """Resize video with optional aspect ratio preservation"""
        try:
            if maintain_aspect:
                scale_filter = f"scale={width}:{height}:force_original_aspect_ratio=decrease"
            else:
                scale_filter = f"scale={width}:{height}"
            
            cmd = [
                'ffmpeg', '-i', input_path, '-vf', scale_filter,
                '-c:a', 'copy', '-y', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Video resized: {width}x{height} -> {output_path}")
                return True
            else:
                self.logger.error(f"Video resize failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error resizing video: {e}")
            return False
    
    def combine_videos(self, video_list: list, output_path: str) -> bool:
        """Combine multiple videos into a single file"""
        try:
            # Create temporary file list
            with open('/tmp/video_list.txt', 'w') as f:
                for video_path in video_list:
                    f.write(f"file '{video_path}'\n")
            
            cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0', '-i', '/tmp/video_list.txt',
                '-c', 'copy', '-y', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Cleanup
            os.remove('/tmp/video_list.txt')
            
            if result.returncode == 0:
                self.logger.info(f"Videos combined successfully: {output_path}")
                return True
            else:
                self.logger.error(f"Video combination failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error combining videos: {e}")
            return False
    
    def add_subtitles(self, video_path: str, subtitle_path: str, output_path: str) -> bool:
        """Add subtitles to video"""
        try:
            cmd = [
                'ffmpeg', '-i', video_path, '-i', subtitle_path,
                '-c:v', 'copy', '-c:a', 'copy', '-c:s', 'mov_text',
                '-y', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Subtitles added successfully: {output_path}")
                return True
            else:
                self.logger.error(f"Subtitle addition failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding subtitles: {e}")
            return False
    
    def convert_format(self, input_path: str, output_path: str, 
                      codec: str = 'libx264', crf: int = 23) -> bool:
        """Convert video to different format with quality control"""
        try:
            cmd = [
                'ffmpeg', '-i', input_path, '-c:v', codec, '-crf', str(crf),
                '-preset', 'medium', '-c:a', 'aac', '-b:a', '128k',
                '-y', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Format converted successfully: {output_path}")
                return True
            else:
                self.logger.error(f"Format conversion failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error converting format: {e}")
            return False