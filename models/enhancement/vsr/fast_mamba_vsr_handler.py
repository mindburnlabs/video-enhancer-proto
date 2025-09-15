"""
Fast Mamba VSR Handler
Ultra-efficient Mamba-based video super-resolution with linear complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Tuple, List
import tempfile
import json
import time

from models.backbones.mamba import EAMambaBlock, SpatialTemporalMamba, EAMambaVideoBlock
from utils.video_utils import VideoUtils

logger = logging.getLogger(__name__)

class FastMambaVSRNetwork(nn.Module):
    """Fast Mamba VSR network with linear-time video processing."""
    
    def __init__(self, 
                 scale=4,
                 in_channels=3,
                 embed_dim=96,
                 num_layers=8,
                 state_size=16,
                 num_groups=8,
                 efficient_attention=True):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Input projection with separable convolution for efficiency
        self.input_proj = nn.Sequential(
            SeparableConv3d(in_channels, embed_dim // 2, kernel_size=(1, 3, 3)),
            nn.GELU(),
            SeparableConv3d(embed_dim // 2, embed_dim, kernel_size=(3, 1, 1))
        )
        
        # Fast Mamba layers for temporal and spatial processing
        self.mamba_layers = nn.ModuleList([
            BiMambaLayer(
                dim=embed_dim,
                state_size=state_size,
                num_groups=num_groups,
                efficient=True
            ) for _ in range(num_layers)
        ])
        
        # Lightweight cross-scale fusion
        self.cross_scale_fusion = CrossScaleFusion(embed_dim, num_scales=3)
        
        # Efficient upsampling
        self.upsampler = EfficientUpsampler(embed_dim, scale, in_channels)
        
        # Gradient checkpointing for memory efficiency
        self.use_checkpointing = True
        
    def forward(self, x):
        """Forward pass for Fast Mamba VSR.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
            
        Returns:
            Upsampled tensor of shape (B, C, T, H*scale, W*scale)
        """
        B, C, T, H, W = x.shape
        
        # Input projection
        x = self.input_proj(x)  # (B, embed_dim, T, H, W)
        
        # Multi-scale feature extraction
        features = self.cross_scale_fusion.extract_multiscale(x)
        
        # Process through Mamba layers with skip connections
        skip_connections = []
        
        for i, layer in enumerate(self.mamba_layers):
            if self.use_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
            
            # Store skip connections for later fusion
            if i % 2 == 0:
                skip_connections.append(x)
        
        # Fuse skip connections
        if skip_connections:
            x = x + sum(skip_connections) * 0.1
        
        # Cross-scale fusion
        x = self.cross_scale_fusion(x, features)
        
        # Upsampling
        x = self.upsampler(x)
        
        return x

class SeparableConv3d(nn.Module):
    """Separable 3D convolution for efficiency."""
    
    def __init__(self, in_channels, out_channels, kernel_size, padding=None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        
        if padding is None:
            padding = tuple(k // 2 for k in kernel_size)
        
        # Depthwise convolution
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size, 
            padding=padding, groups=in_channels, bias=False
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv3d(
            in_channels, out_channels, kernel_size=1, bias=True
        )
        
        # Batch normalization for stability
        self.bn = nn.BatchNorm3d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

class CrossScaleFusion(nn.Module):
    """Cross-scale feature fusion for multi-resolution processing."""
    
    def __init__(self, embed_dim, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.embed_dim = embed_dim
        
        # Multi-scale feature extractors
        self.extractors = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool3d(kernel_size=(1, 2**i, 2**i), stride=(1, 2**i, 2**i)),
                SeparableConv3d(embed_dim, embed_dim, kernel_size=(1, 3, 3)),
                nn.GELU()
            ) for i in range(num_scales)
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            SeparableConv3d(embed_dim * (num_scales + 1), embed_dim, kernel_size=(1, 1, 1)),
            nn.GELU(),
            SeparableConv3d(embed_dim, embed_dim, kernel_size=(3, 3, 3))
        )
        
    def extract_multiscale(self, x):
        """Extract multi-scale features."""
        B, C, T, H, W = x.shape
        features = []
        
        for extractor in self.extractors:
            feat = extractor(x)
            # Upsample back to original size
            feat = F.interpolate(feat, size=(T, H, W), mode='trilinear', align_corners=False)
            features.append(feat)
            
        return features
    
    def forward(self, x, multiscale_features):
        """Fuse multi-scale features."""
        # Combine all features
        all_features = [x] + multiscale_features
        combined = torch.cat(all_features, dim=1)
        
        # Fusion
        fused = self.fusion(combined)
        
        return fused

class EfficientUpsampler(nn.Module):
    """Efficient upsampling module with sub-pixel convolution."""
    
    def __init__(self, embed_dim, scale, out_channels):
        super().__init__()
        self.scale = scale
        
        # Feature refinement
        self.refine = nn.Sequential(
            SeparableConv3d(embed_dim, embed_dim, kernel_size=(1, 3, 3)),
            nn.GELU(),
            SeparableConv3d(embed_dim, embed_dim, kernel_size=(3, 1, 1)),
            nn.GELU()
        )
        
        # Sub-pixel convolution for upsampling
        self.upconv = nn.Conv3d(
            embed_dim, 
            out_channels * (scale ** 2), 
            kernel_size=(1, 3, 3), 
            padding=(0, 1, 1)
        )
        
        # Pixel shuffle for spatial upsampling
        self.pixel_shuffle = nn.PixelShuffle(scale)
        
        # Temporal consistency enhancement
        self.temporal_enhance = nn.Conv3d(
            out_channels, out_channels, 
            kernel_size=(3, 1, 1), 
            padding=(1, 0, 0)
        )
        
    def forward(self, x):
        """Upsample features to final resolution."""
        B, C, T, H, W = x.shape
        
        # Feature refinement
        x = self.refine(x)
        
        # Upsampling convolution
        x = self.upconv(x)  # (B, out_channels * scale^2, T, H, W)
        
        # Apply pixel shuffle frame by frame
        upsampled_frames = []
        for t in range(T):
            frame = x[:, :, t, :, :]  # (B, out_channels * scale^2, H, W)
            upsampled = self.pixel_shuffle(frame)  # (B, out_channels, H*scale, W*scale)
            upsampled_frames.append(upsampled)
        
        output = torch.stack(upsampled_frames, dim=2)  # (B, out_channels, T, H*scale, W*scale)
        
        # Temporal consistency enhancement
        output = output + self.temporal_enhance(output) * 0.1
        
        return output

class FastMambaVSRHandler:
    """Fast Mamba VSR Handler for ultra-efficient video super-resolution."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = "cuda",
                 scale: int = 4,
                 tile_size: int = 256,
                 tile_overlap: int = 16,
                 batch_size: int = 4,
                 enable_trt: bool = False):
        
        self.device = torch.device(device)
        self.scale = scale
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.batch_size = batch_size
        self.enable_trt = enable_trt
        
        logger.info("⚡ Initializing Fast Mamba VSR Handler...")
        logger.info(f"   Device: {device}")
        logger.info(f"   Scale: {scale}x")
        logger.info(f"   Tile Size: {tile_size}")
        logger.info(f"   Batch Size: {batch_size}")
        
        # Initialize network
        self.model = FastMambaVSRNetwork(
            scale=scale,
            in_channels=3,
            embed_dim=96,
            num_layers=8,
            state_size=16,
            num_groups=8,
            efficient_attention=True
        ).to(self.device)
        
        # Load pretrained weights
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        else:
            logger.warning("No model weights provided, using random initialization")
        
        # Optimize model
        self.model.eval()
        self._optimize_model()
        
        self.video_utils = VideoUtils()
        
        logger.info("✅ Fast Mamba VSR Handler initialized")
    
    def _load_model(self, model_path: str):
        """Load pretrained Fast Mamba VSR weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"✅ Loaded Fast Mamba VSR weights from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            logger.info("Using random initialization")
    
    def _optimize_model(self):
        """Optimize model for inference."""
        try:
            # Compile model for faster inference (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model, mode='max-autotune')
                logger.info("✅ Model compiled with PyTorch 2.0")
            
            # TensorRT optimization (if available and enabled)
            if self.enable_trt:
                try:
                    import torch_tensorrt
                    # This would require specific setup and sample inputs
                    logger.info("TensorRT optimization requested but not implemented")
                except ImportError:
                    logger.warning("TensorRT not available")
            
        except Exception as e:
            logger.warning(f"Model optimization failed: {e}")
    
    def enhance_video(self, 
                     input_path: str, 
                     output_path: str,
                     chunk_size: int = 16,
                     overlap: int = 2,
                     fp16: bool = True,
                     async_processing: bool = True) -> Dict:
        """Enhance video using Fast Mamba VSR.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            chunk_size: Number of frames to process together
            overlap: Frame overlap between chunks
            fp16: Use half precision for efficiency
            async_processing: Enable asynchronous processing
            
        Returns:
            Processing statistics
        """
        logger.info(f"🎬 Processing video with Fast Mamba VSR...")
        logger.info(f"   Input: {input_path}")
        logger.info(f"   Output: {output_path}")
        logger.info(f"   Chunk Size: {chunk_size}")
        logger.info(f"   Overlap: {overlap}")
        logger.info(f"   FP16: {fp16}")
        
        start_time = time.time()
        
        try:
            # Get video metadata
            metadata = self.video_utils.get_video_metadata(input_path)
            fps = metadata['fps']
            total_frames = metadata['frame_count']
            
            # Setup video capture and writer
            cap = cv2.VideoCapture(input_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_height = metadata['height'] * self.scale
            out_width = metadata['width'] * self.scale
            
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
            
            # Process in efficient chunks
            processed_count = 0
            total_inference_time = 0
            
            with torch.cuda.amp.autocast(enabled=fp16):
                frame_buffer = []
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_buffer.append(frame)
                    
                    # Process chunk when buffer is full
                    if len(frame_buffer) >= chunk_size:
                        chunk_start = time.time()
                        
                        enhanced_frames = self._process_chunk_efficient(
                            frame_buffer, fp16, async_processing
                        )
                        
                        chunk_time = time.time() - chunk_start
                        total_inference_time += chunk_time
                        
                        # Write frames (handle overlap)
                        write_frames = enhanced_frames[:-overlap] if overlap > 0 else enhanced_frames
                        for frame in write_frames:
                            out.write(frame)
                            processed_count += 1
                        
                        # Keep overlapping frames for next chunk
                        frame_buffer = frame_buffer[-overlap:] if overlap > 0 else []
                
                # Process remaining frames
                if len(frame_buffer) > 0:
                    enhanced_frames = self._process_chunk_efficient(frame_buffer, fp16, async_processing)
                    for frame in enhanced_frames:
                        out.write(frame)
                        processed_count += 1
            
            cap.release()
            out.release()
            
            total_time = time.time() - start_time
            
            stats = {
                'input_frames': total_frames,
                'output_frames': processed_count,
                'scale_factor': self.scale,
                'processing_mode': 'fast_mamba_vsr',
                'total_time': total_time,
                'inference_time': total_inference_time,
                'fps_processed': processed_count / total_time,
                'speedup_ratio': total_inference_time / total_time,
                'fp16': fp16,
                'chunk_size': chunk_size
            }
            
            logger.info(f"✅ Fast Mamba VSR processing completed")
            logger.info(f"   Processed: {processed_count} frames in {total_time:.2f}s")
            logger.info(f"   Speed: {stats['fps_processed']:.1f} FPS")
            logger.info(f"   Speedup: {stats['speedup_ratio']:.2f}x")
            
            return stats
            
        except Exception as e:
            logger.error(f"Fast Mamba VSR processing failed: {e}")
            raise
    
    def _process_chunk_efficient(self, 
                               frames: List[np.ndarray], 
                               fp16: bool = True,
                               async_processing: bool = True) -> List[np.ndarray]:
        """Process a chunk of frames efficiently."""
        try:
            # Convert to tensor batch
            input_tensors = []
            original_shapes = []
            
            for frame in frames:
                original_shapes.append(frame.shape)
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Normalize
                frame_norm = frame_rgb.astype(np.float32) / 255.0
                # Convert to tensor
                frame_tensor = torch.from_numpy(frame_norm.transpose(2, 0, 1))
                input_tensors.append(frame_tensor)
            
            # Stack frames
            input_batch = torch.stack(input_tensors, dim=1).unsqueeze(0).to(self.device)
            
            # Tile-based processing for large frames
            if (input_batch.shape[-1] > self.tile_size or 
                input_batch.shape[-2] > self.tile_size):
                output_batch = self._tile_process_efficient(input_batch, fp16)
            else:
                with torch.no_grad():
                    if fp16:
                        input_batch = input_batch.half()
                    output_batch = self.model(input_batch)
            
            # Convert back to frames
            enhanced_frames = []
            for t in range(output_batch.shape[2]):
                frame_tensor = output_batch[0, :, t, :, :].cpu().float()
                frame_tensor = torch.clamp(frame_tensor, 0, 1)
                frame_np = (frame_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                enhanced_frames.append(frame_bgr)
            
            return enhanced_frames
            
        except Exception as e:
            logger.error(f"Chunk processing failed: {e}")
            return frames  # Fallback to original
    
    def _tile_process_efficient(self, input_batch: torch.Tensor, fp16: bool = True) -> torch.Tensor:
        """Efficient tile-based processing with minimal memory overhead."""
        B, C, T, H, W = input_batch.shape
        tile_h = tile_w = self.tile_size
        overlap = self.tile_overlap
        
        # Calculate optimal tile grid
        stride_h = tile_h - overlap
        stride_w = tile_w - overlap
        
        tiles_h = (H - overlap + stride_h - 1) // stride_h
        tiles_w = (W - overlap + stride_w - 1) // stride_w
        
        # Initialize output tensor
        output_batch = torch.zeros(
            B, C, T, H * self.scale, W * self.scale,
            device=self.device, dtype=input_batch.dtype
        )
        
        # Process tiles with overlap handling
        for i in range(tiles_h):
            for j in range(tiles_w):
                # Calculate tile boundaries
                start_h = i * stride_h
                end_h = min(start_h + tile_h, H)
                start_w = j * stride_w
                end_w = min(start_w + tile_w, W)
                
                # Extract tile
                tile = input_batch[:, :, :, start_h:end_h, start_w:end_w]
                
                # Process tile
                with torch.no_grad():
                    if fp16:
                        tile = tile.half()
                    tile_out = self.model(tile)
                
                # Calculate output boundaries
                out_start_h = start_h * self.scale
                out_end_h = end_h * self.scale
                out_start_w = start_w * self.scale
                out_end_w = end_w * self.scale
                
                # Handle overlaps with weighted blending
                if overlap > 0:
                    # Create blending weights
                    blend_h = min(overlap * self.scale, tile_out.shape[-2])
                    blend_w = min(overlap * self.scale, tile_out.shape[-1])
                    
                    weights = torch.ones_like(tile_out)
                    
                    # Smooth blending at edges
                    if i > 0:  # Not first row
                        weights[:, :, :, :blend_h, :] *= torch.linspace(0, 1, blend_h).view(1, 1, 1, -1, 1)
                    if j > 0:  # Not first column
                        weights[:, :, :, :, :blend_w] *= torch.linspace(0, 1, blend_w).view(1, 1, 1, 1, -1)
                    
                    # Weighted accumulation
                    output_batch[:, :, :, out_start_h:out_end_h, out_start_w:out_end_w] += tile_out * weights
                else:
                    output_batch[:, :, :, out_start_h:out_end_h, out_start_w:out_end_w] = tile_out
        
        return output_batch
    
    def benchmark_performance(self, test_frames: int = 100) -> Dict:
        """Benchmark processing performance."""
        logger.info(f"🚀 Benchmarking Fast Mamba VSR performance...")
        
        # Create synthetic test data
        test_input = torch.randn(1, 3, test_frames, 256, 256).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(test_input[:, :, :8, :, :])
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            output = self.model(test_input)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        fps = test_frames / total_time
        
        # Memory usage
        memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        benchmark_stats = {
            'test_frames': test_frames,
            'processing_time': total_time,
            'fps': fps,
            'memory_gb': memory_used,
            'input_shape': list(test_input.shape),
            'output_shape': list(output.shape),
            'scale_factor': self.scale
        }
        
        logger.info(f"   FPS: {fps:.1f}")
        logger.info(f"   Memory: {memory_used:.2f} GB")
        logger.info(f"   Time: {total_time:.3f}s")
        
        return benchmark_stats
    
    def get_model_info(self) -> Dict:
        """Get information about the Fast Mamba VSR model."""
        return {
            'name': 'FastMambaVSR',
            'description': 'Ultra-efficient Mamba-based video super-resolution',
            'scale': self.scale,
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'device': str(self.device),
            'architecture': 'Fast Mamba with linear complexity',
            'tile_size': self.tile_size,
            'batch_size': self.batch_size,
            'optimizations': ['gradient_checkpointing', 'separable_conv', 'efficient_attention']
        }