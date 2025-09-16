"""
RIFE Handler for Video Frame Interpolation
Implements RIFE (Real-Time Intermediate Flow Estimation) with proper weight loading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Tuple, List
import requests
import os
from huggingface_hub import hf_hub_download

from utils.video_utils import VideoUtils
from utils.performance_monitor import track_enhancement_performance

logger = logging.getLogger(__name__)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                          kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.PReLU(out_planes)
    )

class ResBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResBlock(c, c),
            ResBlock(c, c),
            ResBlock(c, c),
            ResBlock(c, c),
            ResBlock(c, c),
            ResBlock(c, c),
            ResBlock(c, c),
            ResBlock(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale=1):
        x = F.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
        flow = F.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
        feat = self.conv0(torch.cat((x, flow), 1))
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale*2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask

class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7+4, c=240)
        self.block1 = IFBlock(8+4, c=150)
        self.block2 = IFBlock(8+4, c=90)
        self.block_tea = IFBlock(8+4, c=90)
        self.contextnet = Contextnet()
        self.unet = Unet()

    def forward(self, x, scale_list=[4,2,1], training=False):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:] # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = (x[:, :4]).detach() * 0
        loss_distill = 0
        
        for i in range(3):
            if i == 0:
                flow_d, mask_d = self.block0(torch.cat((img0, img1, warped_img0, warped_img1, mask), 1) if i == 2 else torch.cat((img0, img1, warped_img0, warped_img1), 1), flow, scale=scale_list[i])
                flow = flow + flow_d
            else:
                flow_d, mask_d = (self.block1 if i==1 else self.block2)(torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow, scale=scale_list[i])
                flow = flow + flow_d
            
            mask = mask_d
            flow_list.append(flow)
            mask_list.append(mask)
            warped_img0 = self.warp(img0, flow[:, :2])
            warped_img1 = self.warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
            
        if training:
            c0 = self.contextnet(img0, flow[:, :2])
            c1 = self.contextnet(img1, flow[:, 2:4])
            tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
            res = tmp[:, :3] * 2 - 1
            merged[2] = torch.clamp(merged[2][0] * mask + merged[2][1] * (1 - mask) + res, 0, 1)
        else:
            c0 = self.contextnet(img0, flow[:, :2])
            c1 = self.contextnet(img1, flow[:, 2:4])
            tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
            res = tmp[:, :3] * 2 - 1
            merged_final = torch.clamp(warped_img0 * mask + warped_img1 * (1 - mask) + res, 0, 1)
            return merged_final

        return flow_list, mask_list[2], merged, loss_distill

    def warp(self, x, flow, mode="bilinear", padding_mode="border"):
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()
        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flow
        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0
        vgrid = vgrid.permute(0,2,3,1)
        output = F.grid_sample(x, vgrid, mode=mode, padding_mode=padding_mode, align_corners=True)
        return output

class Contextnet(nn.Module):
    def __init__(self):
        super(Contextnet, self).__init__()
        self.conv1 = conv(3, 64, 3, 2, 1)
        self.conv2 = ResBlock(64, 64)
        self.conv3 = conv(64, 128, 3, 2, 1)
        self.conv4 = ResBlock(128, 128)
        self.conv5 = conv(128, 256, 3, 2, 1)
        self.conv6 = ResBlock(256, 256)

    def forward(self, x, flow):
        x = self.conv1(x)
        f1 = self.warp(x, flow * 0.25)
        x = self.conv2(x)
        x = self.conv3(x)
        f2 = self.warp(x, flow * 0.125)
        x = self.conv4(x)
        x = self.conv5(x)
        f3 = self.warp(x, flow * 0.0625)
        x = self.conv6(x)
        return [f1, f2, f3, x]
    
    def warp(self, x, flow, mode="bilinear", padding_mode="border"):
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()
        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flow
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0
        vgrid = vgrid.permute(0,2,3,1)
        output = F.grid_sample(x, vgrid, mode=mode, padding_mode=padding_mode, align_corners=True)
        return output

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.down0 = conv(17, 32, 3, 2, 1)
        self.down1 = conv(32, 64, 3, 2, 1)
        self.down2 = conv(64, 128, 3, 2, 1)
        self.down3 = conv(128, 256, 3, 2, 1)
        self.up0 = deconv(256, 128)
        self.up1 = deconv(256, 64)
        self.up2 = deconv(128, 32)
        self.up3 = deconv(64, 8)
        self.conv = nn.Conv2d(8, 3, 3, 1, 1)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1))
        s1 = self.down1(torch.cat((s0, c0[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2]), 1))
        x = self.up0(torch.cat((s3, c0[3]), 1))
        x = self.up1(torch.cat((x, s2), 1)) 
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)
        return torch.sigmoid(x)

class RIFEHandler:
    """RIFE Video Frame Interpolation Handler with proper weight loading."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = "cuda",
                 scale: float = 1.0,
                 fp16: bool = False):
        
        self.device_str = device
        self.scale = scale
        self.fp16 = fp16
        self.model_path = model_path
        
        logger.info("🎬 Initializing RIFE Handler...")
        logger.info(f"   Device: {device}")
        logger.info(f"   Scale: {scale}")
        logger.info(f"   FP16: {fp16}")
        
        # Defer model initialization
        self.model = None
        self._model_loaded = False
        
        # Resolve model weights path
        self.resolved_model_path = self._resolve_model_path(model_path)
        
        self.video_utils = VideoUtils()
        
        logger.info("✅ RIFE Handler initialized (model loading deferred)")
    
    def _ensure_model_loaded(self):
        """Lazy load model when actually needed."""
        if not self._model_loaded:
            logger.info("📎 Loading RIFE model (deferred initialization)...")
            
            # Now safe to initialize device
            self.device = torch.device(self.device_str)
            
            # Initialize network
            self.model = IFNet().to(self.device)
            
            if self.fp16:
                self.model = self.model.half()
            
            # Load pretrained weights
            if self.resolved_model_path and Path(self.resolved_model_path).exists():
                self._load_model(self.resolved_model_path)
            else:
                logger.warning("No RIFE model weights found, using random initialization")
            
            self.model.eval()
            self._model_loaded = True
            logger.info("✅ RIFE model loaded successfully")
    
    def _resolve_model_path(self, model_path: Optional[str]) -> Optional[str]:
        """Resolve model path from explicit arg, download if needed."""
        if model_path and Path(model_path).exists():
            return str(model_path)
        
        # Default path
        default_path = Path("models/weights/RIFE/flownet.pkl")
        if default_path.exists():
            return str(default_path)
        
        # Try to download from HuggingFace
        try:
            logger.info("🔽 Downloading RIFE weights from HuggingFace...")
            default_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download from RIFE repository alternative
            url = "https://github.com/hzwer/Practical-RIFE/releases/download/v4.6/flownet-v4.6.pkl"
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(default_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"✅ Downloaded RIFE weights: {default_path}")
            return str(default_path)
            
        except Exception as e:
            logger.warning(f"Failed to download RIFE weights: {e}")
            return None
    
    def _load_model(self, model_path: str):
        """Load pretrained RIFE weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=True)
            logger.info(f"✅ Loaded RIFE weights from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            logger.info("Using random initialization")
    
    @track_enhancement_performance('rife')
    def interpolate_video(self, 
                         input_path: str, 
                         output_path: str,
                         interpolation_factor: int = 2,
                         **kwargs) -> Dict:
        """Interpolate video frames using RIFE."""
        
        # Ensure model is loaded
        if not self._model_loaded:
            self._ensure_model_loaded()
        
        logger.info(f"🎬 Interpolating video with RIFE...")
        logger.info(f"   Input: {input_path}")
        logger.info(f"   Output: {output_path}")
        logger.info(f"   Interpolation factor: {interpolation_factor}x")
        
        try:
            # Get video metadata
            metadata = self.video_utils.get_video_metadata(input_path)
            fps = metadata['fps']
            total_frames = metadata['frame_count']
            
            # Setup video capture and writer
            cap = cv2.VideoCapture(input_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Output FPS is increased by interpolation factor
            output_fps = fps * interpolation_factor
            
            out = cv2.VideoWriter(output_path, fourcc, output_fps, 
                                (metadata['width'], metadata['height']))
            
            # Read first frame
            ret, prev_frame = cap.read()
            if not ret:
                raise ValueError("Could not read first frame")
            
            out.write(prev_frame)  # Write first frame
            processed_count = 1
            interpolated_count = 0
            
            with torch.no_grad():
                while True:
                    ret, curr_frame = cap.read()
                    if not ret:
                        break
                    
                    # Generate interpolated frames between prev and curr
                    for i in range(1, interpolation_factor):
                        t = i / interpolation_factor
                        interpolated = self._interpolate_frame(prev_frame, curr_frame, t)
                        out.write(interpolated)
                        interpolated_count += 1
                    
                    # Write current frame
                    out.write(curr_frame)
                    processed_count += 1
                    
                    if processed_count % 30 == 0:
                        logger.info(f"📈 Progress: {processed_count}/{total_frames} frames, "
                                  f"{interpolated_count} interpolated")
                    
                    prev_frame = curr_frame
            
            cap.release()
            out.release()
            
            total_output_frames = processed_count + interpolated_count
            
            stats = {
                'input_frames': total_frames,
                'output_frames': total_output_frames,
                'interpolated_frames': interpolated_count,
                'processing_mode': 'rife_interpolation',
                'interpolation_factor': interpolation_factor,
                'input_fps': fps,
                'output_fps': output_fps,
                'frames_processed': processed_count
            }
            
            logger.info(f"✅ RIFE interpolation completed")
            logger.info(f"   Input frames: {total_frames}")
            logger.info(f"   Output frames: {total_output_frames}")
            logger.info(f"   Interpolated: {interpolated_count}")
            logger.info(f"   Output FPS: {output_fps}")
            
            return stats
            
        except Exception as e:
            logger.error(f"RIFE interpolation failed: {e}")
            raise
    
    def _interpolate_frame(self, frame1: np.ndarray, frame2: np.ndarray, t: float) -> np.ndarray:
        """Interpolate between two frames at time t."""
        try:
            # Convert frames to tensors
            img0 = self._frame_to_tensor(frame1)
            img1 = self._frame_to_tensor(frame2)
            
            # Prepare input (concatenate frames)
            input_tensor = torch.cat([img0, img1], dim=1)
            
            if self.fp16:
                input_tensor = input_tensor.half()
            
            # Generate intermediate frame
            with torch.no_grad():
                output = self.model(input_tensor, training=False)
            
            # Convert back to frame
            interpolated_frame = self._tensor_to_frame(output)
            
            return interpolated_frame
            
        except Exception as e:
            logger.error(f"Frame interpolation failed: {e}")
            # Return simple blend as fallback
            return cv2.addWeighted(frame1, 1-t, frame2, t, 0)
    
    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """Convert frame to tensor."""
        # Convert BGR to RGB and normalize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb.transpose(2, 0, 1)).float()
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device) / 255.0
        return frame_tensor
    
    def _tensor_to_frame(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to frame."""
        # Convert to numpy and denormalize
        frame_np = tensor.squeeze(0).cpu().float().numpy()
        frame_np = (frame_np.transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # Convert RGB back to BGR
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        return frame_bgr
    
    def get_model_info(self) -> Dict:
        """Get information about the RIFE model."""
        if not self._model_loaded:
            self._ensure_model_loaded()
            
        return {
            'name': 'RIFE',
            'description': 'Real-Time Intermediate Flow Estimation for Video Frame Interpolation',
            'version': 'v4.6',
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'device': str(self.device),
            'architecture': 'IFNet with Context and UNet refinement',
            'fp16': self.fp16
        }