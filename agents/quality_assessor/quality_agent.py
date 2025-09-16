"""
Quality Assessment Agent - Video Enhancement Pipeline

This agent is responsible for:
1. Computing quality metrics for enhanced videos
2. Validating results against quality standards
3. Making re-processing decisions
4. Providing quality reports and feedback
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
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models

from agentscope.agent import AgentBase
from agentscope.message import Msg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityAssessmentAgent(AgentBase):
    """Agent responsible for video quality assessment and validation"""
    
    def __init__(self, 
                 name: str = "QualityAssessor",
                 model_configs: Optional[Dict] = None,
                 quality_thresholds: Optional[Dict] = None,
                 **kwargs):
        super().__init__()
        
        # Set agent name and other attributes
        self.name = name
        
        self.model_configs = model_configs or {}
        self.quality_thresholds = quality_thresholds or self._get_default_thresholds()
        
        # Initialize models for perceptual metrics
        self.perceptual_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Quality categories
        self.quality_categories = {
            "excellent": {"min_psnr": 35.0, "min_ssim": 0.95, "max_lpips": 0.1},
            "good": {"min_psnr": 30.0, "min_ssim": 0.90, "max_lpips": 0.2},
            "acceptable": {"min_psnr": 25.0, "min_ssim": 0.80, "max_lpips": 0.4},
            "poor": {"min_psnr": 20.0, "min_ssim": 0.70, "max_lpips": 0.6},
        }
        
        # Processing statistics
        self.assessment_stats = {
            "total_assessments": 0,
            "quality_distribution": {cat: 0 for cat in self.quality_categories},
            "avg_processing_time": 0.0,
            "reprocessing_requests": 0
        }
        
        logger.info(f"Quality Assessment Agent '{name}' initialized")
    
    def _get_default_thresholds(self) -> Dict:
        """Get default quality thresholds"""
        return {
            "psnr": {"min": 25.0, "target": 30.0},
            "ssim": {"min": 0.80, "target": 0.90},
            "lpips": {"max": 0.4, "target": 0.2},
            "temporal_consistency": {"min": 0.85, "target": 0.95},
            "artifact_threshold": 0.3,
            "max_retry_attempts": 3,
            "quality_improvement_min": 0.05
        }
    
    def initialize_models(self):
        """Initialize perceptual quality assessment models"""
        try:
            # Initialize VGG model for perceptual loss
            self.perceptual_model = models.vgg19(pretrained=True).features[:36].to(self.device)
            self.perceptual_model.eval()
            
            # Normalize transform for VGG
            self.vgg_normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            
            logger.info("Perceptual quality assessment models initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def reply(self, x: Msg) -> Msg:
        """Main entry point for quality assessment tasks"""
        try:
            task_type = x.content.get("task", "assess_quality")
            
            if task_type == "assess_quality":
                return await self._assess_video_quality(x)
            elif task_type == "validate_enhancement":
                return await self._validate_enhancement(x)
            elif task_type == "compute_metrics":
                return await self._compute_quality_metrics(x)
            else:
                return Msg(
                    name=self.name,
                    content={
                        "status": "error",
                        "message": f"Unknown task type: {task_type}"
                    },
                    role="assistant"
                )
                
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return Msg(
                name=self.name,
                content={
                    "status": "error",
                    "message": str(e)
                },
                role="assistant"
            )
    
    async def _assess_video_quality(self, message: Msg) -> Msg:
        """Perform comprehensive video quality assessment"""
        start_time = time.time()
        
        content = message.content
        original_path = content.get("original_video")
        enhanced_path = content.get("enhanced_video")
        
        if not all([original_path, enhanced_path]):
            return Msg(
                name=self.name,
                content={
                    "status": "error",
                    "message": "Missing required paths: original_video, enhanced_video"
                },
                role="assistant"
            )
        
        try:
            # Initialize models if not already done
            if self.perceptual_model is None:
                self.initialize_models()
            
            # Load videos
            original_frames = self._load_video_frames(original_path)
            enhanced_frames = self._load_video_frames(enhanced_path)
            
            if len(original_frames) != len(enhanced_frames):
                logger.warning("Frame count mismatch between original and enhanced videos")
            
            # Compute comprehensive quality metrics
            metrics = await self._compute_comprehensive_metrics(
                original_frames, enhanced_frames
            )
            
            # Determine quality category
            quality_category = self._categorize_quality(metrics)
            
            # Make re-processing decision
            reprocess_needed = self._should_reprocess(metrics, quality_category)
            reprocess_recommendations = []
            
            if reprocess_needed:
                reprocess_recommendations = self._generate_reprocess_recommendations(
                    metrics, quality_category
                )
            
            # Update statistics
            self._update_stats(quality_category, time.time() - start_time, reprocess_needed)
            
            # Generate detailed report
            report = self._generate_quality_report(
                metrics, quality_category, reprocess_needed, reprocess_recommendations
            )
            
            return Msg(
                name=self.name,
                content={
                    "status": "success",
                    "quality_assessment": {
                        "metrics": metrics,
                        "quality_category": quality_category,
                        "reprocess_needed": reprocess_needed,
                        "reprocess_recommendations": reprocess_recommendations,
                        "processing_time": time.time() - start_time,
                        "detailed_report": report
                    }
                },
                role="assistant"
            )
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return Msg(
                name=self.name,
                content={
                    "status": "error",
                    "message": f"Quality assessment failed: {e}"
                },
                role="assistant"
            )
    
    def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Load video frames from file"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
        finally:
            cap.release()
        
        logger.info(f"Loaded {len(frames)} frames from {video_path}")
        return frames
    
    async def _compute_comprehensive_metrics(self, 
                                          original_frames: List[np.ndarray],
                                          enhanced_frames: List[np.ndarray]) -> Dict:
        """Compute comprehensive quality metrics"""
        metrics = {
            "objective": {},
            "perceptual": {},
            "temporal": {},
            "enhancement": {}
        }
        
        num_frames = min(len(original_frames), len(enhanced_frames))
        
        # Objective metrics (frame-wise)
        psnr_values = []
        ssim_values = []
        mse_values = []
        mae_values = []
        
        for i in range(num_frames):
            orig = original_frames[i].astype(np.float32)
            enh = enhanced_frames[i].astype(np.float32)
            
            # PSNR
            psnr_val = psnr(orig, enh, data_range=255.0)
            psnr_values.append(psnr_val)
            
            # SSIM (handle small images)
            try:
                # For small images, use smaller window size
                min_dim = min(orig.shape[0], orig.shape[1])
                if min_dim < 7:
                    win_size = 3  # Minimum window size
                elif min_dim < 11:
                    win_size = 7
                else:
                    win_size = None  # Use default
                
                ssim_val = ssim(orig, enh, multichannel=True, data_range=255.0, win_size=win_size, channel_axis=2)
                ssim_values.append(ssim_val)
            except Exception as e:
                logger.warning(f"SSIM computation failed: {e}, using fallback value")
                ssim_values.append(0.5)  # Fallback neutral value
            
            # MSE
            mse_val = np.mean((orig - enh) ** 2)
            mse_values.append(mse_val)
            
            # MAE
            mae_val = np.mean(np.abs(orig - enh))
            mae_values.append(mae_val)
        
        metrics["objective"] = {
            "psnr": {"mean": np.mean(psnr_values), "std": np.std(psnr_values)},
            "ssim": {"mean": np.mean(ssim_values), "std": np.std(ssim_values)},
            "mse": {"mean": np.mean(mse_values), "std": np.std(mse_values)},
            "mae": {"mean": np.mean(mae_values), "std": np.std(mae_values)}
        }
        
        # Perceptual metrics
        lpips_score = await self._compute_lpips(original_frames, enhanced_frames)
        perceptual_loss = await self._compute_perceptual_loss(original_frames, enhanced_frames)
        
        metrics["perceptual"] = {
            "lpips": lpips_score,
            "perceptual_loss": perceptual_loss
        }
        
        # Temporal metrics
        temporal_consistency = self._compute_temporal_consistency(enhanced_frames)
        flickering_score = self._compute_flickering_score(enhanced_frames)
        
        metrics["temporal"] = {
            "temporal_consistency": temporal_consistency,
            "flickering_score": flickering_score
        }
        
        # Enhancement effectiveness metrics
        detail_preservation = self._compute_detail_preservation(original_frames, enhanced_frames)
        artifact_score = self._compute_artifact_score(enhanced_frames)
        color_accuracy = self._compute_color_accuracy(original_frames, enhanced_frames)
        
        metrics["enhancement"] = {
            "detail_preservation": detail_preservation,
            "artifact_score": artifact_score,
            "color_accuracy": color_accuracy
        }
        
        return metrics
    
    async def _compute_lpips(self, original_frames: List[np.ndarray], 
                           enhanced_frames: List[np.ndarray]) -> float:
        """Compute LPIPS (Learned Perceptual Image Patch Similarity)"""
        # This is a simplified implementation - in practice, use the official LPIPS implementation
        try:
            # Sample a subset of frames for efficiency
            sample_indices = np.linspace(0, min(len(original_frames), len(enhanced_frames)) - 1, 
                                       min(10, len(original_frames)), dtype=int)
            
            lpips_scores = []
            
            for idx in sample_indices:
                orig = torch.from_numpy(original_frames[idx]).permute(2, 0, 1).float() / 255.0
                enh = torch.from_numpy(enhanced_frames[idx]).permute(2, 0, 1).float() / 255.0
                
                # Resize to 224x224 for VGG
                orig = F.interpolate(orig.unsqueeze(0), size=(224, 224), mode='bilinear').to(self.device)
                enh = F.interpolate(enh.unsqueeze(0), size=(224, 224), mode='bilinear').to(self.device)
                
                # Normalize
                orig = self.vgg_normalize(orig)
                enh = self.vgg_normalize(enh)
                
                # Get VGG features
                with torch.no_grad():
                    orig_features = self.perceptual_model(orig)
                    enh_features = self.perceptual_model(enh)
                    
                    # Compute L2 distance in feature space
                    lpips_score = F.mse_loss(orig_features, enh_features).cpu().item()
                    lpips_scores.append(lpips_score)
            
            return np.mean(lpips_scores)
            
        except Exception as e:
            logger.error(f"LPIPS computation failed: {e}")
            return 0.5  # Return neutral score on error
    
    async def _compute_lpips_variance(self, frames: List[np.ndarray]) -> float:
        """Compute the variance of LPIPS scores between consecutive frames."""
        if len(frames) < 2:
            return 0.0

        try:
            lpips_scores = []
            for i in range(len(frames) - 1):
                frame1 = torch.from_numpy(frames[i]).permute(2, 0, 1).float() / 255.0
                frame2 = torch.from_numpy(frames[i+1]).permute(2, 0, 1).float() / 255.0

                frame1 = F.interpolate(frame1.unsqueeze(0), size=(224, 224), mode='bilinear').to(self.device)
                frame2 = F.interpolate(frame2.unsqueeze(0), size=(224, 224), mode='bilinear').to(self.device)

                frame1 = self.vgg_normalize(frame1)
                frame2 = self.vgg_normalize(frame2)

                with torch.no_grad():
                    features1 = self.perceptual_model(frame1)
                    features2 = self.perceptual_model(frame2)
                    lpips_score = F.mse_loss(features1, features2).cpu().item()
                    lpips_scores.append(lpips_score)
            
            return np.var(lpips_scores)
        except Exception as e:
            logger.error(f"LPIPS variance computation failed: {e}")
            return 0.1  # Return neutral score on error
    
    async def _compute_perceptual_loss(self, original_frames: List[np.ndarray], 
                                     enhanced_frames: List[np.ndarray]) -> float:
        """Compute perceptual loss using VGG features"""
        try:
            # Sample frames for efficiency
            sample_indices = np.linspace(0, min(len(original_frames), len(enhanced_frames)) - 1, 
                                       min(5, len(original_frames)), dtype=int)
            
            losses = []
            
            for idx in sample_indices:
                orig = torch.from_numpy(original_frames[idx]).permute(2, 0, 1).float() / 255.0
                enh = torch.from_numpy(enhanced_frames[idx]).permute(2, 0, 1).float() / 255.0
                
                orig = F.interpolate(orig.unsqueeze(0), size=(224, 224), mode='bilinear').to(self.device)
                enh = F.interpolate(enh.unsqueeze(0), size=(224, 224), mode='bilinear').to(self.device)
                
                orig = self.vgg_normalize(orig)
                enh = self.vgg_normalize(enh)
                
                with torch.no_grad():
                    orig_features = self.perceptual_model(orig)
                    enh_features = self.perceptual_model(enh)
                    
                    loss = F.l1_loss(orig_features, enh_features).cpu().item()
                    losses.append(loss)
            
            return np.mean(losses)
            
        except Exception as e:
            logger.error(f"Perceptual loss computation failed: {e}")
            return 0.1  # Return neutral score on error
    
    def _compute_temporal_consistency(self, frames: List[np.ndarray]) -> float:
        """Compute temporal consistency across frames"""
        if len(frames) < 2:
            return 1.0
        
        consistency_scores = []
        
        for i in range(len(frames) - 1):
            frame1 = frames[i].astype(np.float32)
            frame2 = frames[i + 1].astype(np.float32)
            
            # Compute optical flow-based consistency
            # Simplified implementation - use Lucas-Kanade optical flow
            gray1 = cv2.cvtColor((frame1 / 255.0 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor((frame2 / 255.0 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Simple frame difference as consistency measure
            diff = np.mean(np.abs(gray1.astype(np.float32) - gray2.astype(np.float32)))
            consistency = 1.0 - min(diff / 255.0, 1.0)
            consistency_scores.append(consistency)
        
        return np.mean(consistency_scores)
    
    def _compute_flickering_score(self, frames: List[np.ndarray]) -> float:
        """Detect flickering artifacts in video"""
        if len(frames) < 3:
            return 0.0
        
        # Compute mean brightness variations
        brightness_values = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray)
            brightness_values.append(brightness)
        
        # Compute second derivative to detect flickering
        brightness_array = np.array(brightness_values)
        if len(brightness_array) >= 3:
            second_derivative = np.abs(np.diff(brightness_array, n=2))
            flickering_score = np.mean(second_derivative) / 255.0
        else:
            flickering_score = 0.0
        
        return min(flickering_score, 1.0)
    
    def _compute_detail_preservation(self, original_frames: List[np.ndarray], 
                                   enhanced_frames: List[np.ndarray]) -> float:
        """Measure how well details are preserved during enhancement"""
        detail_scores = []
        
        num_frames = min(len(original_frames), len(enhanced_frames))
        sample_indices = np.linspace(0, num_frames - 1, min(5, num_frames), dtype=int)
        
        for idx in sample_indices:
            orig = original_frames[idx].astype(np.float32)
            enh = enhanced_frames[idx].astype(np.float32)
            
            # Compute gradient magnitude to measure detail content
            orig_gray = cv2.cvtColor((orig / 255.0 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            enh_gray = cv2.cvtColor((enh / 255.0 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Sobel gradients
            orig_grad_x = cv2.Sobel(orig_gray, cv2.CV_64F, 1, 0, ksize=3)
            orig_grad_y = cv2.Sobel(orig_gray, cv2.CV_64F, 0, 1, ksize=3)
            orig_grad = np.sqrt(orig_grad_x**2 + orig_grad_y**2)
            
            enh_grad_x = cv2.Sobel(enh_gray, cv2.CV_64F, 1, 0, ksize=3)
            enh_grad_y = cv2.Sobel(enh_gray, cv2.CV_64F, 0, 1, ksize=3)
            enh_grad = np.sqrt(enh_grad_x**2 + enh_grad_y**2)
            
            # Compare gradient magnitudes
            try:
                data_range = orig_grad.max() - orig_grad.min()
                if data_range == 0:
                    grad_similarity = 1.0  # Perfect similarity when no gradients
                else:
                    # Handle small images
                    min_dim = min(orig_grad.shape[0], orig_grad.shape[1])
                    if min_dim < 7:
                        win_size = 3
                    elif min_dim < 11:
                        win_size = 7
                    else:
                        win_size = None
                    
                    grad_similarity = ssim(orig_grad, enh_grad, data_range=data_range, win_size=win_size)
                detail_scores.append(grad_similarity)
            except Exception as e:
                logger.warning(f"Detail preservation SSIM failed: {e}, using fallback")
                detail_scores.append(0.7)  # Neutral fallback
        
        return np.mean(detail_scores)
    
    def _compute_artifact_score(self, frames: List[np.ndarray]) -> float:
        """Detect common enhancement artifacts"""
        artifact_scores = []
        
        sample_indices = np.linspace(0, len(frames) - 1, min(3, len(frames)), dtype=int)
        
        for idx in sample_indices:
            frame = frames[idx].astype(np.float32) / 255.0
            
            # Detect blocking artifacts using DCT
            gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Simple blocking artifact detection
            h, w = gray.shape
            block_size = 8
            blocking_score = 0.0
            
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = gray[y:y+block_size, x:x+block_size]
                    
                    # Check for discontinuities at block boundaries
                    if x + block_size < w:
                        right_diff = np.mean(np.abs(block[:, -1].astype(np.float32) - 
                                                   gray[y:y+block_size, x+block_size].astype(np.float32)))
                        blocking_score += right_diff
                    
                    if y + block_size < h:
                        bottom_diff = np.mean(np.abs(block[-1, :].astype(np.float32) - 
                                                    gray[y+block_size, x:x+block_size].astype(np.float32)))
                        blocking_score += bottom_diff
            
            # Normalize blocking score
            num_blocks = ((h // block_size) * (w // block_size))
            if num_blocks > 0:
                blocking_score = blocking_score / (num_blocks * 2 * 255.0)
            
            artifact_scores.append(blocking_score)
        
        return np.mean(artifact_scores)
    
    def _compute_color_accuracy(self, original_frames: List[np.ndarray], 
                              enhanced_frames: List[np.ndarray]) -> float:
        """Measure color reproduction accuracy"""
        color_scores = []
        
        num_frames = min(len(original_frames), len(enhanced_frames))
        sample_indices = np.linspace(0, num_frames - 1, min(3, num_frames), dtype=int)
        
        for idx in sample_indices:
            orig = original_frames[idx].astype(np.float32) / 255.0
            enh = enhanced_frames[idx].astype(np.float32) / 255.0
            
            # Convert to LAB color space for perceptual color comparison
            orig_lab = cv2.cvtColor(orig, cv2.COLOR_RGB2LAB)
            enh_lab = cv2.cvtColor(enh, cv2.COLOR_RGB2LAB)
            
            # Compute color difference
            color_diff = np.mean(np.abs(orig_lab - enh_lab))
            color_similarity = 1.0 - min(color_diff / 100.0, 1.0)  # LAB values typically 0-100
            color_scores.append(color_similarity)
        
        return np.mean(color_scores)
    
    def _categorize_quality(self, metrics: Dict) -> str:
        """Categorize overall quality based on metrics"""
        psnr_mean = metrics["objective"]["psnr"]["mean"]
        ssim_mean = metrics["objective"]["ssim"]["mean"]
        lpips_score = metrics["perceptual"]["lpips"]
        
        for category, thresholds in self.quality_categories.items():
            if (psnr_mean >= thresholds["min_psnr"] and
                ssim_mean >= thresholds["min_ssim"] and
                lpips_score <= thresholds["max_lpips"]):
                return category
        
        return "failed"
    
    def _should_reprocess(self, metrics: Dict, quality_category: str) -> bool:
        """Determine if re-processing is needed"""
        # Check if quality meets minimum thresholds
        if quality_category in ["poor", "failed"]:
            return True
        
        # Check specific metric thresholds
        psnr_mean = metrics["objective"]["psnr"]["mean"]
        ssim_mean = metrics["objective"]["ssim"]["mean"]
        temporal_consistency = metrics["temporal"]["temporal_consistency"]
        artifact_score = metrics["enhancement"]["artifact_score"]
        
        if (psnr_mean < self.quality_thresholds["psnr"]["min"] or
            ssim_mean < self.quality_thresholds["ssim"]["min"] or
            temporal_consistency < self.quality_thresholds["temporal_consistency"]["min"] or
            artifact_score > self.quality_thresholds["artifact_threshold"]):
            return True
        
        return False
    
    def _generate_reprocess_recommendations(self, metrics: Dict, quality_category: str) -> List[str]:
        """Generate specific recommendations for re-processing"""
        recommendations = []
        
        psnr_mean = metrics["objective"]["psnr"]["mean"]
        ssim_mean = metrics["objective"]["ssim"]["mean"]
        lpips_score = metrics["perceptual"]["lpips"]
        temporal_consistency = metrics["temporal"]["temporal_consistency"]
        flickering_score = metrics["temporal"].get("flickering_score", 0.0)
        artifact_score = metrics["enhancement"]["artifact_score"]
        
        if psnr_mean < self.quality_thresholds["psnr"]["min"]:
            recommendations.append("Increase enhancement strength or use more conservative approach")
        
        if ssim_mean < self.quality_thresholds["ssim"]["min"]:
            recommendations.append("Focus on preserving structural information")
        
        if lpips_score > self.quality_thresholds["lpips"]["max"]:
            recommendations.append("Reduce perceptual distortions, consider different enhancement model")
        
        if temporal_consistency < self.quality_thresholds["temporal_consistency"]["min"]:
            recommendations.append("Improve temporal consistency, consider frame-to-frame constraints")
        
        if flickering_score > 0.1:
            recommendations.append("Apply temporal smoothing to reduce flickering")
        
        if artifact_score > self.quality_thresholds["artifact_threshold"]:
            recommendations.append("Reduce enhancement parameters to minimize artifacts")
        
        if not recommendations:
            recommendations.append("Try alternative enhancement strategy or model")
        
        return recommendations
    
    def _generate_quality_report(self, metrics: Dict, quality_category: str, 
                               reprocess_needed: bool, recommendations: List[str]) -> Dict:
        """Generate comprehensive quality assessment report"""
        return {
            "summary": {
                "quality_category": quality_category,
                "reprocess_needed": reprocess_needed,
                "overall_score": self._compute_overall_score(metrics),
                "primary_concerns": self._identify_primary_concerns(metrics)
            },
            "detailed_metrics": metrics,
            "recommendations": recommendations,
            "threshold_analysis": self._analyze_thresholds(metrics),
            "improvement_suggestions": self._suggest_improvements(metrics, quality_category)
        }
    
    def _compute_overall_score(self, metrics: Dict) -> float:
        """Compute weighted overall quality score"""
        weights = {
            "psnr": 0.25,
            "ssim": 0.25,
            "lpips": 0.20,
            "temporal_consistency": 0.15,
            "detail_preservation": 0.10,
            "artifact_score": 0.05
        }
        
        # Normalize metrics to 0-1 scale
        normalized_scores = {
            "psnr": min(metrics["objective"]["psnr"]["mean"] / 40.0, 1.0),
            "ssim": metrics["objective"]["ssim"]["mean"],
            "lpips": 1.0 - min(metrics["perceptual"]["lpips"], 1.0),
            "temporal_consistency": metrics["temporal"]["temporal_consistency"],
            "detail_preservation": metrics["enhancement"]["detail_preservation"],
            "artifact_score": 1.0 - min(metrics["enhancement"]["artifact_score"], 1.0)
        }
        
        overall_score = sum(weights[key] * normalized_scores[key] for key in weights)
        return overall_score
    
    def _identify_primary_concerns(self, metrics: Dict) -> List[str]:
        """Identify primary quality concerns"""
        concerns = []
        
        if metrics["objective"]["psnr"]["mean"] < 25.0:
            concerns.append("Low PSNR - significant pixel-level differences")
        
        if metrics["objective"]["ssim"]["mean"] < 0.80:
            concerns.append("Low SSIM - structural distortions present")
        
        if metrics["perceptual"]["lpips"] > 0.4:
            concerns.append("High perceptual distance - noticeable quality degradation")
        
        if metrics["temporal"]["temporal_consistency"] < 0.85:
            concerns.append("Poor temporal consistency - frame-to-frame variations")
        
        flickering_score = metrics["temporal"].get("flickering_score", 0.0)
        if flickering_score > 0.1:
            concerns.append("Flickering detected - temporal artifacts present")
        
        if metrics["enhancement"]["artifact_score"] > 0.3:
            concerns.append("Enhancement artifacts detected")
        
        return concerns
    
    def _analyze_thresholds(self, metrics: Dict) -> Dict:
        """Analyze metrics against thresholds"""
        analysis = {}
        
        psnr_mean = metrics["objective"]["psnr"]["mean"]
        analysis["psnr"] = {
            "value": psnr_mean,
            "threshold_min": self.quality_thresholds["psnr"]["min"],
            "threshold_target": self.quality_thresholds["psnr"]["target"],
            "meets_min": psnr_mean >= self.quality_thresholds["psnr"]["min"],
            "meets_target": psnr_mean >= self.quality_thresholds["psnr"]["target"]
        }
        
        ssim_mean = metrics["objective"]["ssim"]["mean"]
        analysis["ssim"] = {
            "value": ssim_mean,
            "threshold_min": self.quality_thresholds["ssim"]["min"],
            "threshold_target": self.quality_thresholds["ssim"]["target"],
            "meets_min": ssim_mean >= self.quality_thresholds["ssim"]["min"],
            "meets_target": ssim_mean >= self.quality_thresholds["ssim"]["target"]
        }
        
        return analysis
    
    def _suggest_improvements(self, metrics: Dict, quality_category: str) -> List[str]:
        """Suggest specific improvements based on assessment"""
        suggestions = []
        
        if quality_category == "failed":
            suggestions.append("Consider using a different enhancement model entirely")
            suggestions.append("Apply preprocessing to improve input quality")
            suggestions.append("Use more conservative enhancement parameters")
        
        elif quality_category == "poor":
            suggestions.append("Increase model capacity or use higher quality settings")
            suggestions.append("Apply post-processing filters to reduce artifacts")
        
        elif quality_category == "acceptable":
            suggestions.append("Fine-tune parameters for better quality-speed trade-off")
            suggestions.append("Consider multi-pass enhancement for critical content")
        
        else:
            suggestions.append("Current quality is good, consider optimizing for speed")
        
        return suggestions
    
    def _update_stats(self, quality_category: str, processing_time: float, reprocess_needed: bool):
        """Update processing statistics"""
        self.assessment_stats["total_assessments"] += 1
        self.assessment_stats["quality_distribution"][quality_category] += 1
        
        # Update average processing time
        n = self.assessment_stats["total_assessments"]
        old_avg = self.assessment_stats["avg_processing_time"]
        self.assessment_stats["avg_processing_time"] = (old_avg * (n - 1) + processing_time) / n
        
        if reprocess_needed:
            self.assessment_stats["reprocessing_requests"] += 1
    
    async def _validate_enhancement(self, message: Msg) -> Msg:
        """Validate enhancement results against specific criteria"""
        # Implementation for validation tasks
        pass
    
    async def _compute_quality_metrics(self, message: Msg) -> Msg:
        """Compute specific quality metrics"""
        # Implementation for metric computation tasks
        pass
    
    def get_statistics(self) -> Dict:
        """Get current processing statistics"""
        return self.assessment_stats.copy()
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.assessment_stats = {
            "total_assessments": 0,
            "quality_distribution": {cat: 0 for cat in self.quality_categories},
            "avg_processing_time": 0.0,
            "reprocessing_requests": 0
        }