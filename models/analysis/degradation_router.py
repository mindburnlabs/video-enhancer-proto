"""
Intelligent Degradation Router for Topaz Video AI 7 Killer Pipeline

This module analyzes video content and routes it to appropriate expert pipelines
based on detected degradations like compression artifacts, motion blur, low light,
noise levels, and face prominence.
"""

import torch
import cv2
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DegradationRouter:
    """Routes video content to appropriate expert pipelines based on intelligent analysis"""
    
    def __init__(self, device="cuda"):
        self.device = device
        logger.info("🔍 Initializing Degradation Router for intelligent content analysis")
        
        # Initialize face detection cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Analysis thresholds (tuned for optimal routing)
        self.thresholds = {
            'compression_artifacts': 0.6,
            'motion_blur': 0.5,
            'low_light': 0.6,
            'noise': 0.4,
            'face_prominence': 0.03,
            'scene_complexity': 0.7
        }
        
        # Available SOTA models (checked during initialization)
        self.available_models = self._check_available_models()
        logger.info(f"Available SOTA models: {list(self.available_models.keys())}")
        
        logger.info("✅ Degradation Router initialized with optimal thresholds")
        
    def _check_available_models(self) -> Dict[str, bool]:
        """Check which SOTA model handlers are available."""
        available = {}
        
        # Check SOTA model handlers
        try:
            from models.enhancement.vsr.vsrm_handler import VSRMHandler
            available['vsrm'] = True
        except ImportError:
            available['vsrm'] = False
            
        try:
            from models.enhancement.zeroshot.seedvr2_handler import SeedVR2Handler
            available['seedvr2'] = True
        except ImportError:
            available['seedvr2'] = False
            
        try:
            from models.enhancement.zeroshot.ditvr_handler import DiTVRHandler
            available['ditvr'] = True
        except ImportError:
            available['ditvr'] = False
            
        try:
            from models.enhancement.vsr.fast_mamba_vsr_handler import FastMambaVSRHandler
            available['fast_mamba_vsr'] = True
        except ImportError:
            available['fast_mamba_vsr'] = False
            
        # Check interpolation
        try:
            from models.interpolation.enhanced_rife_handler import EnhancedRIFEHandler
            available['rife'] = True
        except ImportError:
            available['rife'] = False
            
        return available
        
    def analyze_and_route(self, video_path: str) -> Dict:
        """
        Analyze video degradations and create expert routing plan
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Dict containing degradation analysis, content analysis, and expert routing plan
        """
        logger.info(f"🎬 Analyzing video for expert routing: {Path(video_path).name}")
        
        # Sample frames for analysis
        frames = self._sample_frames(video_path, num_samples=12)
        
        if not frames:
            logger.error("❌ Failed to sample frames from video")
            return self._get_fallback_routing()
        
        # Detect degradation types
        logger.info("🔍 Detecting video degradations...")
        degradations = self._detect_degradations(frames)
        
        # Analyze content characteristics
        logger.info("📊 Analyzing content characteristics...")
        content_analysis = self._analyze_content(frames)
        
        # Create routing plan
        logger.info("🗺️ Creating expert routing plan...")
        routing_plan = self._create_routing_plan(degradations, content_analysis)
        
        # Determine optimal processing order
        processing_order = self._determine_processing_order(routing_plan)
        
        result = {
            'degradations': degradations,
            'content_analysis': content_analysis,
            'expert_routing': routing_plan,
            'processing_order': processing_order,
            'confidence_score': self._calculate_routing_confidence(degradations, content_analysis)
        }
        
        self._log_routing_decision(result)
        
        return result
    
    def _detect_degradations(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Detect specific degradation types in video frames"""
        degradations = {
            'compression_artifacts': 0.0,
            'motion_blur': 0.0,
            'low_light': 0.0,
            'noise': 0.0,
            'temporal_inconsistency': 0.0
        }
        
        prev_frame = None
        
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # 1. Compression artifacts detection (DCT analysis)
            try:
                # Resize for DCT analysis (must be multiple of 8)
                h, w = gray.shape
                h_crop = (h // 8) * 8
                w_crop = (w // 8) * 8
                gray_cropped = gray[:h_crop, :w_crop]
                
                # Convert to float32 and normalize
                gray_float = gray_cropped.astype(np.float32) / 255.0
                
                # Apply DCT to 8x8 blocks and analyze frequency distribution
                compression_score = self._analyze_compression_artifacts(gray_float)
                degradations['compression_artifacts'] = max(
                    degradations['compression_artifacts'], compression_score
                )
            except Exception as e:
                logger.warning(f"Compression analysis failed: {e}")
            
            # 2. Motion blur detection (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            motion_blur_score = max(0.0, 1.0 - min(laplacian_var / 1000.0, 1.0))
            degradations['motion_blur'] = max(degradations['motion_blur'], motion_blur_score)
            
            # 3. Low light detection (brightness + histogram analysis)
            brightness = np.mean(gray)
            low_light_score = max(0.0, (120 - brightness) / 120.0)
            
            # Enhance with histogram analysis
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            dark_pixels_ratio = np.sum(hist[:64]) / np.sum(hist)  # Pixels in lower 25%
            hist_low_light = min(1.0, dark_pixels_ratio)
            
            combined_low_light = (low_light_score + hist_low_light) / 2
            degradations['low_light'] = max(degradations['low_light'], combined_low_light)
            
            # 4. Noise level estimation
            noise_estimate = np.std(gray.astype(float) - cv2.GaussianBlur(gray, (5, 5), 0).astype(float))
            noise_score = min(1.0, noise_estimate / 25.0)
            degradations['noise'] = max(degradations['noise'], noise_score)
            
            # 5. Temporal inconsistency (between frames)
            if prev_frame is not None:
                frame_diff = cv2.absdiff(gray, prev_frame)
                temporal_inconsistency = np.mean(frame_diff) / 255.0
                degradations['temporal_inconsistency'] = max(
                    degradations['temporal_inconsistency'], temporal_inconsistency
                )
            
            prev_frame = gray
            
            if i % 3 == 0:
                logger.debug(f"Analyzed frame {i+1}/{len(frames)}")
        
        return degradations
    
    def _analyze_compression_artifacts(self, gray_float: np.ndarray) -> float:
        """Analyze compression artifacts using frequency domain analysis"""
        h, w = gray_float.shape
        compression_scores = []
        
        # Analyze 8x8 blocks (typical DCT block size)
        for y in range(0, h-8, 8):
            for x in range(0, w-8, 8):
                block = gray_float[y:y+8, x:x+8]
                
                # Apply DCT
                dct_block = cv2.dct(block)
                
                # Analyze frequency distribution
                high_freq = np.sum(np.abs(dct_block[4:, 4:]))  # High frequency components
                total_energy = np.sum(np.abs(dct_block))
                
                if total_energy > 0:
                    high_freq_ratio = high_freq / total_energy
                    # Low high-frequency content suggests compression artifacts
                    compression_score = 1.0 - min(high_freq_ratio * 4, 1.0)  # Scale and clamp
                    compression_scores.append(compression_score)
        
        return np.mean(compression_scores) if compression_scores else 0.0
    
    def _analyze_content(self, frames: List[np.ndarray]) -> Dict[str, any]:
        """Analyze content characteristics for specialized processing"""
        content_features = {
            'has_faces': False,
            'face_prominence': 0.0,
            'motion_complexity': 0.0,
            'scene_changes': 0,
            'dominant_colors': [],
            'sharpness_variation': 0.0
        }
        
        face_scores = []
        sharpness_scores = []
        prev_hist = None
        scene_changes = 0
        
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Face detection and prominence analysis
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(faces) > 0:
                content_features['has_faces'] = True
                # Calculate face prominence (total face area / frame area)
                total_face_area = sum(w * h for x, y, w, h in faces)
                frame_area = frame.shape[0] * frame.shape[1]
                face_prominence = total_face_area / frame_area
                face_scores.append(face_prominence)
            
            # Sharpness analysis (for motion complexity estimation)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_scores.append(sharpness)
            
            # Scene change detection (histogram comparison)
            if prev_hist is not None:
                current_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                correlation = cv2.compareHist(prev_hist, current_hist, cv2.HISTCMP_CORREL)
                if correlation < 0.7:  # Scene change threshold
                    scene_changes += 1
            
            prev_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Aggregate content features
        if face_scores:
            content_features['face_prominence'] = np.mean(face_scores)
        
        content_features['scene_changes'] = scene_changes
        content_features['sharpness_variation'] = np.std(sharpness_scores) if sharpness_scores else 0.0
        
        # Motion complexity based on sharpness variation and scene changes
        motion_complexity = min(1.0, (content_features['sharpness_variation'] / 1000 + scene_changes / len(frames)))
        content_features['motion_complexity'] = motion_complexity
        
        return content_features
    
    def _create_routing_plan(self, degradations: Dict, content: Dict, 
                           latency_class: str = 'standard', 
                           allow_diffusion: bool = True,
                           allow_zero_shot: bool = True) -> Dict:
        """Create SOTA routing plan based on analysis and constraints"""
        
        # Calculate degradation scores
        unknown_deg_score = self._calculate_unknown_degradation_score(degradations)
        blur_score = degradations['motion_blur']
        compression_score = degradations['compression_artifacts']
        noise_score = degradations['noise']
        
        # Select primary model based on SOTA strategy
        primary_model = self._select_sota_model(
            unknown_deg_score, blur_score, compression_score, 
            content['motion_complexity'], latency_class, 
            allow_diffusion, allow_zero_shot
        )
        
        routing_plan = {
            # SOTA Model Selection
            'primary_model': primary_model,
            'model_config': self._get_model_config(primary_model, latency_class),
            
            # Preprocessing experts (optional) - deblur disabled until implemented
            'use_deblur_expert': False,  # Disabled: BSSTNet/VD-Diff not implemented
            'use_compression_expert': compression_score > self.thresholds['compression_artifacts'],
            'use_denoising': noise_score > self.thresholds['noise'],
            'use_low_light_expert': degradations['low_light'] > self.thresholds['low_light'],
            
            # Fallback models (using available SOTA models)
            'fallback_model': 'vsrm' if primary_model != 'vsrm' else 'fast_mamba_vsr',
            
            # Post-processing
            'use_face_expert': content['has_faces'] and content['face_prominence'] > self.thresholds['face_prominence'],
            'use_temporal_consistency': degradations['temporal_inconsistency'] > 0.3,
            
            # Interpolation (always last)
            'use_hfr_interpolation': True,
            
            # Strategy metadata
            'latency_class': latency_class,
            'allow_diffusion': allow_diffusion,
            'allow_zero_shot': allow_zero_shot,
            'unknown_deg_score': unknown_deg_score
        }
        
        return routing_plan
    
    def _calculate_unknown_degradation_score(self, degradations: Dict) -> float:
        """Calculate score for unknown/mixed degradations requiring zero-shot handling."""
        known_degradations = ['compression_artifacts', 'motion_blur', 'noise', 'low_light']
        known_scores = [degradations.get(deg, 0) for deg in known_degradations]
        
        # High unknown score if:
        # 1. Multiple degradations are present simultaneously
        # 2. Degradation levels are ambiguous (around thresholds)
        # 3. Temporal inconsistency is high (suggests complex artifacts)
        
        multi_deg_penalty = sum(1 for score in known_scores if score > 0.3) / len(known_scores)
        ambiguity_score = sum(1 for score in known_scores if 0.4 < score < 0.7) / len(known_scores)
        temporal_penalty = min(1.0, degradations.get('temporal_inconsistency', 0) / 0.5)
        
        unknown_score = (multi_deg_penalty + ambiguity_score + temporal_penalty) / 3
        return min(1.0, unknown_score)
    
    def _select_sota_model(self, unknown_deg_score: float, blur_score: float, 
                          compression_score: float, motion_complexity: float,
                          latency_class: str, allow_diffusion: bool, 
                          allow_zero_shot: bool) -> str:
        """Select optimal SOTA model based on analysis, constraints, and availability."""
        
        # Strict latency: prefer fast_mamba_vsr if available
        if latency_class == 'strict':
            if self.available_models.get('fast_mamba_vsr', False):
                return 'fast_mamba_vsr'
            # Fallback to next fastest available
            return self._get_fallback_model(['fast_mamba_vsr'])
        
        # Unknown degradations with zero-shot capability
        if unknown_deg_score > 0.6 and allow_zero_shot:
            if self.available_models.get('ditvr', False):
                return 'ditvr'
            # Fallback to other models
            return self._get_fallback_model(['ditvr'])
        
        # High quality restoration needs with diffusion capability
        if (compression_score > 0.7 or blur_score > 0.6) and allow_diffusion:
            if self.available_models.get('seedvr2', False):
                return 'seedvr2'
            # Fallback to quality-focused alternatives
            return self._get_fallback_model(['seedvr2'])
        
        # High motion complexity needs advanced temporal processing
        if motion_complexity > 0.7:
            if self.available_models.get('vsrm', False):
                return 'vsrm'
            # Fallback to other temporal models
            return self._get_fallback_model(['vsrm'])
        
        # Flexible latency: prefer quality
        if latency_class == 'flexible':
            if allow_diffusion and self.available_models.get('seedvr2', False):
                return 'seedvr2'
            elif allow_zero_shot and self.available_models.get('ditvr', False):
                return 'ditvr' 
            elif self.available_models.get('vsrm', False):
                return 'vsrm'
            return self._get_fallback_model([])
        
        # Standard latency: balanced approach
        if unknown_deg_score > 0.4 and allow_zero_shot and self.available_models.get('ditvr', False):
            return 'ditvr'
        elif (compression_score > 0.5 or blur_score > 0.5) and allow_diffusion and self.available_models.get('seedvr2', False):
            return 'seedvr2'
        elif self.available_models.get('vsrm', False):
            return 'vsrm'  # Default SOTA choice
        else:
            return self._get_fallback_model([])  # Any available model
    
    def _get_fallback_model(self, excluded: List[str]) -> str:
        """Get best available fallback model, excluding specified models."""
        # Priority order for fallbacks (best to worst)
        fallback_priority = ['vsrm', 'ditvr', 'seedvr2', 'fast_mamba_vsr']
        
        for model in fallback_priority:
            if model not in excluded and self.available_models.get(model, False):
                logger.warning(f"Using fallback model: {model}")
                return model
        
        # Ultimate fallback: any available model
        for model, available in self.available_models.items():
            if available and model not in excluded:
                logger.warning(f"Using ultimate fallback model: {model}")
                return model
        
        # Critical error: no models available
        logger.error("No SOTA models available! Check model installation.")
        return 'vsrm'  # Return default, will fail gracefully later
    
    def _get_model_config(self, model_name: str, latency_class: str) -> Dict:
        """Get model-specific configuration based on latency class."""
        configs = {
            'vsrm': {
                'strict': {'window': 5, 'stride': 3, 'fp16': True, 'tile_size': 256},
                'standard': {'window': 7, 'stride': 4, 'fp16': True, 'tile_size': 512},
                'flexible': {'window': 9, 'stride': 5, 'fp16': False, 'tile_size': 768}
            },
            'seedvr2': {
                'strict': {'quality_threshold': 0.4, 'fp16': True, 'tile_size': 256},
                'standard': {'quality_threshold': 0.5, 'fp16': True, 'tile_size': 512}, 
                'flexible': {'quality_threshold': 0.6, 'fp16': False, 'tile_size': 768}
            },
            'ditvr': {
                'strict': {'auto_adapt': False, 'fp16': True, 'tile_size': 224},
                'standard': {'auto_adapt': True, 'fp16': True, 'tile_size': 224},
                'flexible': {'auto_adapt': True, 'fp16': False, 'tile_size': 320}
            },
            'fast_mamba_vsr': {
                'strict': {'chunk_size': 8, 'overlap': 1, 'fp16': True},
                'standard': {'chunk_size': 16, 'overlap': 2, 'fp16': True},
                'flexible': {'chunk_size': 24, 'overlap': 3, 'fp16': False}
            }
        }
        
        return configs.get(model_name, {}).get(latency_class, {})
    
    def _determine_processing_order(self, routing: Dict) -> List[str]:
        """Determine optimal processing order following video enhancement best practices"""
        processing_order = []
        
        # Phase 1: Artifact cleanup (order matters!)
        if routing['use_compression_expert']:
            processing_order.append('compression_cleanup')
        
        if routing['use_denoising']:
            processing_order.append('denoising')
        
        if routing.get('use_deblur_expert', False):
            processing_order.append('deblur_preprocessing')
        
        if routing['use_low_light_expert']:
            processing_order.append('low_light_enhancement')
        
        # Phase 2: SOTA model backbone restoration
        primary_model = routing.get('primary_model', 'vsrm')
        processing_order.append(f'sota_{primary_model}_enhancement')
        
        # Phase 4: Specialized post-processing
        if routing['use_face_expert']:
            processing_order.append('face_restoration')
        
        # Phase 5: Temporal consistency
        if routing['use_temporal_consistency']:
            processing_order.append('temporal_consistency')
        
        # Phase 6: Final temporal enhancement
        if routing['use_hfr_interpolation']:
            processing_order.append('hfr_interpolation')
        
        return processing_order
    
    def _sample_frames(self, video_path: str, num_samples: int = 12) -> List[np.ndarray]:
        """Sample representative frames from video for analysis"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                logger.error("Video has no frames")
                cap.release()
                return []
            
            # Sample frames at regular intervals
            frame_indices = np.linspace(0, total_frames-1, num_samples, dtype=int)
            frames = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                else:
                    logger.warning(f"Failed to read frame at index {idx}")
            
            cap.release()
            logger.info(f"Successfully sampled {len(frames)} frames from {total_frames} total frames")
            
            return frames
            
        except Exception as e:
            logger.error(f"Error sampling frames: {e}")
            return []
    
    def _calculate_routing_confidence(self, degradations: Dict, content: Dict) -> float:
        """Calculate confidence score for the routing decision"""
        # Base confidence on clarity of degradation detection
        degradation_clarity = []
        
        for degradation, score in degradations.items():
            # High confidence when score is clearly above or below threshold
            if degradation in self.thresholds:
                threshold = self.thresholds[degradation]
                clarity = abs(score - threshold) / threshold
                degradation_clarity.append(min(clarity, 1.0))
        
        base_confidence = np.mean(degradation_clarity) if degradation_clarity else 0.5
        
        # Boost confidence if faces are clearly detected
        if content['has_faces'] and content['face_prominence'] > 0.05:
            base_confidence = min(1.0, base_confidence + 0.1)
        
        return base_confidence
    
    def _get_fallback_routing(self) -> Dict:
        """Fallback routing plan when analysis fails"""
        logger.warning("⚠️ Using fallback routing plan")
        
        return {
            'degradations': {
                'compression_artifacts': 0.7,  # Assume moderate compression
                'motion_blur': 0.3,
                'low_light': 0.2,
                'noise': 0.3,
                'temporal_inconsistency': 0.4
            },
            'content_analysis': {
                'has_faces': False,
                'face_prominence': 0.0,
                'motion_complexity': 0.5,
                'scene_changes': 2
            },
            'expert_routing': {
                'primary_model': 'vsrm',  # Default SOTA fallback
                'model_config': {'window': 7, 'stride': 4, 'fp16': True, 'tile_size': 512},
                'fallback_model': 'fast_mamba_vsr',
                'use_deblur_expert': False,
                'use_compression_expert': True,
                'use_denoising': False,
                'use_low_light_expert': False,
                'use_face_expert': False,
                'use_hfr_interpolation': True,
                'use_temporal_consistency': True,
                'latency_class': 'standard',
                'allow_diffusion': True,
                'allow_zero_shot': True
            },
            'processing_order': [
                'compression_cleanup', 
                'sota_vsrm_enhancement', 
                'temporal_consistency', 
                'hfr_interpolation'
            ],
            'confidence_score': 0.5
        }
    
    def _log_routing_decision(self, result: Dict):
        """Log the routing decision for debugging and optimization"""
        logger.info("📋 Expert Routing Decision:")
        logger.info(f"   Confidence: {result['confidence_score']:.2f}")
        
        degradations = result['degradations']
        logger.info("🔍 Detected Degradations:")
        for deg, score in degradations.items():
            status = "HIGH" if score > 0.6 else "MED" if score > 0.3 else "LOW"
            logger.info(f"   {deg}: {score:.3f} ({status})")
        
        content = result['content_analysis']
        logger.info("📊 Content Analysis:")
        logger.info(f"   Faces detected: {content['has_faces']}")
        logger.info(f"   Face prominence: {content['face_prominence']:.3f}")
        logger.info(f"   Motion complexity: {content['motion_complexity']:.3f}")
        
        logger.info("🗺️ Processing Pipeline:")
        for i, step in enumerate(result['processing_order'], 1):
            logger.info(f"   {i}. {step}")


if __name__ == "__main__":
    # Test the degradation router
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python degradation_router.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    router = DegradationRouter()
    result = router.analyze_and_route(video_path)
    
    print("\n🎬 Degradation Analysis Complete!")
    print(f"Confidence: {result['confidence_score']:.2f}")
    print(f"Processing steps: {len(result['processing_order'])}")