"""
Task Specification module for defining video processing tasks and requirements.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


class TaskType(Enum):
    """Enumeration of supported video processing task types."""
    VIDEO_SUPER_RESOLUTION = "video_super_resolution"
    VIDEO_RESTORATION = "video_restoration"
    VIDEO_ENHANCEMENT = "video_enhancement"
    FACE_RESTORATION = "face_restoration"
    DENOISING = "denoising"
    DEBLURRING = "deblurring"
    INTERPOLATION = "interpolation"
    UPSCALING = "upscaling"


class Priority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class Quality(Enum):
    """Quality levels for processing."""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"
    MAXIMUM = "maximum"


@dataclass
class VideoSpecs:
    """Video specifications and requirements."""
    input_resolution: Tuple[int, int]
    target_resolution: Optional[Tuple[int, int]] = None
    fps: Optional[float] = None
    duration: Optional[float] = None
    codec: Optional[str] = None
    bitrate: Optional[str] = None
    has_faces: bool = False
    degradation_types: List[str] = None
    
    def __post_init__(self):
        if self.degradation_types is None:
            self.degradation_types = []


@dataclass
class ProcessingConstraints:
    """Processing constraints and resource limits."""
    max_memory_gb: Optional[float] = None
    max_processing_time: Optional[float] = None
    gpu_required: bool = True
    batch_size: Optional[int] = None
    model_precision: str = "fp16"  # fp16, fp32, int8
    tile_size: Optional[Tuple[int, int]] = None
    overlap: int = 32


@dataclass
class TaskSpecification:
    """Complete specification for a video processing task."""
    
    # Core task definition
    task_id: str
    task_type: TaskType
    priority: Priority = Priority.NORMAL
    quality: Quality = Quality.BALANCED
    
    # Input/Output specifications  
    input_path: str
    output_path: str
    video_specs: VideoSpecs
    
    # Processing configuration
    processing_constraints: ProcessingConstraints
    model_preferences: Dict[str, Any] = None
    preprocessing_steps: List[str] = None
    postprocessing_steps: List[str] = None
    
    # Metadata
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values for optional fields."""
        if self.model_preferences is None:
            self.model_preferences = {}
        if self.preprocessing_steps is None:
            self.preprocessing_steps = []
        if self.postprocessing_steps is None:
            self.postprocessing_steps = []
        if self.metadata is None:
            self.metadata = {}
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the task specification.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Validate required fields
        if not self.task_id:
            errors.append("task_id is required")
        if not self.input_path:
            errors.append("input_path is required")
        if not self.output_path:
            errors.append("output_path is required")
            
        # Validate video specs
        if not self.video_specs.input_resolution:
            errors.append("input_resolution is required")
        elif len(self.video_specs.input_resolution) != 2:
            errors.append("input_resolution must be a tuple of (width, height)")
        elif any(dim <= 0 for dim in self.video_specs.input_resolution):
            errors.append("input_resolution dimensions must be positive")
            
        # Validate target resolution if specified
        if self.video_specs.target_resolution:
            if len(self.video_specs.target_resolution) != 2:
                errors.append("target_resolution must be a tuple of (width, height)")
            elif any(dim <= 0 for dim in self.video_specs.target_resolution):
                errors.append("target_resolution dimensions must be positive")
        
        # Validate processing constraints
        if self.processing_constraints.max_memory_gb and self.processing_constraints.max_memory_gb <= 0:
            errors.append("max_memory_gb must be positive")
        if self.processing_constraints.max_processing_time and self.processing_constraints.max_processing_time <= 0:
            errors.append("max_processing_time must be positive")
        if self.processing_constraints.batch_size and self.processing_constraints.batch_size <= 0:
            errors.append("batch_size must be positive")
            
        return len(errors) == 0, errors
    
    def get_scale_factor(self) -> Optional[float]:
        """Calculate scale factor if target resolution is specified."""
        if not self.video_specs.target_resolution:
            return None
        
        input_w, input_h = self.video_specs.input_resolution
        target_w, target_h = self.video_specs.target_resolution
        
        scale_w = target_w / input_w
        scale_h = target_h / input_h
        
        # Return the average scale factor
        return (scale_w + scale_h) / 2
    
    def requires_upscaling(self) -> bool:
        """Check if the task requires upscaling."""
        if not self.video_specs.target_resolution:
            return False
        
        scale_factor = self.get_scale_factor()
        return scale_factor and scale_factor > 1.0
    
    def estimate_complexity(self) -> str:
        """Estimate processing complexity based on specifications."""
        complexity_score = 0
        
        # Resolution-based complexity
        input_pixels = self.video_specs.input_resolution[0] * self.video_specs.input_resolution[1]
        if input_pixels > 3840 * 2160:  # 4K+
            complexity_score += 4
        elif input_pixels > 1920 * 1080:  # HD+
            complexity_score += 2
        else:
            complexity_score += 1
            
        # Scale factor complexity
        scale_factor = self.get_scale_factor()
        if scale_factor and scale_factor > 2:
            complexity_score += 2
        elif scale_factor and scale_factor > 1:
            complexity_score += 1
            
        # Task type complexity
        complex_tasks = [TaskType.VIDEO_RESTORATION, TaskType.VIDEO_ENHANCEMENT]
        if self.task_type in complex_tasks:
            complexity_score += 2
            
        # Quality complexity
        if self.quality == Quality.MAXIMUM:
            complexity_score += 2
        elif self.quality == Quality.HIGH_QUALITY:
            complexity_score += 1
            
        # Degradation complexity
        if len(self.video_specs.degradation_types) > 2:
            complexity_score += 1
            
        # Return complexity level
        if complexity_score <= 3:
            return "low"
        elif complexity_score <= 6:
            return "medium"
        elif complexity_score <= 9:
            return "high"
        else:
            return "very_high"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task specification to dictionary."""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type.value,
            'priority': self.priority.value,
            'quality': self.quality.value,
            'input_path': self.input_path,
            'output_path': self.output_path,
            'video_specs': {
                'input_resolution': self.video_specs.input_resolution,
                'target_resolution': self.video_specs.target_resolution,
                'fps': self.video_specs.fps,
                'duration': self.video_specs.duration,
                'codec': self.video_specs.codec,
                'bitrate': self.video_specs.bitrate,
                'has_faces': self.video_specs.has_faces,
                'degradation_types': self.video_specs.degradation_types
            },
            'processing_constraints': {
                'max_memory_gb': self.processing_constraints.max_memory_gb,
                'max_processing_time': self.processing_constraints.max_processing_time,
                'gpu_required': self.processing_constraints.gpu_required,
                'batch_size': self.processing_constraints.batch_size,
                'model_precision': self.processing_constraints.model_precision,
                'tile_size': self.processing_constraints.tile_size,
                'overlap': self.processing_constraints.overlap
            },
            'model_preferences': self.model_preferences,
            'preprocessing_steps': self.preprocessing_steps,
            'postprocessing_steps': self.postprocessing_steps,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'created_at': self.created_at,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskSpecification':
        """Create TaskSpecification from dictionary."""
        video_specs = VideoSpecs(**data['video_specs'])
        processing_constraints = ProcessingConstraints(**data['processing_constraints'])
        
        return cls(
            task_id=data['task_id'],
            task_type=TaskType(data['task_type']),
            priority=Priority(data['priority']),
            quality=Quality(data['quality']),
            input_path=data['input_path'],
            output_path=data['output_path'],
            video_specs=video_specs,
            processing_constraints=processing_constraints,
            model_preferences=data.get('model_preferences', {}),
            preprocessing_steps=data.get('preprocessing_steps', []),
            postprocessing_steps=data.get('postprocessing_steps', []),
            user_id=data.get('user_id'),
            session_id=data.get('session_id'),
            created_at=data.get('created_at'),
            metadata=data.get('metadata', {})
        )