"""
Core Agent Classes and Communication Infrastructure
"""

from .base_agent import (
    BaseVideoProcessingAgent,
    AgentCapabilities,
    VideoTaskSpecification,
    ProcessingResult,
    AgentMessage,
    MessageType,
    TaskStatus,
    Priority
)

from .task_specification import (
    TaskSpecification,
    TaskType,
    Quality,
    VideoSpecs,
    ProcessingConstraints,
    Priority as TaskPriority
)

__all__ = [
    'BaseVideoProcessingAgent',
    'AgentCapabilities',
    'TaskSpecification',
    'VideoTaskSpecification',
    'ProcessingResult', 
    'AgentMessage',
    'MessageType',
    'TaskStatus',
    'Priority',
    'TaskType',
    'Quality',
    'VideoSpecs',
    'ProcessingConstraints',
    'TaskPriority'
]
