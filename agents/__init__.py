"""
Video Enhancement Multi-Agent System

This module contains all the specialized agents for the video enhancement pipeline:
- CoordinatorAgent: Manages overall workflow and task distribution
- VideoAnalyzerAgent: Performs video analysis and quality assessment 
- VideoEnhancerAgent: Handles video enhancement using various models
- QualityAssessmentAgent: Evaluates enhancement results and provides feedback
"""

# Import available agents
try:
    from .base_agent import BaseVideoAgent
    _base_available = True
except ImportError:
    _base_available = False

try:
    from .coordinator_agent import CoordinatorAgent
    _coordinator_available = True
except ImportError:
    _coordinator_available = False

try:
    from .video_analyzer_agent import VideoAnalyzerAgent
    _analyzer_available = True
except ImportError:
    _analyzer_available = False

try:
    from .video_enhancer_agent import VideoEnhancerAgent
    _enhancer_available = True
except ImportError:
    _enhancer_available = False

try:
    from .quality_assessment_agent import QualityAssessmentAgent
    _quality_assessor_available = True
except ImportError:
    _quality_assessor_available = False

# Build __all__ list dynamically
__all__ = []
if _base_available:
    __all__.append("BaseVideoAgent")
if _coordinator_available:
    __all__.append("CoordinatorAgent")
if _analyzer_available:
    __all__.append("VideoAnalyzerAgent")
if _enhancer_available:
    __all__.append("VideoEnhancerAgent")
if _quality_assessor_available:
    __all__.append("QualityAssessmentAgent")

# Agent registry for dynamic agent creation
AGENT_REGISTRY = {}
if _coordinator_available:
    AGENT_REGISTRY["coordinator"] = CoordinatorAgent
if _analyzer_available:
    AGENT_REGISTRY["analyzer"] = VideoAnalyzerAgent
if _enhancer_available:
    AGENT_REGISTRY["enhancer"] = VideoEnhancerAgent
if _quality_assessor_available:
    AGENT_REGISTRY["quality_assessor"] = QualityAssessmentAgent

def create_agent(agent_type: str, **kwargs):
    """
    Factory function to create agents dynamically
    
    Args:
        agent_type: Type of agent to create ("coordinator", "analyzer", "enhancer", "quality_assessor")
        **kwargs: Additional arguments to pass to the agent constructor
        
    Returns:
        Agent instance
        
    Raises:
        ValueError: If agent_type is not supported
    """
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}. Supported types: {list(AGENT_REGISTRY.keys())}")
    
    agent_class = AGENT_REGISTRY[agent_type]
    return agent_class(**kwargs)

def get_available_agents():
    """Get list of available agent types"""
    return list(AGENT_REGISTRY.keys())