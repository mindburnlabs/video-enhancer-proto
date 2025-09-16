#!/usr/bin/env python3
"""
Base Agent Classes for Multi-Agent Video Processing System
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


from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
import uuid
import time
import logging
import asyncio
import json
from datetime import datetime

# Try to import agentscope, use fallback if not available
try:
    import agentscope
    from agentscope.agent import AgentBase
    from agentscope.message import Msg
    AGENTSCOPE_AVAILABLE = True
except ImportError:
    # Fallback implementations
    class AgentBase:
        def __init__(self, name: str = "agent", **kwargs):
            self.name = name
    
    class Msg:
        def __init__(self, name: str = "", content: str = "", **kwargs):
            self.name = name
            self.content = content
    
    AGENTSCOPE_AVAILABLE = False

# Import TaskSpecification from separate module
from .task_specification import TaskSpecification as VideoTaskSpecification

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of messages exchanged between agents"""
    TASK = "task"
    STATUS = "status"
    RESULT = "result" 
    ERROR = "error"
    RESOURCE = "resource"
    HEARTBEAT = "heartbeat"
    SHUTDOWN = "shutdown"

class TaskStatus(Enum):
    """Status of tasks"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Priority(Enum):
    """Task priority levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AgentCapabilities:
    """Agent capabilities and resources"""
    agent_type: str
    capabilities: List[str]
    max_concurrent_tasks: int = 1
    cpu_cores: int = 1
    gpu_memory: int = 0  # GB
    available: bool = True
    specialized_models: List[str] = field(default_factory=list)

# TaskSpecification is now imported from task_specification.py as VideoTaskSpecification

@dataclass
class ProcessingResult:
    """Result of agent processing"""
    task_id: str
    status: TaskStatus
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    processing_time: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentMessage:
    """Base message structure for agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    sender: str = ""
    receiver: str = ""
    message_type: MessageType = MessageType.TASK
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.MEDIUM
    correlation_id: Optional[str] = None

class BaseVideoProcessingAgent(ABC):
    """
    Base class for all video processing agents
    
    Provides common functionality for:
    - Agent registration and discovery
    - Message handling and routing
    - Task management and queuing
    - Error handling and recovery
    - Resource monitoring
    """
    
    def __init__(self, name: str, capabilities: AgentCapabilities, **kwargs):
        self.name = name
        
        self.capabilities = capabilities
        self.agent_id = f"{capabilities.agent_type}_{name}_{uuid.uuid4().hex[:8]}"
        
        # Task management
        self.active_tasks: Dict[str, VideoTaskSpecification] = {}
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        
        # Performance metrics
        self.metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
        
        # Resource monitoring
        self.resource_usage = {
            'cpu_utilization': 0.0,
            'memory_usage': 0.0,
            'gpu_utilization': 0.0
        }
        
        # Communication
        self.message_handlers: Dict[MessageType, Callable] = {
            MessageType.TASK: self._handle_task_message,
            MessageType.STATUS: self._handle_status_message,
            MessageType.RESULT: self._handle_result_message,
            MessageType.ERROR: self._handle_error_message,
            MessageType.RESOURCE: self._handle_resource_message,
            MessageType.HEARTBEAT: self._handle_heartbeat_message,
        }
        
        # Agent registry for discovery
        self.known_agents: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized {self.agent_id} with capabilities: {capabilities.capabilities}")
    
    def reply(self, x: Msg) -> Msg:
        """Main message handling entry point"""
        try:
            # Parse incoming message
            agent_message = self._parse_message(x)
            
            # Route to appropriate handler
            handler = self.message_handlers.get(agent_message.message_type)
            if handler:
                response = handler(agent_message)
                return self._create_response_message(response, agent_message)
            else:
                logger.warning(f"No handler for message type: {agent_message.message_type}")
                return self._create_error_response(f"Unsupported message type", agent_message)
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return self._create_error_response(str(e), None)
    
    @abstractmethod
    async def process_task(self, task: VideoTaskSpecification) -> ProcessingResult:
        """Process a task - to be implemented by subclasses"""
        pass
    
    def _parse_message(self, x: Msg) -> AgentMessage:
        """Parse incoming message"""
        try:
            if isinstance(x.content, dict):
                content = x.content
            else:
                content = json.loads(x.content)
            
            return AgentMessage(
                id=content.get('id', str(uuid.uuid4())),
                timestamp=content.get('timestamp', datetime.now().isoformat()),
                sender=content.get('sender', x.name),
                receiver=self.agent_id,
                message_type=MessageType(content.get('message_type', 'task')),
                payload=content.get('payload', {}),
                priority=Priority(content.get('priority', 'medium')),
                correlation_id=content.get('correlation_id')
            )
        except Exception as e:
            logger.error(f"Failed to parse message: {e}")
            raise ValueError(f"Invalid message format: {e}")
    
    def _handle_task_message(self, message: AgentMessage) -> ProcessingResult:
        """Handle incoming task messages"""
        try:
            # Extract task specification
            task_data = message.payload
            
            # Handle both old and new task specification formats
            if 'video_specs' in task_data and 'processing_constraints' in task_data:
                # New VideoTaskSpecification format
                task = VideoTaskSpecification.from_dict(task_data)
            else:
                # Legacy format - convert to new format
                from .task_specification import VideoSpecs, ProcessingConstraints, TaskType, Priority as TaskPriority, Quality
                
                video_specs = VideoSpecs(
                    input_resolution=task_data.get('input_resolution', (1920, 1080)),
                    target_resolution=task_data.get('target_resolution'),
                    fps=task_data.get('fps'),
                    duration=task_data.get('duration'),
                    has_faces=task_data.get('has_faces', False),
                    degradation_types=task_data.get('degradation_types', [])
                )
                
                processing_constraints = ProcessingConstraints(
                    max_memory_gb=task_data.get('requirements', {}).get('gpu_memory'),
                    max_processing_time=task_data.get('timeout', 300),
                    gpu_required=task_data.get('requirements', {}).get('gpu_required', True),
                    batch_size=task_data.get('requirements', {}).get('batch_size')
                )
                
                task = VideoTaskSpecification(
                    task_id=task_data.get('task_id', str(uuid.uuid4())),
                    task_type=TaskType(task_data.get('task_type', 'video_enhancement')),
                    input_path=task_data.get('input_data', {}).get('input_path', ''),
                    output_path=task_data.get('input_data', {}).get('output_path', ''),
                    video_specs=video_specs,
                    processing_constraints=processing_constraints,
                    metadata=task_data.get('metadata', {})
                )
            
            # Check if we can handle this task
            if not self._can_handle_task(task):
                return ProcessingResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error_message=f"Agent {self.agent_id} cannot handle task type: {task.task_type}"
                )
            
            # Add to active tasks
            self.active_tasks[task.task_id] = task
            
            # Process task asynchronously
            try:
                # Check if event loop is running
                loop = asyncio.get_running_loop()
                # If loop is running, create task instead of using asyncio.run
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.process_task(task))
                    result = future.result()
            except RuntimeError:
                # No event loop running, safe to use asyncio.run
                result = asyncio.run(self.process_task(task))
            
            # Update metrics
            self._update_metrics(result)
            
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            return result
            
        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            return ProcessingResult(
                task_id=message.payload.get('task_id', 'unknown'),
                status=TaskStatus.FAILED,
                error_message=str(e)
            )
    
    def _handle_status_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle status requests"""
        return {
            'agent_id': self.agent_id,
            'status': 'active',
            'capabilities': self.capabilities.__dict__,
            'active_tasks': len(self.active_tasks),
            'metrics': self.metrics,
            'resource_usage': self.resource_usage
        }
    
    def _handle_result_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle result messages from other agents"""
        # Default implementation - can be overridden
        logger.info(f"Received result from {message.sender}")
        return {'acknowledged': True}
    
    def _handle_error_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle error messages"""
        logger.error(f"Error from {message.sender}: {message.payload}")
        return {'error_acknowledged': True}
    
    def _handle_resource_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle resource-related messages"""
        resource_type = message.payload.get('type', '')
        
        if resource_type == 'request':
            return self._handle_resource_request(message.payload)
        elif resource_type == 'allocation':
            return self._handle_resource_allocation(message.payload)
        else:
            return {'error': f'Unknown resource message type: {resource_type}'}
    
    def _handle_heartbeat_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle heartbeat/health check messages"""
        return {
            'agent_id': self.agent_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'active_tasks': len(self.active_tasks)
        }
    
    def _can_handle_task(self, task: VideoTaskSpecification) -> bool:
        """Check if agent can handle the given task"""
        # Check if task type is in capabilities
        task_capability_map = {
            'video_analysis': 'video-analysis',
            'video_enhancement': 'video-enhancement', 
            'quality_assessment': 'quality-assessment',
            'code_generation': 'code-generation',
            'coordination': 'coordination',
            'video_super_resolution': 'video-enhancement',
            'video_restoration': 'video-enhancement',
            'face_restoration': 'face-restoration',
            'denoising': 'video-enhancement',
            'deblurring': 'video-enhancement',
            'interpolation': 'video-enhancement',
            'upscaling': 'video-enhancement'
        }
        
        required_capability = task_capability_map.get(task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type))
        if required_capability and required_capability not in self.capabilities.capabilities:
            return False
        
        # Check resource requirements from processing constraints
        if hasattr(task, 'processing_constraints'):
            if task.processing_constraints.max_memory_gb and task.processing_constraints.max_memory_gb > self.capabilities.gpu_memory:
                return False
        
        # Check concurrent task limit
        if len(self.active_tasks) >= self.capabilities.max_concurrent_tasks:
            return False
        
        return True
    
    def _update_metrics(self, result: ProcessingResult):
        """Update agent performance metrics"""
        if result.status == TaskStatus.COMPLETED:
            self.metrics['tasks_completed'] += 1
            self.completed_tasks.append(result.task_id)
        elif result.status == TaskStatus.FAILED:
            self.metrics['tasks_failed'] += 1
            self.failed_tasks.append(result.task_id)
        
        # Update processing time metrics
        self.metrics['total_processing_time'] += result.processing_time
        total_tasks = self.metrics['tasks_completed'] + self.metrics['tasks_failed']
        if total_tasks > 0:
            self.metrics['average_processing_time'] = (
                self.metrics['total_processing_time'] / total_tasks
            )
    
    def _handle_resource_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resource allocation requests"""
        requested_resources = payload.get('resources', {})
        
        # Simple resource availability check
        available_resources = {
            'cpu_cores': max(0, self.capabilities.cpu_cores - len(self.active_tasks)),
            'gpu_memory': self.capabilities.gpu_memory,  # Simplified
            'available': self.capabilities.available and len(self.active_tasks) < self.capabilities.max_concurrent_tasks
        }
        
        return {
            'agent_id': self.agent_id,
            'requested': requested_resources,
            'available': available_resources,
            'can_fulfill': all(
                available_resources.get(k, 0) >= v 
                for k, v in requested_resources.items()
                if k != 'available'
            )
        }
    
    def _handle_resource_allocation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resource allocation notifications"""
        allocated_resources = payload.get('allocated', {})
        logger.info(f"Received resource allocation: {allocated_resources}")
        return {'allocation_acknowledged': True}
    
    def _create_response_message(self, response_data: Any, original_message: AgentMessage) -> Msg:
        """Create response message"""
        response = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'sender': self.agent_id,
            'receiver': original_message.sender,
            'message_type': MessageType.RESULT.value,
            'payload': response_data if isinstance(response_data, dict) else {'data': response_data},
            'correlation_id': original_message.id
        }
        
        return Msg(
            name=self.agent_id,
            content=response,
            role="assistant"
        )
    
    def _create_error_response(self, error_message: str, original_message: Optional[AgentMessage]) -> Msg:
        """Create error response message"""
        response = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'sender': self.agent_id,
            'receiver': original_message.sender if original_message else 'unknown',
            'message_type': MessageType.ERROR.value,
            'payload': {'error': error_message},
            'correlation_id': original_message.id if original_message else None
        }
        
        return Msg(
            name=self.agent_id,
            content=response,
            role="assistant"
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.capabilities.agent_type,
            'status': 'active',
            'capabilities': self.capabilities.capabilities,
            'active_tasks': list(self.active_tasks.keys()),
            'metrics': self.metrics,
            'resource_usage': self.resource_usage
        }

# Export classes
__all__ = [
    'BaseVideoProcessingAgent',
    'AgentCapabilities', 
    'VideoTaskSpecification',  # Updated name
    'ProcessingResult',
    'AgentMessage',
    'MessageType',
    'TaskStatus', 
    'Priority'
]
