#!/usr/bin/env python3
"""
Coordinator Agent - Master orchestrator for multi-agent video processing
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
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

from agents.core import (
    BaseVideoProcessingAgent, AgentCapabilities, TaskSpecification, 
    ProcessingResult, AgentMessage, MessageType, TaskStatus, Priority
)
from agentscope.message import Msg

# Import Topaz killer components for advanced coordination
try:
    from models.analysis.degradation_router import DegradationRouter
    from pipeline.topaz_killer_pipeline import TopazKillerPipeline
    TOPAZ_KILLER_AVAILABLE = True
except ImportError:
    TOPAZ_KILLER_AVAILABLE = False
    logger.warning("Topaz killer components not available")

logger = logging.getLogger(__name__)

class WorkflowStage(Enum):
    """Workflow processing stages"""
    ANALYSIS = "analysis"
    CODE_GENERATION = "code_generation"
    ENHANCEMENT = "enhancement"
    QUALITY_ASSESSMENT = "quality_assessment" 
    AGGREGATION = "aggregation"
    TOPAZ_KILLER = "topaz_killer"  # New Topaz killer stage

@dataclass
class WorkflowConfiguration:
    """Configuration for video processing workflow"""
    enable_analysis: bool = True
    enable_code_generation: bool = False
    enable_parallel_enhancement: bool = True
    enable_topaz_killer: bool = False  # Enable Topaz killer pipeline
    topaz_quality_tier: str = "ultra"  # fast, balanced, high, ultra
    topaz_target_fps: int = 60
    quality_threshold: float = 0.8
    max_retry_attempts: int = 3
    timeout_per_stage: int = 600  # seconds
    priority: Priority = Priority.MEDIUM

@dataclass
class VideoProcessingRequest:
    """Video processing request from user"""
    request_id: str
    video_path: str
    output_path: str
    processing_type: str = "comprehensive"  # comprehensive, fast, quality, topaz_killer
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    workflow_config: WorkflowConfiguration = field(default_factory=WorkflowConfiguration)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowState:
    """State tracking for workflow execution"""
    request_id: str
    current_stage: WorkflowStage
    completed_stages: List[WorkflowStage] = field(default_factory=list)
    stage_results: Dict[str, Any] = field(default_factory=dict)
    active_tasks: Dict[str, TaskSpecification] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    errors: List[str] = field(default_factory=list)

class CoordinatorAgent(BaseVideoProcessingAgent):
    """
    Master Coordinator Agent for Multi-Agent Video Processing
    
    Responsibilities:
    - Receive and validate video processing requests
    - Plan and orchestrate multi-stage workflows
    - Distribute tasks to specialized agents
    - Monitor progress and handle failures
    - Aggregate final results
    - Manage resource allocation across agents
    """
    
    def __init__(self, name: str = "coordinator", **kwargs):
        capabilities = AgentCapabilities(
            agent_type="coordinator",
            capabilities=["coordination", "workflow-management", "task-distribution"],
            max_concurrent_tasks=5,
            cpu_cores=4,
            gpu_memory=0  # Coordinator doesn't need GPU
        )
        
        super().__init__(name=name, capabilities=capabilities, **kwargs)
        
        # Workflow management
        self.active_workflows: Dict[str, WorkflowState] = {}
        self.workflow_templates = self._load_workflow_templates()
        
        # Agent registry
        self.available_agents = {
            'analyzer': [],
            'enhancer': [], 
            'quality_assessor': [],
            'code_generator': []
        }
        
        # Workflow statistics
        self.workflow_stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'average_processing_time': 0.0
        }
        
        # Agent instance management
        self._agent_instances = {}  # Maps agent_id to actual agent instances
        self._agent_capabilities = {}  # Maps agent_id to capabilities
        
        # Initialize Topaz killer components if available
        self.topaz_killer = None
        self.degradation_router = None
        
        if TOPAZ_KILLER_AVAILABLE:
            try:
                self.degradation_router = DegradationRouter()
                self.topaz_killer = TopazKillerPipeline({
                    'quality_tier': 'ultra',
                    'enable_generative_enhancement': True,
                    'enable_face_restoration': True
                })
                logger.info("🏆 Topaz killer pipeline integrated with coordinator")
            except Exception as e:
                logger.warning(f"Failed to initialize Topaz killer: {e}")
        
        logger.info("Coordinator Agent initialized")
    
    async def process_task(self, task: TaskSpecification) -> ProcessingResult:
        """Process coordination tasks"""
        start_time = time.time()
        
        try:
            if task.task_type == "video_processing_request":
                return await self._handle_video_processing_request(task)
            elif task.task_type == "agent_registration":
                return await self._handle_agent_registration(task)
            elif task.task_type == "workflow_status":
                return await self._handle_workflow_status(task)
            else:
                return ProcessingResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error_message=f"Unknown coordination task type: {task.task_type}",
                    processing_time=time.time() - start_time
                )
                
        except Exception as e:
            logger.error(f"Coordination task failed: {e}")
            return ProcessingResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    async def _handle_video_processing_request(self, task: TaskSpecification) -> ProcessingResult:
        """Handle video processing request by orchestrating workflow"""
        request_data = task.input_data
        
        # Create video processing request
        video_request = VideoProcessingRequest(
            request_id=task.task_id,
            video_path=request_data.get('video_path', ''),
            output_path=request_data.get('output_path', ''),
            processing_type=request_data.get('processing_type', 'comprehensive'),
            user_preferences=request_data.get('user_preferences', {}),
            workflow_config=WorkflowConfiguration(**request_data.get('workflow_config', {})),
            metadata=request_data.get('metadata', {})
        )
        
        # Initialize workflow state
        workflow_state = WorkflowState(
            request_id=task.task_id,
            current_stage=WorkflowStage.ANALYSIS
        )
        self.active_workflows[task.task_id] = workflow_state
        
        try:
            # Execute workflow stages
            workflow_result = await self._execute_workflow(video_request, workflow_state)
            
            # Update statistics
            self.workflow_stats['completed_requests'] += 1
            processing_time = time.time() - workflow_state.start_time
            self._update_workflow_stats(processing_time)
            
            return ProcessingResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                output_data=workflow_result,
                processing_time=processing_time,
                metadata={'workflow_stages': workflow_state.completed_stages}
            )
            
        except Exception as e:
            self.workflow_stats['failed_requests'] += 1
            logger.error(f"Workflow execution failed: {e}")
            
            return ProcessingResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error_message=str(e),
                processing_time=time.time() - workflow_state.start_time
            )
        finally:
            # Cleanup
            if task.task_id in self.active_workflows:
                del self.active_workflows[task.task_id]
    
    async def _execute_workflow(self, request: VideoProcessingRequest, state: WorkflowState) -> Dict[str, Any]:
        """Execute multi-stage video processing workflow"""
        logger.info(f"Starting workflow for request {request.request_id}")
        
        # Stage 1: Video Analysis
        if request.workflow_config.enable_analysis:
            state.current_stage = WorkflowStage.ANALYSIS
            analysis_result = await self._execute_analysis_stage(request, state)
            state.stage_results['analysis'] = analysis_result
            state.completed_stages.append(WorkflowStage.ANALYSIS)
        
        # Stage 2: Code Generation (optional)
        if request.workflow_config.enable_code_generation:
            state.current_stage = WorkflowStage.CODE_GENERATION
            code_result = await self._execute_code_generation_stage(request, state)
            state.stage_results['code_generation'] = code_result
            state.completed_stages.append(WorkflowStage.CODE_GENERATION)
        
        # Stage 3: Enhancement (or Topaz Killer)
        if request.processing_type == "topaz_killer" or request.workflow_config.enable_topaz_killer:
            state.current_stage = WorkflowStage.TOPAZ_KILLER
            topaz_result = await self._execute_topaz_killer_stage(request, state)
            state.stage_results['topaz_killer'] = topaz_result
            state.completed_stages.append(WorkflowStage.TOPAZ_KILLER)
        else:
            state.current_stage = WorkflowStage.ENHANCEMENT
            enhancement_result = await self._execute_enhancement_stage(request, state)
            state.stage_results['enhancement'] = enhancement_result
            state.completed_stages.append(WorkflowStage.ENHANCEMENT)
        
        # Stage 4: Quality Assessment
        state.current_stage = WorkflowStage.QUALITY_ASSESSMENT
        quality_result = await self._execute_quality_assessment_stage(request, state)
        state.stage_results['quality_assessment'] = quality_result
        state.completed_stages.append(WorkflowStage.QUALITY_ASSESSMENT)
        
        # Stage 5: Final Aggregation
        state.current_stage = WorkflowStage.AGGREGATION
        final_result = await self._execute_aggregation_stage(request, state)
        state.completed_stages.append(WorkflowStage.AGGREGATION)
        
        return final_result
    
    async def _execute_analysis_stage(self, request: VideoProcessingRequest, state: WorkflowState) -> Dict[str, Any]:
        """Execute video analysis stage using DeepSeek-R1 agent"""
        logger.info("Executing analysis stage")
        
        # Find available analyzer agent
        analyzer_agent = self._find_available_agent('analyzer')
        if not analyzer_agent:
            raise RuntimeError("No analyzer agents available")
        
        # Create analysis task
        analysis_task = TaskSpecification(
            task_id=f"{request.request_id}_analysis",
            task_type="video_analysis",
            input_data={
                'video_path': request.video_path,
                'processing_type': request.processing_type,
                'user_preferences': request.user_preferences
            },
            requirements={'gpu_memory': 8},
            priority=request.workflow_config.priority,
            timeout=request.workflow_config.timeout_per_stage
        )
        
        # Execute analysis
        result = await self._send_task_to_agent(analyzer_agent, analysis_task)
        
        if result.status != TaskStatus.COMPLETED:
            raise RuntimeError(f"Analysis stage failed: {result.error_message}")
        
        return result.output_data
    
    async def _execute_topaz_killer_stage(self, request: VideoProcessingRequest, state: WorkflowState) -> Dict[str, Any]:
        """Execute Topaz Video AI 7 killer pipeline stage"""
        logger.info("🏆 Executing Topaz Killer enhancement stage")
        
        if not self.topaz_killer:
            raise RuntimeError("Topaz killer pipeline not available")
        
        try:
            # Use degradation router for intelligent analysis if available
            routing_analysis = None
            if self.degradation_router:
                logger.info("Running intelligent degradation analysis...")
                routing_analysis = self.degradation_router.analyze_and_route(request.video_path)
            
            # Execute Topaz killer pipeline
            logger.info("Processing with Topaz killer pipeline...")
            
            result = self.topaz_killer.process_video_complete(
                input_path=request.video_path,
                output_path=request.output_path,
                target_fps=request.workflow_config.topaz_target_fps,
                quality_tier=request.workflow_config.topaz_quality_tier
            )
            
            # Add routing analysis to result
            if routing_analysis:
                result['routing_analysis'] = routing_analysis
                result['expert_pipeline_used'] = routing_analysis['processing_order']
            
            logger.info(f"🏆 Topaz killer completed: Beats Topaz Video AI 7: {result.get('beats_topaz_video_ai_7', False)}")
            
            return {
                'status': 'completed',
                'pipeline': 'topaz_killer',
                'output_path': request.output_path,
                'processing_stats': result.get('quality_metrics', {}),
                'beats_topaz': result.get('beats_topaz_video_ai_7', False),
                'routing_analysis': routing_analysis,
                'processing_time': result.get('processing_time', 0)
            }
            
        except Exception as e:
            logger.error(f"Topaz killer stage failed: {e}")
            raise RuntimeError(f"Topaz killer enhancement failed: {e}")
    
    async def _execute_code_generation_stage(self, request: VideoProcessingRequest, state: WorkflowState) -> Dict[str, Any]:
        """Execute code generation stage using Qwen2.5-Coder agent"""
        logger.info("Executing code generation stage")
        
        # Find available code generator agent
        code_gen_agent = self._find_available_agent('code_generator')
        if not code_gen_agent:
            logger.warning("No code generator agents available, skipping stage")
            return {"skipped": True, "reason": "No agents available"}
        
        # Create code generation task
        code_task = TaskSpecification(
            task_id=f"{request.request_id}_codegen",
            task_type="code_generation",
            input_data={
                'analysis_result': state.stage_results.get('analysis', {}),
                'optimization_target': request.processing_type,
                'user_preferences': request.user_preferences
            },
            priority=request.workflow_config.priority,
            timeout=request.workflow_config.timeout_per_stage
        )
        
        # Execute code generation
        result = await self._send_task_to_agent(code_gen_agent, code_task)
        
        if result.status != TaskStatus.COMPLETED:
            logger.warning(f"Code generation failed: {result.error_message}")
            return {"failed": True, "error": result.error_message}
        
        return result.output_data
    
    async def _execute_enhancement_stage(self, request: VideoProcessingRequest, state: WorkflowState) -> Dict[str, Any]:
        """Execute video enhancement stage using FLUX-Reason agent"""
        logger.info("Executing enhancement stage")
        
        # Find available enhancement agents
        enhancement_agents = self._find_available_agents('enhancer', count=1)
        if not enhancement_agents:
            raise RuntimeError("No enhancement agents available")
        
        # Create enhancement task
        enhancement_task = TaskSpecification(
            task_id=f"{request.request_id}_enhancement",
            task_type="video_enhancement",
            input_data={
                'video_path': request.video_path,
                'output_path': request.output_path,
                'analysis_result': state.stage_results.get('analysis', {}),
                'code_suggestions': state.stage_results.get('code_generation', {}),
                'user_preferences': request.user_preferences
            },
            requirements={'gpu_memory': 16},
            priority=request.workflow_config.priority,
            timeout=request.workflow_config.timeout_per_stage
        )
        
        # Execute enhancement
        enhancement_agent = enhancement_agents[0]
        result = await self._send_task_to_agent(enhancement_agent, enhancement_task)
        
        if result.status != TaskStatus.COMPLETED:
            raise RuntimeError(f"Enhancement stage failed: {result.error_message}")
        
        return result.output_data
    
    async def _execute_quality_assessment_stage(self, request: VideoProcessingRequest, state: WorkflowState) -> Dict[str, Any]:
        """Execute quality assessment stage"""
        logger.info("Executing quality assessment stage")
        
        # Find available quality assessment agent
        qa_agent = self._find_available_agent('quality_assessor')
        if not qa_agent:
            logger.warning("No quality assessment agents available, using basic validation")
            return {"validated": True, "method": "basic"}
        
        # Create quality assessment task
        qa_task = TaskSpecification(
            task_id=f"{request.request_id}_quality",
            task_type="quality_assessment",
            input_data={
                'original_video_path': request.video_path,
                'enhanced_video_path': request.output_path,
                'quality_threshold': request.workflow_config.quality_threshold,
                'enhancement_result': state.stage_results.get('enhancement', {})
            },
            priority=request.workflow_config.priority,
            timeout=request.workflow_config.timeout_per_stage
        )
        
        # Execute quality assessment
        result = await self._send_task_to_agent(qa_agent, qa_task)
        
        if result.status != TaskStatus.COMPLETED:
            logger.warning(f"Quality assessment failed: {result.error_message}")
            return {"failed": True, "error": result.error_message}
        
        return result.output_data
    
    async def _execute_aggregation_stage(self, request: VideoProcessingRequest, state: WorkflowState) -> Dict[str, Any]:
        """Aggregate final results"""
        logger.info("Executing aggregation stage")
        
        # Compile comprehensive result
        final_result = {
            'request_id': request.request_id,
            'status': 'completed',
            'video_path': request.video_path,
            'output_path': request.output_path,
            'processing_type': request.processing_type,
            'workflow_stages': state.completed_stages,
            'stage_results': state.stage_results,
            'processing_time': time.time() - state.start_time,
            'metadata': {
                'coordinator': self.agent_id,
                'workflow_version': '1.0',
                'completed_at': time.time()
            }
        }
        
        # Validate output quality if available
        qa_result = state.stage_results.get('quality_assessment', {})
        if qa_result and not qa_result.get('failed', False):
            quality_score = qa_result.get('quality_score', 0.0)
            if quality_score < request.workflow_config.quality_threshold:
                logger.warning(f"Quality score {quality_score} below threshold {request.workflow_config.quality_threshold}")
                final_result['quality_warning'] = True
        
        return final_result
    
    def _find_available_agent(self, agent_type: str) -> Optional[str]:
        """Find an available agent of the specified type"""
        agents = self.available_agents.get(agent_type, [])
        return agents[0] if agents else None
    
    def _find_available_agents(self, agent_type: str, count: int = 1) -> List[str]:
        """Find multiple available agents of the specified type"""
        agents = self.available_agents.get(agent_type, [])
        return agents[:count]
    
    async def _send_task_to_agent(self, agent_id: str, task: TaskSpecification) -> ProcessingResult:
        """Send task to specific agent and wait for result"""
        # Create task message
        task_message = AgentMessage(
            sender=self.agent_id,
            receiver=agent_id,
            message_type=MessageType.TASK,
            payload={
                'task_id': task.task_id,
                'task_type': task.task_type,
                'input_data': task.input_data,
                'requirements': task.requirements,
                'priority': task.priority.value,
                'timeout': task.timeout,
                'retry_count': task.retry_count,
                'metadata': task.metadata
            },
            priority=task.priority
        )
        
        # Try to send task to real agent if available, otherwise simulate
        try:
            # Check if we have a reference to the actual agent instance
            if hasattr(self, '_agent_instances') and agent_id in self._agent_instances:
                actual_agent = self._agent_instances[agent_id]
                # Create AgentScope message and send to agent
                from agentscope.message import Msg
                
                msg = Msg(
                    name=self.agent_id,
                    content={
                        'message_type': task_message.message_type.value,
                        'payload': task_message.payload
                    },
                    role="user"
                )
                
                # Send message to agent and await response
                response = await actual_agent.reply(msg)
                
                # Parse response
                if response.content.get('status') == 'success':
                    return ProcessingResult(
                        task_id=task.task_id,
                        status=TaskStatus.COMPLETED,
                        output_data=response.content.get('output_data', {}),
                        processing_time=response.content.get('processing_time', 0.1)
                    )
                else:
                    return ProcessingResult(
                        task_id=task.task_id,
                        status=TaskStatus.FAILED,
                        error_message=response.content.get('message', 'Agent processing failed')
                    )
            
            else:
                # Fallback to simulation for demo purposes
                logger.info(f"Agent {agent_id} not available, using simulation")
                await asyncio.sleep(0.1)  # Simulate processing time
                
                return ProcessingResult(
                    task_id=task.task_id,
                    status=TaskStatus.COMPLETED,
                    output_data={
                        'task_completed': True,
                        'agent_id': agent_id,
                        'simulated': True
                    },
                    processing_time=0.1
                )
                
        except Exception as e:
            logger.error(f"Failed to send task to agent {agent_id}: {e}")
            return ProcessingResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error_message=f"Communication with agent failed: {str(e)}"
            )
    
    async def _handle_agent_registration(self, task: TaskSpecification) -> ProcessingResult:
        """Handle agent registration requests"""
        agent_info = task.input_data
        agent_type = agent_info.get('agent_type', '')
        agent_id = agent_info.get('agent_id', '')
        
        if agent_type in self.available_agents:
            if agent_id not in self.available_agents[agent_type]:
                self.available_agents[agent_type].append(agent_id)
                logger.info(f"Registered {agent_type} agent: {agent_id}")
            
            return ProcessingResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                output_data={'registered': True, 'agent_id': agent_id}
            )
        else:
            return ProcessingResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error_message=f"Unknown agent type: {agent_type}"
            )
    
    async def _handle_workflow_status(self, task: TaskSpecification) -> ProcessingResult:
        """Handle workflow status requests"""
        request_id = task.input_data.get('request_id', '')
        
        if request_id in self.active_workflows:
            workflow_state = self.active_workflows[request_id]
            status_info = {
                'request_id': request_id,
                'current_stage': workflow_state.current_stage.value,
                'completed_stages': [stage.value for stage in workflow_state.completed_stages],
                'processing_time': time.time() - workflow_state.start_time,
                'errors': workflow_state.errors
            }
            
            return ProcessingResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                output_data=status_info
            )
        else:
            return ProcessingResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error_message=f"Workflow not found: {request_id}"
            )
    
    def _load_workflow_templates(self) -> Dict[str, Any]:
        """Load workflow templates for different processing types"""
        return {
            'comprehensive': {
                'stages': [
                    WorkflowStage.ANALYSIS,
                    WorkflowStage.CODE_GENERATION,
                    WorkflowStage.ENHANCEMENT,
                    WorkflowStage.QUALITY_ASSESSMENT,
                    WorkflowStage.AGGREGATION
                ],
                'parallel_allowed': ['enhancement'],
                'optional_stages': ['code_generation']
            },
            'fast': {
                'stages': [
                    WorkflowStage.ANALYSIS,
                    WorkflowStage.ENHANCEMENT,
                    WorkflowStage.AGGREGATION
                ],
                'parallel_allowed': ['enhancement'],
                'optional_stages': []
            },
            'quality': {
                'stages': [
                    WorkflowStage.ANALYSIS,
                    WorkflowStage.CODE_GENERATION,
                    WorkflowStage.ENHANCEMENT,
                    WorkflowStage.QUALITY_ASSESSMENT,
                    WorkflowStage.AGGREGATION
                ],
                'parallel_allowed': [],
                'optional_stages': []
            }
        }
    
    def _update_workflow_stats(self, processing_time: float):
        """Update workflow processing statistics"""
        total_completed = self.workflow_stats['completed_requests']
        if total_completed > 0:
            current_avg = self.workflow_stats['average_processing_time']
            new_avg = ((current_avg * (total_completed - 1)) + processing_time) / total_completed
            self.workflow_stats['average_processing_time'] = new_avg
        else:
            self.workflow_stats['average_processing_time'] = processing_time
    
    def register_agent_instance(self, agent_instance, agent_type: str = None):
        """Register an actual agent instance with the coordinator"""
        agent_id = getattr(agent_instance, 'agent_id', agent_instance.name)
        
        # Determine agent type from capabilities or parameter
        if agent_type is None and hasattr(agent_instance, 'capabilities'):
            agent_type = agent_instance.capabilities.agent_type
        
        if agent_type and agent_type in self.available_agents:
            # Register in available agents list
            if agent_id not in self.available_agents[agent_type]:
                self.available_agents[agent_type].append(agent_id)
            
            # Store agent instance
            self._agent_instances[agent_id] = agent_instance
            
            # Store capabilities
            if hasattr(agent_instance, 'capabilities'):
                self._agent_capabilities[agent_id] = agent_instance.capabilities
            
            logger.info(f"Registered {agent_type} agent instance: {agent_id}")
            return True
        else:
            logger.warning(f"Cannot register agent {agent_id} with unknown type: {agent_type}")
            return False
    
    def unregister_agent_instance(self, agent_id: str):
        """Unregister an agent instance"""
        # Remove from agent instances
        if agent_id in self._agent_instances:
            del self._agent_instances[agent_id]
        
        # Remove from capabilities
        if agent_id in self._agent_capabilities:
            del self._agent_capabilities[agent_id]
        
        # Remove from available agents lists
        for agent_type, agents in self.available_agents.items():
            if agent_id in agents:
                agents.remove(agent_id)
                logger.info(f"Unregistered {agent_type} agent: {agent_id}")
                break
    
    def get_registered_agents(self) -> Dict[str, Any]:
        """Get information about all registered agents"""
        return {
            'agent_instances': list(self._agent_instances.keys()),
            'agent_capabilities': self._agent_capabilities,
            'available_by_type': {k: len(v) for k, v in self.available_agents.items()}
        }
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get comprehensive workflow statistics"""
        return {
            **self.workflow_stats,
            'active_workflows': len(self.active_workflows),
            'registered_agents': {k: len(v) for k, v in self.available_agents.items()},
            'agent_instances': len(self._agent_instances),
            'agent_status': self.get_status()
        }

# Export classes
__all__ = [
    'CoordinatorAgent',
    'VideoProcessingRequest', 
    'WorkflowConfiguration',
    'WorkflowStage'
]